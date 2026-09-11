本目录用于判断已训练的查询修正通过哪条路径影响第二轮，而不是根据扰动结果直接宣布结构改进。候选尚未完成时只准备脚本；主线程负责选择已完成候选、执行诊断、调用 rec-harness 工具并归档 memory。

当前第二轮查询为 `q = b + δ`，其中 `b = (e + z1) / 2`。同一个 q 同时进入 CrossValue 的检索投影 `w_q`、显式乘积交互投影 `w_qi` 和工作层残差。本诊断只改变第二次 L2；保留历史和 mask、第一轮工作、第一轮预测及预测状态不反馈的结构。预测第二轮可以响应被改变的第二轮工作输出，这是干预的下游效应。

| 模式 | 检索输入 | 交互输入 | 残差输入 |
| --- | --- | --- | --- |
| original / identity | q | q | q |
| zero | b | b | b |
| shuffle | b + δ[perm] | b + δ[perm] | b + δ[perm] |
| drop_retrieval | b | q | q |
| drop_interaction | q | b | q |
| drop_residual | q | q | b |
| retrieval_only | q | b | b |

这里的 δ 是实际加法后的 `corrected_query - base_query`，因此包含原加法的浮点舍入。移除的是修正，不是清空整个查询、attention 或 residual。分路干预继续调用原 attention，仅替换对应投影的输入；额外归一化工作层和未知结构会明确拒绝。identity 使用原 forward，要求与 original 的四头概率和每个样本 BCE 均逐位相同。退出上下文，包括异常退出，恢复原方法和 hooks。

固定诊断协议：

- 输入为单个已有 `result.json` 的完整 v1 或 v2 quality run；拒绝 smoke、未完成预算或错误目录。保持原 seed `20262027`、两轮、入口权重 0.5，确认原训练 2,193 步。
- 复用 `../query-conditioning/reports/paired-query-parts/subset.npz` 中已冻结的 5,514 行、4,647 个用户、四标签和 `population_weight`，不重新抽样。逐一核验其分层计数与权重，再核验实际 validation 数据中的 `user_id`、`item_index` 对应的 archive `item_id` 和四个标签。
- CPU float32、4 个线程、eval、全部参数 `requires_grad=False`，不调用训练、反向或 optimizer step。构造器会创建原类型优化器供模型初始化，但不加载其状态，不执行步骤。重建同 seed 模型后、加载权重前 clone A/B，用来报告加载权重相对初始化的变化。
- 遍历原 8,192 行 loader payload，仅 collate 命中的样本，并在每个 payload 内按最多 128 个样本分组。该边界与之前 paired-query-parts 诊断一致。每组固定 `torch.Generator().manual_seed(20262027 + 已处理选中行数)`，生成同一 permutation 供 shuffle 使用。记录 donor 行号、不同用户比例和固定点比例。
- 开始前校验 manifest 中全部模型 source SHA、对应 runner SHA、optimizer factory SHA 和实际读取的 validation/feature metadata SHA。导入的模型模块必须属于已验证 source。结束后重查这些文件、checkpoint、subset 和 exporter/helper SHA，并验证内存完整 state hash 未改变。不读取 train 或 test 数据文件。

先运行默认的整体修正干预：

```bash
PYTHONDONTWRITEBYTECODE=1 workspace/ai4ai-runtime/opd-native-env/bin/python \
  qformer-loop/taobao-transfer/query-correction-diagnostics/export_routes.py \
  --run-dir /absolute/path/to/completed/run \
  --output-dir /absolute/path/to/new/diagnostic-output
```

需要进一步区分路径时，在一个新的输出目录运行：

```bash
PYTHONDONTWRITEBYTECODE=1 workspace/ai4ai-runtime/opd-native-env/bin/python \
  qformer-loop/taobao-transfer/query-correction-diagnostics/export_routes.py \
  --run-dir /absolute/path/to/completed/run \
  --output-dir /absolute/path/to/new/route-output \
  --modes identity,zero,shuffle,drop_retrieval,drop_interaction,drop_residual,retrieval_only
```

`--run-dir` 和 `--output-dir` 必填，输出目录必须不存在。即使自定义 modes 未写 identity，也会强制先加入。失败只产生失败证据，不生成完成标记；不得把部分 `paired.tsv` 当作完成结果。只在 `completion.json` 的状态和 SHA 都通过后分析。

产物及计算：

- `losses.npz`：与固定 subset 同序的行号、用户/物品 ID、四标签、IPW、shuffle donor 行号，各模式的逐样本四任务 BCE，以及逐样本 base/δ 范数和二者比值。
- `paired.tsv`：每模式、每任务、每用户一行，字段与既有 rec-harness `analyze_perturbations.py` 相同。若验证总体行数为 N，选中用户数为 U，每行样本权重为 `w_i=N_stratum/n_stratum`，则用户贡献为 `U/N * sum(w_i * BCE_i)`；用户贡献的均值即 IPW 总体平均 BCE。`paired-user-index.npz` 显式映射 step 到用户 ID。
- `audit.json`：A/B 有限性、范数、加载前初始化 SHA 和变化；base/δ 每样本 L2 范数；按每个 slot/channel 跨样本中心化的 δ 方差、总中心化方差及其占平方范数的比例。分别报告 subset 等权和总体 IPW 统计，避免把分层样本分布当作总体分布。还记录 identity exact、mask/权重不变、donor 不同用户比例和诊断训练更新数 0。
- `identities.json`、`protocol.json`、`completion.json`：代码/数据/权重身份、运行时、协议和完整产物 SHA。`progress.json` 仅用于进度；失败时有 `failure.json`。

主线程调用现有工具的格式：

```bash
workspace/ai4ai-runtime/opd-native-env/bin/python \
  rec-harness/model-diagnostic-toolkit/scripts/analyze_perturbations.py \
  /absolute/path/to/completed/diagnostic-output/paired.tsv \
  --output-dir /absolute/path/to/completed/diagnostic-output/toolkit \
  --block-size 1 --bootstrap 500 --seed 20262027
```

正的 loss delta 表示移除或打乱修正后损失上升。zero 检查已训练修正的整体作用，shuffle 检查修正与当前样本的对应关系，分路移除和 retrieval_only 检查作用经过哪条输入路径；路径之间可能交互，效果不能直接相加。参数或 δ 非零只表示修正存在，不证明它有用；移除有害也不证明增加该分支会提高重新训练 AUC。

全部统计条件于固定分层样本、CPU 精度和一个已训练 checkpoint。工具的用户 bootstrap 不包含分层抽样设计不确定性、训练 seed 变化或候选选择偏差；500 次诊断重采样不是新增训练 seed。shuffle 可能保留同用户 donor 和固定点，分路移除也可能使激活离开训练分布。最终结构收益仍需固定训练协议的完整 validation AUC 对照，本 exporter 不计算或宣称 AUC 增益。
