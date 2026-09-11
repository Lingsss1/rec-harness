# QFormerCross11 → depth-recurrent KuaiRec experiment

**比较口径已按用户修正：Cross8 为主基线，Cross11 为优化参考。**
与 Cross8 或 Cross11 接近的循环模型也保留为继续优化的候选；候选资格不等于已证明提升。
`cross8_original` 使用原始配置的默认 clip=10 和全 Query；`cross8_matched` 使用与现有
Cross11 实验相同的 clip=1 / 验证 Q3 置零，用来识别优化设置带来的差异。
旧实验 ID `baseline` 为保证记录可追溯保留原名，其实际模型是 Cross11，不能误称原始基线。

当前状态：循环模型和验证入口已实现，9 项正确性测试在 CPU 和 A800 GPU 上均通过；
官方数据已下载并完成预处理，第一轮 22 次完整训练的结果见 [RESULTS.md](RESULTS.md)。
第二轮使用 rec-harness 诊断并完成 6 次状态混合重训，见 [RESULTS_ROUND2.md](RESULTS_ROUND2.md)。
用户随后要求重点改变结构和循环位置；第三轮已比较五种位置并完成 9 次训练，
结果见 [RESULTS_ROUND3.md](RESULTS_ROUND3.md)，优先保留“历史读取后返回字段”结构。
第四轮已完成回读来源与归一化接口的四组 × 三 seed，共 12 次训练，累计 49 次。
优先候选 `feedback_context` 在历史读取后只回读用户/上下文，平均 AUC=0.869887，
较 Cross8 增加 0.000523；与优化基线的差距仍有不确定性，NDCG 低于上一轮字段反馈。
结果、成本与诊断见 [RESULTS_ROUND4.md](RESULTS_ROUND4.md)；本轮 CPU/GPU 各 9 项正确性测试通过。
第五轮已完成六种减层 / self-attention 结构 × 三 seed，共 18 次训练，累计 67 次。
优先保留 `compact_seq1`：2 个字段块 + 1 个历史块、共享循环，平均 AUC=0.870296，
参数 9.037M，前向延迟接近 Cross8；`attn_full` AUC=0.870442，保留为精度方向。
完整结果与诊断见 [RESULTS_ROUND5.md](RESULTS_ROUND5.md)；CPU/GPU 各 4 项新结构测试通过。
第六轮按要求不增加 seed，仅使用已有 20262027 完成首批八种结构及四个后续组合，累计 79 次训练。
同 seed 下，末尾上下文回读 AUC=0.870420；小模型在历史后加 attention 达到
AUC=0.870153、8.115M 参数、13.01 ms。比较与取舍见 [RESULTS_ROUND6.md](RESULTS_ROUND6.md)。
第七轮固定已有 seed=20262027，完成八种记忆 / 归一化 / attention 次数结构，累计 87 次训练。
最佳候选 `r7_small_attention_first` 保留 1 个字段块和 1 个历史块共享循环，额外 attention 只在第一次历史读取后运行。
同 seed AUC=0.871234，比 Cross8/Cross11 全 Query 分别高 0.001687/0.001261；参数 8.115M、前向 p50=12.52 ms。
CPU/GPU 各 5 项测试和 20 个诊断调用通过；结果与指标取舍见 [RESULTS_ROUND7.md](RESULTS_ROUND7.md)。
第八轮在同一已有 seed 上完成八组 attention 强度 / 归一化 / 循环次数对照，累计 95 次成功训练。
本轮未刷新最高 AUC，保留 `r7_small_attention_first`；一次/两次/三次循环从头训练的 AUC 分别为
0.870096/0.871234/0.868071。一次循环前向 8.53 ms，较两次少 32.7%，NDCG 更高，作为低成本排序候选保留。
CPU/GPU 各 5 项测试、17 个诊断调用通过；配置解析失败记录、完整结果与取舍见 [RESULTS_ROUND8.md](RESULTS_ROUND8.md)。
第九轮在已有 seed 上完成八组回读位置、分轮归一化与相同初值的独立历史 block 对照，累计 103 次训练。
当前最佳仍为共享交替结构（AUC=0.871234）；去掉第二次字段回读为 0.869712，独立历史权重为 0.870773。
结果及下一步循环子层方向见 [RESULTS_ROUND9.md](RESULTS_ROUND9.md)；CPU/GPU 各 5 项测试通过。
第十轮固定已有 seed，完成八组第二次字段/历史 block 选择性子层实验，累计 111 次完整训练。
本轮最高 AUC=0.869990，未超过完整共享 block 的 0.871234；精简前向为 11.47–12.55 ms。
CPU/GPU 各 5 项测试、18 个诊断调用通过；结果和循环轮次条件化方向见 [RESULTS_ROUND10.md](RESULTS_ROUND10.md)。
测试记录见 `reports/cpu-tests-middle.txt` 和 `reports/gpu-tests-middle.txt`；
这些是合成输入测试，推荐效果来自独立的完整验证集评测。
代码包历史 validation AUC 为 0.8877503059607915，checkpoint 未打包。
历史实验未包含处理后数据、切分生成脚本或标签构造规则。本次采用下述新协议，
不能与这个历史数字作等价比较，使用这次重新训练的 Cross8 和 Cross11 对照。

## 本次数据协议：kuairec-loop-v1

官方来源：[KuaiRec Zenodo](https://zenodo.org/records/18164998)，文件 `KuaiRec.zip`，
MD5 `261550d472c48eff4990fb13c0e5bcf7` 已校验。预处理入口 `prepare_kuairec.py`。

- 使用完整 big_matrix，12,530,806 条交互，7176 个用户。
- high_watch 定义为 `watch_ratio > 2.0`，固定后不根据验证结果调阈值。
- 每个用户按时间排序切分 80/10/10；边界相同时间戳归入较早分区。
- 历史仅取严格早于目标的同用户交互，最多 100 条；当前目标的 action 固定为 0。
  验证后部样本可以读取已经发生的验证前部交互，这是滚动历史预测协议。
- 使用发布的用户快照和视频固有属性，不输入目标观看反馈或每日互动统计。
  用户快照的采集时点没有逐事件保证，因此这是使用静态侧信息的 warm-start benchmark，
  不能表述为严格在线因果评测。
- category_1/2 是原始 `item_categories.csv` 中前两个 tag，不是 caption 文件的层级分类。
  属性字典只在训练事件涉及的实体上拟合，padding=0，未见属性=1。
- 所有对照均设置 drop_last=False、num_workers=0；单进程加载用于适配容器共享内存限制。
  公开 loader 的顺序种子固定为 2026+epoch，模型 seed 改变初始化，数据顺序保持一致。
- 训练 10,022,544 条，验证 1,253,001 条，测试 1,255,261 条；测试不用于候选选择。

`reports/data-audit.json` 记录了从原始交互独立核对 57 位用户、798 个目标截断点、
114 条序列的结果。时间切分测试和按用户 bootstrap 的 AUC 算法测试另见 `tests/`。
真实 batch=8192 的 4 轮循环已通过反向传播检查，峰值显存约 54 GB。

## 设计依据

- [Universal Transformers](https://arxiv.org/abs/1807.03819)：沿深度共享 Transformer 参数。
- [Reasoning with Latent Thoughts](https://arxiv.org/abs/2502.17416)：比较 k 层循环 L 次与 kL 个独立层。
- [Huginn / Recurrent Depth](https://arxiv.org/abs/2502.05171)：通过潜空间循环增加计算深度。

这些论文提供循环结构的依据，没有证明它能提升 KuaiRec 排序。
本实现是针对 QFormer 的固定深度实验，不是论文模型的完整复现；未引入动态停止、
随机深度或截断反向传播。原版 CrossValue 的乘性交互也不等于共享参数的深度循环。

`Q0 = FieldStage(X)`；`Qr = Coreθ(Qr-1, S, mask)`；最后做一次原有 final norm、
Query 路由和预测塔。Core 是现有 3 层序列 QFormer；层间参数独立，轮间共享。
每轮 K/V 始终来自同一原始 S。第一阶段只运行一次。
可选 anchor 消融在第 2 轮起使用 `0.9 * Qr-1 + 0.1 * Q0` 作为输入。
第二批 `sequence_loop_scope=middle` 借鉴 Huginn 的 prelude/core/coda 分段：
序列第 1 层只运行一次，中间层循环，最后 1 层只运行一次。单轮仍精确退化为原版。
这是基于首批负结果提出的后续探索，不能描述为预先注册的全部实验。

模型通过 QFormerCross8 的 stage factory 接入，在 optimizer 创建前安装完整模块。
没有 forward 中动态创建参数或后替换模块。循环轮数是配置，不保存在 state_dict 中，
必须随 checkpoint 保存；单轮与原版有相同 state_dict 键和形状。

## 实验矩阵

| 名称 | 序列核心独立层数 | 循环次数 | 有效序列深度 | 用途 |
|---|---:|---:|---:|---|
| baseline | 3 | 1 | 3 | 在相同数据上重训 Cross11 优化参考 |
| loop1 | 3 | 1 | 3 | 兼容性对照 |
| loop2 | 3 | 2 | 6 | 整段循环候选，单 seed |
| loop4 | 3 | 4 | 12 | 深度消融 |
| loop2_anchor | 3 | 2 | 6 | 初始 Query 注入消融 |
| deep6 | 6 | 1 | 6 | 对应 loop2 的非共享深度对照 |
| deep12 | 12 | 1 | 12 | 对应 loop4 的非共享深度对照 |
| middle2 | 3 | 中间层 2 次 | 4 | 优先候选，保留独立输入/输出层 |
| middle4 | 3 | 中间层 4 次 | 6 | 与 deep6 对照有效深度 |

另外对 Cross8 原始设置及 Cross8 匹配优化设置分别运行 20262027/28/29 三个 seed。
上表的参数共享比较以 Cross11 为参照；Cross8 自身模块结构和参数量不同，报告使用实测值。

上表的 Cross11 系列都保留 2 层字段阶段、8 个 Query、256 维、max_len=100、原始优化器、
batch_size=8192、1 epoch、clip=1，以及训练全 Query / 验证 Q3 置零的 incumbent 路由。
torch.compile 为便于首轮诊断统一关闭，BF16 统一开启；基线也按相同设置重新训练。
深度对照具有相同核心层调用次数，不意味着测得耗时或总训练计算成本完全相同。

第一阶段固定 seed=20262027、训练样本和更新次数，只在完整 validation 上筛选。
接近 Cross8 或 Cross11 的模型均可保留为优化候选，不要求先超过 Cross11。
优先候选与 Cross8、Cross11 使用相同的 3 个 seed（20262027/28/29）配对复测，
报告 AUC 差值均值、各 seed 差值、logloss、UAUC、gAUC、NDCG(10)、参数量和耗时。
对小幅收益需要按用户分组的 paired bootstrap（需额外保存预测）及更多 seed，
不能把单次训练差值当作稳定收益。最终固定模型后才评估 test；本入口不读取 test。
增加循环会增加训练和推理成本；同更新次数对照之外，还需固定 GPU 时间的预算对照，
以及同 batch 的推理吞吐/p50/p95 延迟测量，才能判断部署价值。
若 Q3 置零掩盖新模型收益，后续单独增加全 Query 路由对照，双方采用相同调参预算。

## 文件与依赖

- `upstream/`：原代码包解压副本；没有修改源码。
- `model_zoo/QFormerCross11Loop.py`：新增模型，可复制到完整 UniRank 的 model_zoo，
  并在其 `__init__.py` 添加 `from .QFormerCross11Loop import QFormerCross11Loop`。
- `model_zoo/QFormerCross11Placement.py`：结构与循环位置实验，支持字段、序列首层/末层、
  历史后返回字段和字段—历史交替循环；保持 Cross11 参数键/形状不变。
- `model_zoo/QFormerCross11Feedback.py`：区分额外回读的目标物品或用户/上下文来源，
  并控制字段 final norm 的调用位置；初始字段编码继续读取全部特征。
- `model_zoo/QFormerCross11Compact.py`：减少独立字段/历史块，在循环内可增加共享
  Query self-attention；新增模块在创建优化器之前注册。
- `model_zoo/QFormerCross11Schedule.py`：在紧凑结构中调整字段/历史调用顺序，
  支持同调用次数的不共享历史层对照及历史后额外 self-attention。
- `model_zoo/QFormerCross11Memory.py`：在额外字段回读中保留可微的初始状态记忆，
  调整既有归一化和 attention 的位置；保持父模型参数数量不变。
- `runtime/UniRank`：公开框架，固定在 `90bb99a60724e975ec565836266fd86dcad0039d`。
  它不一定是历史实验使用的框架版本。运行器直接加载所需模型，避免不完整 bundle registry。
- `runtime/deps`：隔离安装的框架依赖，不修改现有 LLM 环境。
- `config/model_config.yaml`：从打包的 Base 和 KuaiRec 配置生成的完整实验配置。
- `tests/test_loop.py`：使用真实公开 UniRank 的合成正确性测试，不模拟框架。
- `reports/provenance.json`：代码包哈希、框架版本、历史指标出处及未复现状态。

本机 Python：`workspace/ai4ai-runtime/opd-native-env/bin/python`。
额外依赖为 scipy==1.15.3、scikit-learn==1.9.0、joblib==1.6.0、
threadpoolctl==3.6.0、h5py==3.16.0、polars==1.0.0、keras-preprocessing==1.1.2、
heavyball==3.2.0、narwhals==2.25.0。

```bash
cd workspace/qformer-loop
workspace/ai4ai-runtime/opd-native-env/bin/python tests/test_loop.py
```

第十一轮在已有 seed 上完成八组轮次向量 / Query 门控结构，累计 119 次训练。
历史向量 AUC=0.871217，接近父模型 0.871234；本轮未产生新最佳。CPU/GPU 各 5 项测试、18 个 rec-harness 调用及 2 个条件中和探针完成。
完整结果、取舍与下一步子层顺序方向见 [RESULTS_ROUND11.md](RESULTS_ROUND11.md)。

第十二轮在同一已有 seed 上完成八组完整循环 block 子层顺序对照，累计 127 次训练。
本轮最佳历史 Self→FFN→Cross AUC=0.870325，未超过父模型 0.871234；保留超过 Cross8/Cross11 的结构候选。
CPU/GPU 各 5 项测试、18 个诊断调用及结果审计通过；见 [RESULTS_ROUND12.md](RESULTS_ROUND12.md)。下一步为两轮之间的历史表示更新。

第十三轮完成八组 Query反馈/历史self-attention 源更新，累计 135 次训练，仍沿用已有 seed。
本轮 AUC 最佳 0.870695，未超过父模型 0.871234；两个零初始候选的历史 writer 诊断贡献很小。
CPU/GPU 各 5 项测试、18 个工具调用和 2 个 writer 探针完成；结果见 [RESULTS_ROUND13.md](RESULTS_ROUND13.md)。

第十四轮完成八组预算接近的门控 FFN/SiLU 对照，累计 143 次训练，未新增 seed。
本轮双位置 SwiGLU AUC=0.870201，UAUC/NDCG 略高于父模型，但 AUC 和推理速度未占优。
CPU/GPU 各 5 项测试、18 个诊断调用及用户配对 bootstrap 完成，见 [RESULTS_ROUND14.md](RESULTS_ROUND14.md)。

第十五轮完成八组普通GELU FFN宽度对照，累计151次训练。字段宽度512候选AUC=0.871053，参数7.853M，接近r7但尚无明显速度优势。
CPU/GPU各5项测试、18个诊断调用和用户配对bootstrap完成；见 [RESULTS_ROUND15.md](RESULTS_ROUND15.md)。没有新增训练seed。

第十六轮完成6组可变训练深度实验，累计157次训练；验证保持固定两轮，未增加seed。
AUC均未超过父模型；先1后3课程的NDCG=0.421731，但AUC=0.864855，保留为排序取舍。
CPU/GPU各5项测试、27个诊断调用和计算量只读重放通过；见 [RESULTS_ROUND16.md](RESULTS_ROUND16.md)。

## 复现实验

本次处理后的数据位于 `data/KuaiRec_Big_Watch_Action`，包含 feature map 和分块数据。
下列命令从头训练，不需要历史 checkpoint；默认不加 `--run` 时只检查配置和路径。

```bash
python run_validation.py --data-dir data/KuaiRec_Big_Watch_Action --experiment cross8_original
python run_validation.py --data-dir data/KuaiRec_Big_Watch_Action --experiment cross8_original --gpu 0 --run
python run_validation.py --data-dir data/KuaiRec_Big_Watch_Action --experiment baseline --gpu 0 --run
python run_validation.py --data-dir data/KuaiRec_Big_Watch_Action --experiment middle2 --gpu 0 --run
python run_validation.py --data-dir data/KuaiRec_Big_Watch_Action --experiment feedback_context --seed 20262027 --gpu 0 --run
python run_validation.py --data-dir data/KuaiRec_Big_Watch_Action --experiment compact_seq1 --seed 20262027 --gpu 0 --run
python run_validation.py --data-dir data/KuaiRec_Big_Watch_Action --experiment attn_full --seed 20262027 --gpu 0 --run
python run_validation.py --data-dir data/KuaiRec_Big_Watch_Action --experiment r6_field_after --seed 20262027 --gpu 0 --run
python run_validation.py --data-dir data/KuaiRec_Big_Watch_Action --experiment r6_small_attention_after --seed 20262027 --gpu 0 --run
python run_validation.py --data-dir data/KuaiRec_Big_Watch_Action --experiment r7_small_attention_first --seed 20262027 --gpu 0 --run
python run_validation.py --data-dir data/KuaiRec_Big_Watch_Action --experiment r8_first_loop1 --seed 20262027 --gpu 0 --run
python run_validation.py --data-dir data/KuaiRec_Big_Watch_Action --experiment r9_untied_clone --seed 20262027 --gpu 0 --run
python run_validation.py --data-dir data/KuaiRec_Big_Watch_Action --experiment r10_history_cross_ffn --seed 20262027 --gpu 0 --run
```

每次训练使用新目录，保存配置、输入文件 SHA-256、模型源码哈希、checkpoint 和
validation 结果。默认从头训练；当前没有 checkpoint warm-start 命令。
训练完成后 `result.json` 保存验证指标，`validation_predictions.npz` 保存逐样本预测；
`compare_runs.py` 可以做配对用户 bootstrap，`benchmark_inference.py` 测量相同真实 batch
上的纯模型推理耗时。该 bootstrap 条件于固定 checkpoint，不能代替多 seed 的训练波动分析。

第十七轮完成6组阶段顺序实验，累计163次训练。最佳AUC=0.870439，未超过r7；CPU/GPU各5项测试、18个诊断调用与审计通过。见 [RESULTS_ROUND17.md](RESULTS_ROUND17.md)。

第十八轮完成4组按执行深度独立最终LayerNorm对照，累计167次完整训练，未增加seed。最佳AUC=0.870213，相对匹配共享norm对照+0.000900，仍低于r7=0.871234；CPU/GPU各5项测试、18个诊断调用和结果审计通过。配置解析失败在训练前修复并保留记录。见 [RESULTS_ROUND18.md](RESULTS_ROUND18.md)。
