# 新增参数与优化器布局：合成单步重放草稿

时间：2026-09-09 07:52:46 UTC。状态：`design_only_not_executed`。本文件仅提出有界诊断；没有采集真实梯度、执行模型或优化器、启动训练或修改既有协议。本轮八组负 AUC 结果保持原判读，不能用下面尚未执行的机制检验解释或撤销。

目标是区分三个实现因素：新增梯度进入全局 clip；同桶 fresh 参数的 Q 初始化共用自动尺度；参数桶的顺序、大小和成员位置改变随机探针的分配。首选只做合成 optimizer-only replay，不需要真实模型、checkpoint、数据或训练梯度快照。

## 已核实的实现边界

- 本轮 runner 只向 PSGDKron 额外传 `preconditioner_update_probability=1.0`，其余沿用原默认值；`orig_shapes=None, merge_dims=False, split=False, multi_tensor=True`。普通非 FSDP 参数没有 owner 分隔。见 `run_readout_retrieval.py:54,108`、HeavyBall `__init__.py:1434,1465`、`chainable.py:2053,2444`。
- `BucketGuard` 按 shape/dtype/device/owner 分桶，再区分 fresh/ready；shared 的 `[1]` 与活跃输出 bias `[1]` 可同桶，task 的 `[4]` 不与这些 `[1]` 同桶。见 `chainable.py:568–603`。
- Q 的首维是成员批次，不是跨参数协方差轴；更新及施用保留这一批次维。可是初始化的 `mean_root` 会把整个 fresh slab 展平，产生各成员共用的标量尺度。见 `utils.py:2494–2552,2611–2649,3118,3612`。因此“逐成员 Q”不能表述为“优化全过程完全独立”。
- `dampen_grad` 使用 `torch.randn_like`，默认路径用其 V 更新 Q；更新概率 1 本身采用累计规则，不是在随机决定是否更新。见 `chainable.py:1595–1609`、`utils.py:2819,3558`。不同编译实现的随机数消费粒度不可凭张量元素数推断。

## 固定输入与成本

第一阶段只用 CPU float32、4 个 Torch/OMP/MKL 线程、原版本 HeavyBall。为避免此前 CPU Inductor 环境错误，独立进程内临时使用该库已有 eager 开关并在 finally 恢复；这必须写入 receipt，不能称作正式 CUDA compiled 路径复现。禁止修改库源码、优化器超参数和原训练文件。保留 PSGDKron 的默认动量、Q 设置和更新裁剪，学习率 `1e-4`、weight decay 0、外部 dense clip 1、更新概率 1；没有 sparse 参数或 Adagrad 更新。

旧参数按以下固定顺序建立：`pre.weight:[2,3]`、`work.weight:[2,2]`、四个 `tower.{0..3}.bias:[1]`。它们分别编号 j=0..5，参数内部按连续扁平索引 k 从 0 开始，显式构造 `p[j,k]=0.01*(j+1)+0.001*k`、`g[j,k]=0.25+0.05*j+0.01*k`，然后转 float32。小矩阵代表不同形状的旧桶，四个 `[1]` 与真实输出 bias 形状一致；不声称这些数值代表真实梯度分布。

两个候选都在 `pre.weight` 与 `work.weight` 之间插入新参数。shared 为全零 `[1]`，指定 raw grad `[4]`；task 为全零 `[4]`，指定 raw grad `[1,1,1,1]`。二者在实数下满足 shared 梯度为 task 梯度和，专用于展示参数化与裁剪的区别，不冒充真实反传结果。所有输入由公式生成，不用随机数据生成、不调用 manual_seed、不增加训练 seed。

构造完成后保存旧 weights、raw gradients、参数顺序、全部初始 optimizer state（空状态及原 param-group 设置）和当前 CPU Torch RNG 快照。每格恢复同一快照，只在隔离副本上做一次 step；原始快照只读并核验 dtype/shape/bytes SHA256。全过程 fork/restore 调用前 RNG，记录实际 RNG bytes/hash，使重放可审核；RNG 快照身份不是新训练 seed。Python/NumPy RNG 若导入也只核验不变。

总上限：**12 次单步 optimizer replay，0 forward、0 backward、0 training update、0 GPU、0 数据读取**。墙钟 120 秒、累计 CPU 480 秒、RAM 2 GiB、产物 64 MiB；到限立即停止并保存已完成证据，不自动重试、不扩展形状/梯度扫描。12 次副本参数更新必须单独计为 `optimizer_replay_steps`，不能混写成没有执行优化器。构建或 eager 路径异常同样终止。

## 对照：最多 12 次单步

| 次数 | 条件 | 用途 |
|---:|---|---|
| 2 | 仅旧参数 baseline：一次无观测包装、一次只读观测包装 | 输出参数、旧状态与结束 RNG 必须 bitwise 相同；第二次记录每个旧参数的初始化 Q 和 V |
| 8 | shared 布局的 C/S/R 全部 2³ 组合，含 `000` 原生合成候选和 `111` 三项固定 | 不按正负或效应大小筛选；展示主效应及相互依赖，不预设相加 |
| 2 | task 布局 `000`、`111` | 最小形状对照；不将这两格称为 task 的完整因素消融 |

三个位的定义均针对**旧参数**，不移动候选桶、不删新参数、不改变旧 raw grad；干预仅作用于独立诊断副本，不是正式优化器配置：

- **C=0**：使用候选全部梯度的原生 clip。**C=1**：旧参数使用 baseline 的实际 clipped gradients；新增参数仍使用该候选 `000` 的 clipped gradient。这样固定旧参数的 clip 输入，不把新增梯度改动再混入此位。记录真实 clip 返回范数及逐参数结果；不能只用手写系数公式代替原调用。
- **S=0**：保留候选 slab 原生自动初始化。**S=1**：先按原路径完成初始化，再仅把每个旧成员的初始 Q slice 替换为 baseline 对应成员的初始 Q；新增成员维持其原生初始化。保持当前桶形状、成员顺序及其他状态不变。必须在 Q 第一次更新前捕获/替换，并验证全部旧 slice、dtype 和形状 exact。不能给整个混合桶强设一个 baseline scale，因为那也会直接改变新成员。S 固定的是共享尺度影响旧 Q 的通路，不是改成全局手工 `precond_init_scale`。
- **R=0**：保留该布局实际产生的 V。**R=1**：原生 `dampen_grad` 仍按该布局调用并消费 RNG，随后只用 baseline 按稳定参数名保存的 V 替换旧成员 V；新增成员保持候选原生 V。以本格实际输入 g 和原 dampening 重新计算对应 `g+(damp+eps*abs(g))*V`，不能直接拷贝 baseline 的扰动后 g，从而误删 C 的效应。核验旧成员 V exact、调用次数与成员映射完整。R 固定的是旧成员得到的探针，**不保证各布局 step 结束的全局 RNG 状态相同**。

所有包装都必须在 finally 恢复，记录捕获次数、桶签名及 fresh/ready 状态；缺失映射、额外调用、形状漂移、非有限值均判 invalid。Q 的状态逐成员读写，不手写 PSGD 更新公式来替代被测优化器。若现有接口无法实现这些局部干预，停止并报告该因素不可辨识，不能把未执行位记为通过。

## 可观测量与判读

每格保存输入/source/RNG 身份，参数顺序及 bucket/fresh 成员，global raw norm 与实际 clip 结果，初始化函数收到的 slab 梯度，自动 scale，逐旧成员 Q 初值、更新后 Q/动量状态，V 与扰动后 g，以及实际 `p_after-p_before`。合成数组很小，可以完整保存，不只保存均值。

每个旧参数均报告对 baseline 的 bitwise equality、最大绝对差、非零差元素数、update 差的 L2 与 `||Δu||₂/max(||u_baseline||₂,1e-12)`；余弦只在两边更新范数非零时描述。新增参数另报 finite/非零更新及是否在优化器覆盖内。不能从单个标量总误差反推具体通路。

按 C 的配对差检查 clip 输入，按 S 的配对差检查 Q 初值，按 R 的配对差检查旧 V，再比较它们在各因素背景下对最终 update 的影响。clip 系数或 Q 初值改变而最终 update 近似抵消，也要如实记为“中间量改变，末端作用小/未检出”。若 `111` 仍不能恢复旧 update/state，列为未解释残差，排查未控制的状态、批量内核或其他裁剪行为；不强迫三机制解释全部差异。

两次 baseline 必须 exact，否则先判重放/观测不可靠，剩余格不执行。正确执行却没有差异属于 null；只有个别末位差异属于数值差异，不能自动升级为值得新增训练的理由。该固定合成输入没有抽样置信区间、统计显著性或 AUC 估计。

## 编译 RNG 与可选真实检查

合成 eager 结果只能证明该实现路径存在可观测机制。若之后单独研究正式 compiled 路径，先核验原生重复和无扰动观测包装一致，记录 PyTorch/HeavyBall/编译产物、设备和 RNG 身份；不得假设 CPU eager、CUDA eager 和 Inductor 使用相同随机序列。

若 compiled 路径不能稳定复现同一快照、不能捕获每个 V，或插桩导致重新编译而改变原生输出/RNG，原生 RNG 因果项记为 `unresolved_compiled_rng`。可以保留已验证的 clip/scale 证据，但不能以 eager 结果替代，也不能换 seed、禁用编译后声称正式路径已复现。到预定成本即停止；不自动启动 GPU 检查。

真实 q8 首步仅是满足下面条件后的**可选**后续。当前没有完整旧梯度快照，不能由 smoke 范数日志或最终 checkpoint 补造。若以后选择该检查，需另行固定同一原训练首 batch、初始化 weights、未裁剪梯度、空 optimizer state、RNG 与实际 compiled 身份，并首先证明旧 raw grad exact；只复用本轮既有 seed，不增加 seed。真实检查须另定最小对照和预算，当前文档不授权采集、forward/backward 或 GPU 执行。

## 何时才值得考虑 optimizer-preserving 结构变体

进入独立实现/审查的最低证据是：baseline 重复与包装 exact；至少一项可直接观测的机制在固定输入下改变旧参数 update，且相应局部干预可逆地恢复该通路；`111` 恢复旧参数 update/state exact，同时新增参数仍有限、获得非零更新。若只有舍入末位差异、恢复失败或只有中间量变化，先停止，不因此安排新训练。

这只说明“构造可检验的隔离变体”有实现基础。若要以控制本轮训练混杂为理由执行正式结构变体，还需要可选真实 q8 检查证明该现象在本轮布局和原生路径上存在，并让变体在固定旧梯度条件下同时保留旧 clip 输入、Q/state 更新、探针及旧 RNG 轨迹；单纯新增一个参数组、关闭 compile 或把新梯度排除 global clip 都不能未经验证就称为 optimizer-preserving。随后由根重新明确对照、预算和是否执行，不自动训练。

一次 replay 只能识别给定输入和初态下的实现通路，不能证明多步梯度轨迹相同、任务获得了信息增量、最终 AUC 下降原因、哪种 readout 更好，或替八组负结果“免责”。即使后续变体保留初步优化行为，也仍须独立完成原定质量、逐任务比较和成本审核；既有候选阈值不变。

## 设计输入身份

以下路径相对 `external/home/lhr`，均为写草稿时仅读取的源文件；未读 NPZ、checkpoint 或数据。

| 文件 | SHA256 |
|---|---|
| `qformer-loop/taobao-transfer/recurrence-readout-retrieval/protocol.json` | `a50f0b55fc33d6ff7d1fb52aa122da478f85ee50b8bcaf762b7e535c47ca8e04` |
| `qformer-loop/taobao-transfer/run_readout_retrieval.py` | `d9f64074066696fb292bcb6811c37e5dc7aaeb8082d58ff55862627e73cc72d4` |
| `qformer-loop/taobao-transfer/optimizer-psgd/deps/heavyball/__init__.py` | `a23c16fd1f89a183f85c7411ea20b6246de35bc2b6f8eeb06d200ef792766331` |
| `qformer-loop/taobao-transfer/optimizer-psgd/deps/heavyball/chainable.py` | `d29401f55e517a4df3b93ca7a9e9b63e3e74b4d9cd8a17e8e09a37841990e527` |
| `qformer-loop/taobao-transfer/optimizer-psgd/deps/heavyball/utils.py` | `ade3484d1acd8a22b567b80789268ac4d4dd56dc347d5e26652e9fddfadf6f46` |
| `qformer-loop/runtime/UniRank/unirank/pytorch/models/rank_model.py` | `c729cd9d20b22a4a2f0120325ddff9a1528206519f514e5e02387e6b1b3dfb9f` |
