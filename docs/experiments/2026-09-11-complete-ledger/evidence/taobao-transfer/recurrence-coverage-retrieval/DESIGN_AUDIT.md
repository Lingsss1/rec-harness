# 跨槽 coverage 第二检索：设计与独立公式审查

**这组实验有明确可证伪的问题，但没有已知的收益保证。** 它检验：保持两轮 no-feedback 和原参数布局时，把第一工作轮的跨槽参考 attention 概率变成第二工作轮的有限 score bias，能否改善完整验证预测；平均 coverage 的正向确认与负向抑制谁较好；排除对应自身槽是否优于包含它。此前方向检索八组没有形成候选，不能据旧 q4.buy 路径依赖预报本轮改善。[前轮结果](../recurrence-directional-retrieval-v2/RESULTS.md)、[本地查重](../recurrence-research/COVERAGE_RETRIEVAL_AUDIT.md)

此文与 `test_formula_reference.py` 只涉及数学、小型 CPU 合成张量及源码。不运行真实模型、checkpoint、数据、bootstrap、训练或 GPU kernel，不修改协议或 memory。CPU autograd 可能在内部探测 CUDA availability；这与运行 CUDA 张量/模型有别，测试报告单独保留该 runtime 警告。

## 固定设计和对照的解释范围

[协议](protocol.json) 固定 q4/q8 × mean/loo × sign±1，共八次独立训练；strength=1，PSGDKron/LR/clip1、两轮、原 seed20262027 与 2193 更新不变。第一工作读取是 `unified_qformer.layers[1]` 的第一次调用，不是入口层或预测层。P1 是对该次 q_norm 后 Q1、k_norm 后 K1 重算的参考分布；只在同一工作层的第二次 SDPA 加 bias。

| 预定比较 | 数量 | 能回答 | 不能直接归因 |
|---|---:|---|---|
| 八候选对各自原循环 | 8 | coverage 与其可微浮点 mask 执行路径的整体预测/成本是否有改进 | 不能单独把结果归为 coverage 数值，排除 kernel/浮点执行路径影响。 |
| 每个 parent、每种 aggregation 的 positive−negative | 4 | 同一公式/幅度/形状下正向确认相对负向抑制的完整训练差别 | 被关注位置是否有用、负向是否有信息论新颖性；两者都差时也不能挽救内部赢家。 |
| 每个 parent、每种 sign 的 loo−mean | 4 | 排除对应首轮槽的完整聚合方案是否较好 | 不是严格等幅干预，也没有 own-slot 同变换控制；不能证明跨槽信息独立有益。mean 与 loo 的广播 mask 形状也不同。 |

全部 16 个比较必须保留正、负、跨零区间与四任务/六项直接任务差。最多两个 best-single 条件配对沿原预定 per-parent 赢家规则；它们仍有选择条件，不能当作无选择的重复验证。原 500 用户重采样 seed20260906 不变。

`identity` 必须绕开全部 coverage/P1/浮点 mask 改写，维持原生执行路径。`format_zero` 必须先计算同图、同 dtype、同 shape 的有限 g，再用 `g*0` 保留 mask 的梯度路径，最后添加 padding mask；不能用不带梯度的 zeros_like 冒充同 format。该控制只校准局部数值/梯度差异，不证明训练等效。只有将来需要把优于原模型的收益单独归为 coverage 数值时，才有理由事先加入 format-zero 训练控制；本轮不因此自动扩矩阵。

## 公式及数学退化

逐 batch/head 令槽数 m 为 4 或 8，原有效历史集合为 V、长度 L；不沿 head 轴或任务轴聚合：

`P1[j,l] = softmax_valid(FP32(Q1[j])·FP32(K1[l])*scale)`

`Cmean[l] = mean_j P1[j,l]`

`Cloo[j,l] = mean_{i != j} P1[i,l]`

`g = tanh(L*C−1)`，`bias = sign*g`，`P2' = softmax_valid(S2+bias)`。

mean bias 为 `[B,H,1,Lmax]`，loo 为 `[B,H,m,Lmax]`；原 Q2/K2/V2、第二 `w_qi` 直达分支及外层 residual 输入不直接替换。它们后续值及训练梯度当然可能变化。查询槽不是四项任务的标签；任务头读取完整槽表示，不能将 q4 四个槽按点击/加购/收藏/购买命名。

- 非空支持上 C 为概率分布，`sum_l C=1`，因而 `L*C−1` 在有效位置的和为 0；tanh 之后不必和为 0。任何位置共同常数会被第二 softmax 消去，不需要为此发明额外归一化。
- 正向对高于均匀 coverage 的位置加分、低于均匀者减分；负向相反。数学上 `g∈[−tanh(1), tanh(L−1)]⊂[−1,1]`。sign± 的 bias 精确互为相反数，但 softmax、预测误差和重新训练的效果不会互为相反数。
- m 不进入总概率质量，避免 sum coverage 把 q8 的幅度机械翻倍；但这不等于 q4/q8 或 mean/loo 实际干预强度相同。`Cloo_j−Cmean=(Cmean−P1_j)/(m−1)`，对应 1/3 与 1/7；tanh 又进一步改变局部幅度。
- 一条位置对另一条的 odds 被乘以 `exp(sign*(g_l−g_k))`，上界小于 `exp(2)`。这只是有限重新加权，并非硬屏蔽“读过”的物品，也不是槽间无重复约束。
- `L=0`：先给 empty row 构造有限 softmax 输入，再使 P1 全零；bias 的 padding 位置为有限零，再与原 −inf mask 合成。不要先制造 NaN 再靠 where 遮盖。空历史下原 attention 聚合的零行为要保留，但其后线性 affine bias、interaction/residual/后续层**不能一并置零**。
- `L=1`：P1=1、C=1、g=0，应精确退化到零 bias。所有槽在有效位置均匀时 g 在实数下为 0；浮点 `1/L` 舍入可能给极小非零残差，应报告而不是暗改公式。
- 所有槽分布相同时，mean 与 loo 相同；此时没有额外跨槽分布信息。m=1 的 loo 必须明确拒绝，本轮 m=4/8 不遇此项。m=2 的 loo 只剩另一槽，但其 tanh 概率变换仍不等于 log-prior 的线性 Q 混合。
- 孔洞 mask 需要按有效位置计数，不能用 Lmax 或最后有效下标代替 L；None mask 等价全有效。无效 key 的概率、有限 bias 和经该参考路径的梯度应为零。

### 为什么不是旧 log-prior 代数重命名

在原共享 K/mask/scale 下，同槽 `α log P1` 等价 `Q2+αQ1` 的打分；mean/loo `mean(log P1)` 也等价相应 Q1 均值的线性混合。但本轮先平均 **P1** 再用 `tanh(L*C−1)`，通常不存在该固定线性混合恒等式。独立公式 fixture 在一维 key `[-1,0,1]` 上检查：mean(log P1) 的二阶差为零，而 log(mean P1) 与本轮 g 可有非零二阶差。这仅说明代数结构不同，不证明预测收益；特定退化输入仍可给零/相同 bias。

## 精度定义与前瞻 tiny CUDA 门槛

仅 `autocast(enabled=False)` 和 `.float()` 不能自动保证 CUDA 没有 TF32：输入/输出 dtype 与矩阵乘内部精度是两件事。本轮协议额外**要求** `cuda.matmul.allow_tf32=False`、`float32_matmul_precision='highest'` 并记录，原训练全局开关不由 helper 修改；不满足即停止，而不是静默改原协议。即使这些设置成立，也不应声称与 FP64、所有归约顺序或原 BF16 fused-SDPA 隐藏概率逐位等价。P1 是明确的 FP32 参考分布，不是已捕获的 native P1。

建议在任何实际 tiny CUDA 结果出现前固定如下校准口径；最终以根代理冻结的 CUDA 协议为执行标准，不用测试后调宽容差：

| 校准 | 预定建议 | 解释 |
|---|---|---|
| 原 stage 与 direct identity | 输出逐位 exact；梯度存在性一致，全部可比较输入/参数梯度逐位 exact | 证明绕过 coverage 时保留原路径；不能只用近似相等。 |
| 原生与对应 shape 的 mean-zero/loo-zero | 每输出 tensor forward NRMSE≤0.02；每个可比较输入/参数 tensor gradient NRMSE≤0.05 | 只接受小型本次数值校准，不能称 kernel 一致或训练等效。分别报告逐 tensor 和整体误差，不能只拼大向量掩盖局部误差。 |
| 两个独立 P1@V vs 原 SDPA 校准（q4/q8） | 仅预定 2% forward 门槛，比较范围明确到 attention 聚合；不增加 backward | 禁用 autocast 的参考与 native BF16 SDPA允许有量化/归约差异；梯度校准由已固定的 16 次 stage backward 覆盖，不把中间校准扩充成真实 shadow 或训练。 |

建议 NRMSE 固定为 `RMS(actual−reference)/max(RMS(reference),1e−6)`；当参考 RMS<1e−6 时单列绝对门槛 `max_abs_error≤1e−6`，避免近零梯度被分母或总体平均隐藏；全部比较另报 max_abs_error。参考应指原生或独立 oracle，而非挑更有利的分母。任何不满足都保留失败证据并审查，不临时改变门槛。

额外纯不变量：P1 FP32/finite，masked P1/bias=0、allpad=0，有效概率行和误差≤1e−6，L1 g=0，sign± 互反、g 在约定范围；原 Q2/K/V/interaction/residual 入口未被替换。真实 tiny CUDA 计划为 B4、D384/H4/d96/Lmax100、q4/q8，两者各 original/identity/mean-zero/loo-zero/mean±/loo± 共 16 stage forward/backward，加两组 P1@V 校准；四条合成行覆盖全空、单 key、孔洞、全有效。这里没有运行该 CUDA 计划。

## 梯度、成本与可解释性的限制

P1 与 C 不 detach。对一个第二槽的 bias，mean 把梯度传回所有首轮槽；loo 对其自身 P1 的偏导为零，但仍通过共享 K 和其他槽形成连接。这不意味着网络中该槽或共享 Wq 的总梯度为零。独立 CPU fixture 用显式 other-slot gather 的 NumPy oracle 和有限差分核对这条路径。

`tanh` 导数为 `L*sech²(L*C−1)`；对每个首槽 mean/loo 再分别乘 1/m 或 1/(m−1)。有界 bias 不等于原参数梯度不变或必然更小；稀疏覆盖可能饱和，接近均匀处反而响应较大。强度固定 1 不增加参数，不代表其实际梯度与之前 .25 旋转方案可按同一标量比较。

额外显式 QK/softmax/coverage 需要计算和反传缓存，并非只增加一条常数 bias。训练 B8192/H4/L100 时，一份完整 FP32 P 在 q4/q8 分别约 50/100 MiB；scores、softmax graph、第二浮点 mask 的梯度和 backend workspace 可能再增开销，不能用这一个数组估计总峰值。mean 输出可广播一槽、loo 为 m 槽，实际 mask/梯度 shape 与 kernel dispatch 也可能不同。需测量原控制及全部八组的真实同设备延迟、训练峰值和 fit 成本。

最后，首轮被关注不等于已无用；多个槽/头通过同一历史也可能提取不同信息。现有 accepted probe 的 overlap 是**同槽跨轮**指标，不是跨槽冗余。若本轮的负向 bias 获胜，只能在固定模型/训练/对照下描述该方向较好，不能推导所有已关注历史应被抑制；若正向更好，也不证明重复检索普遍必要。任何内部比较都要回到同 q 原循环和既定完整 macro/成本门槛，不按某任务的局部正点另宣布成功。
