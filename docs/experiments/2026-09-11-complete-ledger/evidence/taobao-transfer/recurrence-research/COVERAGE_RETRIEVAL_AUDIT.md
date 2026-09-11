# 第二轮跨查询槽 coverage bias：本地查重与数学审查

结论：**未发现精确的跨 slot coverage score-bias 已训实现。它可以作为独立结构假设；目前没有证据证明跨 slot 冗余是误差来源，也没有证据支持优先抑制首轮关注位置。** 若继续，均值 coverage 与排除本槽的均值 coverage 比 max coverage 更容易解释；每种必须保留同幅正、负方向与原模型。不要把 mean(log P1) 换一个名称当作新的概率机制，也不建议把 max、多个系数一起扩成宽 sweep。

此次只读 memory、小型结果 JSON、模型及已接受诊断源码；没有读取真实 NPZ/权重，未执行 forward、bootstrap、训练或修改 memory。当前策略账本仍为 117 次已入账训练，187 条 episode 包含新增 6 条 v1 implementation-smoke failure；另有 v2 八次正式训练通过 quality audit，尚处于配对后处理阶段。失败 smoke 没有 AUC。

## 1. 查重：已做与尚未做

| 本地记录 | 实际干预 | 对 coverage 的覆盖程度 |
|---|---|---|
| [原检索替代查重](../optimizer-layout-replay/retrieval-alternatives-audit.md)，以及 [log-prior 代数验证](../optimizer-layout-replay/verify_query_prior_equivalence.py) | 讨论/合成验证 `S2+α log P1` 与 q_norm 后线性 Q 混合的等价 | 不是一次真实训练；未实现跨 slot coverage。 |
| `nf_entry025`、`nf_entry075`、`nf_fixed_memory`，以及 [QueryMixing](../recurrence-query-mixing/QFormerCross8QueryMixing.py:57) | q_norm 前 entry/work 全输入固定或低秩条件混合，连带 interaction/residual | 与直接改第二轮历史 score 不同；没有跨 slot 概率聚合。 |
| `route_q4_retrieval` / `route_q8_retrieval`；[QueryRoutes](../recurrence-query-routes/QFormerCross8QueryRoutes.py) | 第二 Wq 的输入低秩修正 | 不读取首轮 P，不是 coverage。其他分路也没有这个机制。 |
| `role_q{4,8}_query_{norm,projection}`；[RetrievalUntied](../recurrence-readout-retrieval/QFormerCross8RetrievalUntied.py) | 克隆第二轮 q_norm/Wq 参数 | 已有负训练结果，但不能当作 coverage 负结果。 |
| `taobao-second-read-attention-originals-5514`、`taobao-second-query-direction-norm-5514` | 固定 checkpoint 的同槽 P2←P1；或第二 Q 的方向/范数替换 | 已完成的路径依赖诊断。没有跨槽聚合、没有 coverage 训练，也没有跨槽冗余判定。 |
| [Directional v2 模型](../recurrence-directional-retrieval-v2/QFormerCross8DirectionalRetrieval.py:52) 与本轮八组 formal | 每槽 `-Q1`、uniform K、P1K、P1(K−mean K) 来源，统一有界切向旋转并恢复 Q2 范数 | 三个 key 来源仍按各自槽的 P1 读取；uniform K 是对历史 key 的均值，不是对 query slots 的 coverage。没有跨槽概率约束。 |

特别区分：固定 `α log P1` 的**精确未归一化查询混合**没有本地已训记录。当前 v2 extrapolate 带范数保持/source floor/切向投影，旧 learned mixing 又在 q_norm 前；不能把它们写成“精确 log-prior 已训练失败”。只能说这些属于相邻的查询混合/方向假设，代数重命名不构成独立新机制。

本轮 [quality-summary](../recurrence-directional-retrieval-v2/reports/quality-summary.json) 的原模型宏 AUC 点差：

| 来源 | q4 Δmacro AUC | q8 Δmacro AUC |
|---|---:|---:|
| extrapolate | −0.000474740 | −0.004004762 |
| uniform_key | −0.002420325 | −0.000453636 |
| attended_key | −0.002480571 | −0.003827819 |
| centered_key | −0.002274683 | −0.002763634 |

八个点差均负，并低于既有近候选 −0.0001 下限。这里未读取仍在运行的 paired 输出，不编造其区间。这批结果削弱继续放大/旋转第二 Q 的理由；它们没有直接否证保持 Q2 不变的跨槽历史 bias。

## 2. 准确定义 coverage，避免代数重复

逐样本、逐 head 记第一轮 `P1[j,l]`，槽数 `m∈{4,8}`，有效历史集合 `V`、长度 `L`。原生第二分数为 `S2[j,l]=s Q2[j]·K[l]`。这里 Q 是 q_norm 后的 head 向量，两轮共享 K、mask、scale，原 attention dropout=0；[原 CrossValue](../recurrence-directional-retrieval-v2/source-snapshot/QFormerCross8.py:179) 和已接受 probe 的 exact invariants 支持此条件。建议只沿 slot 轴聚合，暂不混 head；slot 也不是任务标签，四个任务头读取完整槽表示。

三种不同含义：

- `Cmean[l] = (1/m) Σj P1[j,l]`：首轮平均概率质量，所有第二轮槽共享它。单个尖峰槽也能抬高它，不是纯“多个槽重复计数”。
- `Cmax[l] = max_j P1[j,l]`：至少有一个槽强烈关注的程度。多个槽同时关注并不会进一步增加 max，因而**尤其不能叫冗余计数**。
- `Cloo[j,l] = (Σi≠j P1[i,l])/(m−1)`：其他首轮槽覆盖当前历史的位置，给各第二轮槽不同的 bias。它排除了本槽，但仍不是“其他槽已把所需信息完整提取”的证明。

考虑 `P2' = softmax_valid(S2 + α G)`：

1. **同槽 `G=log P1[j]` 等价线性 Q 混合。** 因 `log P1[j,l]=s Q1[j]·K[l]−log Zj`，逐槽常数被 softmax 消去，故等价 `Q2[j]+αQ1[j]`。
2. **先取 log 再均值，仍等价。** `G=(1/m)Σi log P1[i]` 等价 `Q2[j]+α mean_i Q1[i]`；LOO 的 log 均值等价 `Q2[j]+α mean_i≠j Q1[i]`。任何对 key 位置不变的 slot 加权 log 概率和都满足这一点。这不是新的非线性 coverage 机制。
3. **先均值概率再取 log，通常不等价上述固定线性混合。** `log Cmean[l] = logsumexp_i(s Q1[i]·K[l]−log Zi)−log m`；LOO 同理。这是不同查询打分的非线性混合，不能把 `log(mean P)` 写成 `mean(log P)`。
4. **`log Cmax` 通常也是非线性的。** 它等于各槽归一化 log 概率在每个 key 上的最大值，获胜槽可随 key 改变；若一个槽在所有有效 key 上都获胜，则退化为该槽的单查询先验。由于每槽概率和都是 1，这种全位置支配实际上要求其他槽分布也与之相同，允许平局。直接使用 `Cmean/Cmax/Cloo` 的有限函数作为 bias，一般同样不是固定线性 Q 混合。

上述“通常不等价”指不存在统一的那种线性 Q1 混合恒等式，不是在任意有限 K 矩阵上声称绝不可能求出某个有效 query 拟合 bias。实数函数等价也不保证 BF16 逐位、不同 kernel 或裁剪后的等价；detach 首轮路径会改变梯度，不能靠前向代数声称训练过程相同。

## 3. 正负方向与有限偏置

对同一预定有限函数 `g(C)`，比较 `S2 + λ g(C)` 与 `S2 − λ g(C)`，两者使用相同 `|λ|`、平滑/截断和计算路径。正向支持再次确认首轮偏好的历史，负向支持转向其他位置；**都只是被测假设**。若使用 log coverage，正向相当于乘一个 coverage 因子，负向相当于除该因子，并不凭数学自动产生更相关的信息。

需在训练前确定是否用概率、log 概率或其他有界变换。直接 mean 概率的典型尺度约 `1/L`，同一 λ 会随历史长度改变实际强度；sum 概率又把 q4/q8 的槽数混进幅度。若用相对均匀分布的平滑 log coverage，例如先把均值型 C 与 `1/L` 按固定比例平滑，再对 `L*C` 取 log，必须明确 floor、有限上限和有效历史内的中心化方式。它不再与无 floor 的 log-prior 精确等价。这里不选新超参数或扩展 sweep。

padding 不参与 coverage 聚合、均值、log 或 bias 中心化；不能用负系数直接乘 `log(0)=-inf`，也不能让无效历史产生 `+inf`。为支持有限 bias，先安全构造有效值，再与原 mask 合成，不能先制造 NaN 后用 where 掩盖。参数布局可保持，但额外首轮梯度、clip/PSGDKron 状态、score materialization 和 additive-mask kernel 开销都会改变；需要如实测量，不能宣称训练轨迹或 FLOPs 不变。未指定 detach 时应把它作为正常可微结构；后续若 detach 则属于另一个训练假设。

## 4. 退化与归因边界

- `L=0`：不形成概率分布；coverage/bias 取安全零，原 attention 的空历史语义照常保留，不能把整个输出、线性 bias 或 residual 一并归零。`L=1`：三种 coverage 都是 1，位置间无差异；加有限常数不改变 attention。
- 所有首轮槽均匀时，coverage 在有效位置上恒定，softmax 结果不变；相同 key 导致的均匀打分也如此。这不是数据应被删除的理由。
- 所有槽分布相同时，mean、LOO mean、max 完全相同。无平滑/非线性截断的 log 版本会退化为同槽 log-prior；**此时所谓跨槽机制没有新的分布信息**。
- q4/q8 均有 `m>1`，LOO 定义有效。代数上 `Cloo_j=Cmean+(Cmean−P1_j)/(m−1)`，所以 LOO 与 mean 的差异在 q4 带 `1/3`、q8 带 `1/7` 系数；同 λ 不等于等幅干预。`m=1` 若将来出现必须显式回退；`m=2` 的 LOO 只剩另一槽，log 版本又退化为另一 Q1 的线性混合。
- `sum_l Cmax` 随槽数及专长分布变化，不是始终为 1。log 后按历史归一化只增加常数，softmax 会消去它；直接 Cmax bias 的归一化则会改变有效强度。max 还让梯度集中到获胜槽，平局处理可能取一个索引或分摊；不应与平滑 mean 的差别全部归因“覆盖类型”。
- 即使所有第二轮槽共享 mean/max bias，它们仍保留各自的 S2；这不是硬禁止重复，也不是给槽分配不同历史的约束。LOO 能提供不同槽的参考分布，但正负偏置都可能抹掉互补或有用的共同信号。
- 关注强度不等于语义有用性。多个槽关注同一物品，仍可能通过不同 head、query interaction、value 变换服务多个任务；反之分布不重叠也不能证明信息互补。不要把“被关注”命名为“已无用/已解释完”。

## 5. 现有诊断能回答到哪里

已接受 [attention probe](../memory/tool-evolution/candidates/taobao-second-read-attention-v1/src/attention_probe.py:79) 计算的是**相同槽的跨轮** entropy、`sum_l min(P1[j,l],P2[j,l])` 和 JS，再由 [描述摘要](../memory/tool-evolution/candidates/taobao-second-read-attention-v1/artifacts/descriptive-summary.md) 对 head/slot 汇总。约 0.844/0.849 的 overlap 不是首轮不同槽之间的 overlap。保存的逐槽熵、同槽跨轮 JS/overlap 不足以重建 `P1[i]` 与 `P1[j]` 的位置对齐关系；该 probe 又明确不保存完整 P，因此不能用现有摘要补造跨槽冗余诊断。

[方向/范数结果](../recurrence-query-direction-norm/RESULTS.md) 仅支持原 q4.buy 对指定第二方向替换的条件依赖。它不证明 coverage 会改善，也不支持按 buy 单独选结构或把其他任务当作不需要循环。多个相同边际熵的槽既可能关注同一位置，也可能关注不同位置，现存摘要不能区分。

## 6. 有用与不值得试的判断

**有用：** 把“第二槽能否利用其他第一槽的历史覆盖情况”作为独立、可证伪的结构问题，保持 Q2、第二 interaction/residual 与两轮 no-feedback 路径不变。LOO mean 最直接对应这个问题；mean-including-own 是解释排除本槽是否有用的近邻控制。正负 bias 各自对原模型的比较决定是否有改进，正−负比较决定这两种方向在本设置谁更好；若都低于原模型，内部胜负不能挽救候选。若要宣称跨槽相对同槽的独立收益，还需要相同 bias 变换的 own-slot 控制；mean/LOO 内部获胜本身不完成该归因。

**暂缓：** max-slot。它更像“任何一个槽强关注”的并集信号而非重复覆盖，还引入硬 winner 梯度和槽数相关幅度。mean/LOO 尚未证明有用时，不值得把三种聚合和多个 λ 全铺开。可以先固定一种聚合的正负两个方向；若要同时解释自身槽是否应计入，再加入 mean/LOO 成对控制，不能只挑负 bias 预设 novelty 正确。

**不作为独立新机制：** 同槽 `α log P1`、mean/LOO `mean(log P1)`、改写为同一个 Q 线性混合后重复记名；继续密扫旋转系数；或仅凭旧 overlap、q4.buy 依赖及某个描述统计为新结构预报 AUC 收益。当前结论是本地未试且有明确边界的假设，不是已有效的训练计划。
