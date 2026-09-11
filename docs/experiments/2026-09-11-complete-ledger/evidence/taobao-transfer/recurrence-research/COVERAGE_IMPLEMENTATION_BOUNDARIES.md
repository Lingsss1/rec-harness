# 跨槽 coverage 的实现边界

2026-09-09，仅源码、小型配置与已有审计收据的只读分析。本次只新增此研究笔记：没有实现模型、运行 forward/训练/GPU、读取权重或预测数组，也没有修改冻结源与 memory。下一队列尚未冻结。

当前八个方向修正候选的 macro AUC 点差均负。这里只说明保持原 Q2 的跨槽 score bias 如何成为可检验的不同假设，不据此预测收益；查重、代数边界与已有结果以 [coverage audit](COVERAGE_RETRIEVAL_AUDIT.md) 为准。概率均值与 `mean(log P1)` 不可互换，后者仍能退化为查询线性混合。

## 1. 唯一目标调用与最小改动

两家原生调度相同，按原代码的零起始 layer index 表示：

```text
e  = L0(ns_queries, H, mask)
z1 = L1(e, H, mask)
y1 = L2(e, z1, None)
z2 = L1((e + z1) / 2, H, mask)
y2 = L2(y1, z2, None)
h2 = original_post_self_attention_and_FFN(y2)
```

q4 为 `ContextReadStage(context_reread=False, attention_position='end')`，q8 为 `ExploreStage(recurrence_mode='dual_no_feedback')`。原两轮中，预测状态 y 不反馈到工作状态 z；H/mask 不变，L1 的权重共享，post self-attention 只执行一次。[q4 调度](../recurrence-directional-retrieval-v2/source-snapshot/QFormerCross8ContextRead.py)、[q8 调度](../recurrence-directional-retrieval-v2/source-snapshot/QFormerCross8RecurrenceExplore.py)

目标精确到 **`unified_qformer.layers[1].cross_attn` 第二次调用内部的 `F.scaled_dot_product_attention`**，不是入口 L0，也不是两次预测层 L2。注意旧讨论中口语“第二层/L2”容易与源码 index=2 混淆。

最小生产结构可以是保持原成员名称/参数结构的 attention 薄子类：首次工作读取照常调用原生 SDPA，同时从原归一化 Q1/K1 计算临时 coverage；第二次只把原 boolean `attn_mask` 合成一个 additive mask。后续 `w_qi(queries) * attended_features → w_v_pair → w_o` 以及外层残差完全保持。任意历史位置 bias 一般不能通过仅替换 `w_q`/`q_norm` 输入或输出来精确实现；不应为保留旧 hook 形式而再次改变 Q2。

如果需要替换 attention 实例，应在原模型 `compile()` 构造优化器**之前**、原 stage 构造期间完成，并保留原模块路径、参数注册顺序及初始化 RNG。额外实例构造须隔离 RNG，随后核对原底座参数值，而不是在 optimizer 已绑定旧 Parameter 后换模块。固定强度与确定性聚合可做到新增 Parameter/buffer 为零；可学习强度则是另一个参数布局假设。临时状态只属于本次 forward，正常/异常均在 `finally` 清除；不改全局 SDPA 函数，不引入持久 Tensor cache。[原 attention 与层定义](../recurrence-directional-retrieval-v2/source-snapshot/QFormerCross8.py)

## 2. 实际形状、精度与 P1 的定义

配置为 B=8192、head 数 Hn=4、总维 D=384、每 head 维 d=96、查询槽 S=4 或8、历史物理长度 L=100；最后不足批次的 B 可小于8192。有效历史数 `L_b` 可从0到100，与物理 L 不同。两家共享 temperature=1，原 `scale=1/sqrt(96)≈0.1020620726`，`dropout_p=0`、`is_causal=False`。[固定参考配置](../recurrence-directional-retrieval-v2/references.json)

| 张量/位置 | 形状 | 当前精度边界 |
|---|---|---|
| 原外层 queries，H | `[B,S,384]`、`[B,100,384]` | 参数为FP32；位置表/LayerNorm/残差路径产生FP32状态 |
| `w_q`、`w_k` 投影后 | `[B,4,S,96]`、`[B,4,100,96]` | BF16 autocast 下的 Linear 输出为BF16 |
| `q_norm(Q)`、`k_norm(K)` 后 | 同上 | 原 LayerNorm 路径在此混合精度环境使用FP32；不要假定其dtype等于后续SDPA输出 |
| `w_fi(H)` 即 V | `[B,4,100,96]` | BF16 autocast 下为BF16，没有额外 value LayerNorm |
| 原 `mask` / `key_mask` | `[B,100]` / `[B,1,1,100]` | loader mask 为FP32；attention转boolean，True表示有效 |
| `attended_features` | `[B,4,S,96]` | 原SDPA位于BF16 autocast上下文中；计算类型受自动转换和backend决定 |
| 显式参考 scores/P1 | `[B,4,S,100]` | 必须冻结定义，不能靠外层autocast意外选择精度 |

形状来自原源代码；FP32 norm Q 与BF16 SDPA source 的实际混合已由上一轮 [tiny CUDA 收据](../recurrence-directional-retrieval-v2/reports/tiny-cuda-autocast/result.json) 验证，曾导致 v1 dtype guard 失败。这不是本次对完整两模型每个张量的新测量。非连续 view/stride 也应保留，不能无条件把所有投影 `.contiguous()` 或 cast 到同一dtype。

**原生 SDPA 只返回 P@V，不返回 P。** 首轮 post-norm Q/K 在禁用 autocast 的 FP32 乘法和安全 softmax 下重算出的概率，应称 **reference coverage P1**；CUDA上还需固定/记录TF32与float32 matmul precision，FP32 tensor标签不等于保证全精度乘法。它不保证逐位等于原生 fused SDPA 在BF16转换/融合运算中的隐含概率。若选择先模拟低精度 Q/K 舍入再重算，那是另一明确数值定义，也不能自动取得 fused P 的逐位身份。需要独立 P@V 校准，而不是把参考分布叫“捕获到的原概率”。第一轮预测计算仍应走原 SDPA；不为了导出 P 改写第一轮输出。

## 3. 聚合、符号及 padding

只沿 slot 轴聚合，保留 batch/head/历史位置对应关系：

```text
Cmean[b,h,1,j]  = mean_s P1[b,h,s,j]
Cloo[b,h,s,j]   = (sum_r P1[b,h,r,j] - P1[b,h,s,j]) / (S - 1)
valid_bias     = sign * lambda * g(C, L_b)
SDPA_score     = original_scale * Q2 @ K2.T + additive_mask
additive_mask  = valid_bias at valid positions, -inf at padding positions
```

mean 的 mask 可保留 `[B,4,1,100]` 广播；LOO 为 `[B,4,S,100]`。不要无谓复制 mean 到所有槽。`Cloo` 的 sum-minus-self 实现与直接排除自身求和在浮点尾部可能有差异，fixture 要包含极偏分布；若后续取 log，平滑/floor须提前固定。

正、负方向使用同一函数、相同绝对强度、相同精度与 mask 路径。g 若对 C 单调递增，正向增加首轮高覆盖位置的相对打分，负向减少；若 g 未规定单调性，就不能用这两个行为给它命名。不得将负方向称为“剔除无用历史”。`mean P1` 的典型尺度随 `1/L_b` 变化，sum 还引入槽数缩放；具体变换/强度/平滑均须在训练前固定。mean/LOO 虽均值归一化，LOO与mean的差值仍含 `1/(S−1)`，q4/q8不是天然等幅。

- 首轮有效行在padding logits置 `-inf` 后归一化；**全padding行必须先使用安全有限占位 logits，softmax后显式得到全零P1**，避免先构造NaN再用where掩盖。全padding没有概率分布，不计算 `1/0` 或 `log(0)`。
- g只处理安全有效输入，全padding coverage/bias为零。先形成有限 signed bias，再与原有效mask合成；不能把负号乘到padding的 `-inf` 上变成 `+inf`。
- 第二次SDPA在全padding仍需保持全mask语义并验证其原生安全行为；不能因为coverage为零就把整行mask放开。空历史在本数据里是100个padding，非必须构造长度为0的物理axis。
- all-padding attention结果虽可为零，原 `w_v_pair/w_o` 的bias、后续残差仍存在，不能把整个CrossValue/层输出归零。history token已掩零也不等于投影后的padding K/V为零，Linear有bias。
- 有效长度1或coverage在有效位置上恒定时，有限bias仅加行常数，实数softmax不变；浮点/backend逐位不变仍需另验。

## 4. 两种 identity 回答不同问题

| 控制 | 实现边界 | 可以证明 |
|---|---|---|
| exact identity | 零强度/identity分支在coverage计算前直接走原forward，仍传原boolean mask | 新类装配、参数/RNG/调度未改变底座；输出和未裁剪梯度应逐位相同 |
| same-format 零bias | 保留候选的coverage流程及相同shape/dtype的浮点additive mask，有效位置bias置零、padding仍为 `-inf` | 量化mask格式、额外计算图、backend选择造成的数值和成本差异；不预设逐位相同 |

保留 `F.scaled_dot_product_attention` API **不保证保留原 attention fused kernel**。浮点bias、mask的broadcast形状/stride、dtype以及是否要求mask梯度，都可能影响backend可用性。此环境安装的 [PyTorch SDPA文档源码](../../../ai4ai-runtime/opd-native-env/lib/python3.12/site-packages/torch/nn/functional.py) 明确描述了自动选择、输入限制及跨backend数值差异。不要因为方法名仍为SDPA就声称kernel identity；也不要强制手写全FP32 attention作为候选、却把其全部影响归因于coverage。

候选bias可先用FP32构造，再按冻结的兼容dtype边界送入SDPA；必须验证实际autocast和mask-gradient支持，不能通过擅自将原Q/K/V全转FP32来绕过问题。mean与LOO各自需要对应shape的same-format控制。若该控制与原路明显不一致，应停止将效果单独解释为coverage，先给出实质对照或修改协议；数学零偏置等式不能替代这一证据。CPU校准也不能证明CUDA fused backend身份。本次不新增GPU测试。

## 5. 额外计算与显存数量级

下表是B8192/Hn4/L100/d96下的解析元素数乘dtype大小，MiB=2^20 bytes；不是运行时测量，也不是整模型峰值。只计算额外重算一次 `Q1K1^T`，一个乘加按2 FLOPs。

| 项目 | q4 | q8 |
|---|---:|---:|
| 显式scores或P1的元素数 | 13,107,200 | 26,214,400 |
| 每一份FP32 scores/P1/LOO bias | 50 MiB | 100 MiB |
| 广播mean coverage或bias，一份FP32 | 12.5 MiB | 12.5 MiB |
| 额外QK乘法 | 2.5166 GFLOPs | 5.0332 GFLOPs |
| 首轮Q，若整份FP32复制 | 48 MiB | 96 MiB |
| 首轮K，一份FP32 | 1200 MiB | 1200 MiB |
| 一个完整K或V的BF16表示 | 600 MiB | 600 MiB |
| 原boolean key mask `[B,1,1,100]` | 0.78125 MiB | 0.78125 MiB |

此外有 `O(B*Hn*S*L)` 的softmax、归约、bias变换和第二轮加法；训练还增加这些算子的backward，不能把上表的forward QK数称为总训练FLOPs。mean与LOO共用同一次首轮QK/softmax，不需要额外首轮value聚合，也不需要多一次模型forward。

仅“scores + P1 + 最终bias”若同时存活，mean约112.5/212.5 MiB，LOO约150/300 MiB。它们只是列出的三种Tensor合计：没有包括平滑中间量、mask格式转换、autograd保存和backend workspace。原SDPA若因mask变化fallback到materialize attention，还可能增加第二轮scores/P及其他workspace。

**保留首轮原K引用，不复制K。** coverage形成后尽早释放Python临时引用，`finally`兜底清理。即使没有clone，可微FP32 QK的backward也可能延长这份1200 MiB norm K的存活时间，相对原生autocast路径增加峰值；因此不能把“无复制”当作“无额外大Tensor成本”。detach可改变保存需求，但也会切断首轮参数经coverage到第二轮loss的梯度，属于另一个训练假设。CPU源码分析不能保证既有GPU显存一定够用。

## 6. 冻结前最小 fixture 与身份要求

以下均为待实现要求，不是本次已通过的测试。

1. **原两家tiny stage装配：** S=4/8，保留真实两轮调度和H不变；CPU FP64/FP32使用小D和短历史。exact identity核同seed初始化参数名称/shape/值/注册顺序、CPU RNG、底座与四tower输出、未裁剪梯度；固定强度版本额外参数/buffer为零。若换模块，核dense/sparse optimizer成员顺序及实际Parameter引用没有失效，不需要先做优化器重放。
2. **独立公式：** NumPy/解析非均匀P1验证mean/LOO归约及正负bias的第二次输出与梯度；在原S2平局且两位置coverage不同的例子，正负方向应相反。验证同步slot置换对coverage算子等变，不能把整个带有位置化tower的模型称为槽置换不变。
3. **路径隔离：** first work Q/K/V/mask/输出、原第二外层输入 `(e+z1)/2`、第二Q2与interaction输入均保持；只有第二work的mask含bias。路由仍为 `L0(H), L1(H), L2(z1), L1(H), L2(z2)`，post1次；第二K/V/mask的原内容与首轮一致。prediction变化不能反向进入工作状态。
4. **可微与恢复：** 不detach时，第一Q/K须经coverage获得有限梯度；用其他槽的非退化例子验证跨槽边确实存在，独立公式区分mean/LOO。正/负、异常、重复forward后cache/context均清空，模型state、输入、RNG不被诊断记录修改；临时引用清理不能意外detach尚需backward的图。
5. **有效mask与退化：** 正常、左padding、内部mask孔洞、全padding、单有效历史、相同槽/均匀槽、极偏概率与large finite scores；有效位置集合不变，padding K/V即使因Linear bias非零也不可泄漏。相同有效V的合成例允许attention变化而P@V不变，防止将分布变化误作信息收益。
6. **原生精度与格式：** CPU实际BF16 autocast下核FP32 norm Q/K与BF16 V/输出边界、bias dtype及backward；exact identity和same-format零bias分开报告。reference P1的P@V与原SDPA以预定数值容差校准，FP64公式检验与BF16误差不可混用。真实 d96、CUDA原生backend及bias梯度支持仍需后续授权的tiny fixture；本次不执行，也不把CPU通过写成CUDA通过。

只有这些实现边界明确，才适合决定是否冻结少量mean/LOO正负对照。zero-format显著偏差、mask梯度不支持、首轮校准失败或实际资源不合预算，都应先解决；本轮八组负结果仍保持原判定。
