# 首轮跨查询槽 coverage 能否帮助第二轮历史检索

研究笔记，2026-09-09。仅实际浏览下列三篇原始论文/作者报告并核对已有小型结果 JSON；本次无模型修改、训练、新诊断、数据数组读取或 global memory 写入。所有新假设及其效果均未验证。

当前事实：本轮八个方向修正候选（六个 K 反馈、两个 query 外推）的 macro AUC 点估计全部低于各自原循环，不能据此预设换成 coverage 就会有效。原始指标见 [V2 quality-summary](../recurrence-directional-retrieval-v2/reports/quality-summary.json)。这里不提前声称配对区间结论，也不将这八个结果泛化为所有第二轮检索修改都无效。

要回答的具体问题是：**保留原第二轮 Q2、K/V、交互分支和残差，只让第二轮历史 score 感知首轮其他 query slots 的读取分布，是否具有独立预测价值？** 这与本轮把 K 聚合反馈到 Q2 方向的改造不同，但“机制不同”本身不构成收益证据。

1. **Tu et al., 2016, Modeling Coverage for Neural Machine Translation，ACL。**

   原文位置：§3.1–3.2、式(7)–(8)，§4 的辅助目标讨论，§5.2–5.3。每个源位置维护累积 attention；语言学版本为 `C(i,j)=sum_{k<=i} alpha(k,j)/Phi(j)`，其中 fertility `Phi` 表示预计生成量。此前 coverage 进入下一次 attention 的可学习打分器；设计意图是降低充分读取位置的权重，并非硬性“每处只能读一次”。最有信息量的负结果：额外惩罚总 attention 与 fertility 的差异，虽略改善 alignment，却持续降低翻译质量。作者指出上下文化 hidden state 不等于原词本身，attention 质量不应替代预测目标。[ACL 原文 PDF，pp.79–82](https://aclanthology.org/P16-1008.pdf#page=4)

   对本任务的推论：可以借用“显式保留读取记录”的位置选择，不能搬用“每条历史应消费完一次”的语义。推荐历史允许反复提供有效证据；降低 overlap 或提高覆盖率只能证明分布发生变化。

2. **Hao et al., 2021, Diversity Regularized Interests Modeling for Recommender Systems（DRIM），作者预印本。**

   原文位置：§3.3–3.5、式(2)、(6)–(13)，§4.5 表3。其 capsule routing 对每条历史在兴趣槽之间归一化，且对兴趣向量施加 entropy、离均值距离、角度等约束；没有跨读取轮次的 coverage。方法效果依赖数据：Amazon Books 的 HR@50，mean/div 为 0.6463/0.7073，均低于 ComiRec 的 0.7892；工业集则 div 最好。不能从“向量更分散”推出各数据集预测更好。另式(8)的 `1−cos` 以 `+lambda` 加入最小化目标；若 lambda 为正，其字面方向与鼓励分离不符，照搬前须核清符号/实现，本次未下载代码。[作者原文 PDF，pp.4–6](https://arxiv.org/pdf/2103.12404#page=4)

   对本任务的推论：跨槽协调是合理的独立问题，但兴趣向量距离、历史分配竞争、历史 attention coverage 是三个不同对象。本仓库的 attention 是每个槽在历史位置上归一化；不能把 DRIM 的归一化轴直接认作已有实现，更不能将四/八个槽当成四个预测任务。

3. **Hu et al., 2018, Reinforced Mnemonic Reader for Machine Reading Comprehension，IJCAI。**

   原文位置：§2.3 式(4)–(5)，§5.3 表4，§5.4 表5。reattention 将先前两种 attention 分布的内积作为关联项，加到当前 similarity：`E_t=f(current representations)+gamma*overlap(previous distributions)`；gamma 可学习、文中初值为3。因此它可强化已有关联，并非直接减去累计覆盖。去掉 reattention，SQuAD dev EM/F1 从 78.9/86.3 降到 78.1/85.8；五个 block 又低于原三个。所谓 attention deficiency 以 ensemble 平均 attention 为参照，不是独立真实 attention 标注。[IJCAI 原文 PDF，§2.3](https://www.ijcai.org/proceedings/2018/0570.pdf#page=3)、[消融与描述量，pp.4104–4105](https://www.ijcai.org/proceedings/2018/0570.pdf#page=6)

   对本任务的推论：首轮分布可以是下一轮打分信息，但应允许“强化重复证据”这一相反假设。该论文的 question-context/self-attention 矩阵及 NLP 结果不等同于推荐 query slots；不应为移植它额外引入整套历史 self-attention。

以下是结合本仓库结构的待证伪设计推导，不是上述论文已验证的推荐方法。

令 `P1[b,h,s,j]` 为原首轮在有效历史 mask 下的概率，`S` 是查询槽数，`L_b` 是有效历史长度。保留 head 轴，可以定义：

```text
C_all[b,h,j]      = mean_s P1[b,h,s,j]
C_other[b,h,s,j]  = sum_{r != s} P1[b,h,r,j] / (S - 1)
score2_native    = original_scale * Q2 @ K.transpose(-1, -2)
score2_modified  = score2_native + sign * lambda * g(C_other, L_b)
```

`C_all` 只描述整体首轮分配；`C_other` 排除当前槽自己的首轮读取，才能更直接提出“其他槽已读过什么”的协调假设。二者都不是已消费信息量。所有槽相同时，两者相同；仅有单槽时 `C_other` 不定义。这里的 Q2 是原 `w_q` 与 `q_norm` 后的检索张量；原 `w_qi(queries)`、V、`w_v_pair/w_o` 和第二轮残差均保留。实际路径见 [冻结原 attention 源码](../recurrence-directional-retrieval-v2/source-snapshot/QFormerCross8.py)。

有界的 `g` 可避免少数峰值任意放大 logits；例如 `tanh(L_b*C−1)` 是一个**尚未定案**的长度缩放形式。仅减去 `1/L_b` 属于每行常数，对 softmax 不产生影响；长度乘数和非线性才会改变干预幅度。跨槽取均值避免 q4/q8 因槽数翻倍就直接翻倍，但不保证两家具有相同 score 敏感度。

| 可检验问题 | 什么结果才有信息 | 不能据此声称什么 |
|---|---|---|
| 抑制还是强化首轮覆盖？ | 在预先固定幅度与预算下，分别考察负/正 score bias；保留原模型作为对照 | overlap 下降本身不证明去掉了无用信息 |
| 跨槽信息还是单槽再次读取？ | `C_other` 与本槽 `P1`、或全槽 `C_all` 的预定对照，区分读取来源 | 多槽协调不能自动解释为任务专属分工 |
| 分布变化是否影响有效 value？ | 相同有效 V 的合成例应允许 P 改变但 P@V 不变；真实结论仍需预测指标 | entropy/JS/coverage 是质量替代指标 |
| 反馈内容还是训练梯度路径？ | 如果进入训练，先固定是否 stop-gradient；可微 P1 会新增跨轮梯度边，虽然没有新增参数 | 固定 checkpoint 干预的好坏等同于重训收益 |

实施前仍缺三类证据：本构造的隔离和数值正确性、任务相关的实际证据、固定训练预算下的效果。成本也不是零：首轮概率的获取/保存、跨槽归约和第二轮加性 bias 都有开销。原代码使用原生 SDPA，新增浮点 bias 可能改变实际 kernel；零幅度控制应走原路径，不能只凭数学等式宣称 BF16 bitwise identity。必须保持原 scale、有效 mask 和 dropout=0；全 padding 应延续原行为，不能对全 `-inf` 手写 softmax 后制造 NaN，后续投影的 affine bias 也不能抹掉。

现有八组负点估计限制的是本轮方向修正方案。三篇论文只支持上述问题值得被明确区分，不支持直接把 coverage 列为优化成功候选；本笔记不启动下一轮实验。
