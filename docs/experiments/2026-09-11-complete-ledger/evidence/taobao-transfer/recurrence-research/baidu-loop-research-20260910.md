# ITT：信息流与复现边界（2026-09-10 调研）

用户已确认目标为 **Inner Thinking Transformer: Leveraging Dynamic Depth Scaling to Foster Adaptive Internal Thinking**。此前 LASAR、Sparse Growing Transformer 只是题名未确定时的百度候选；现在排除，不据此扩展试验。B 的 paired source 工作已暂停，`paired_analysis.py`、`test_paired_analysis.py` 均尚未创建。本次只浏览原论文和查找作者代码，未执行模型、数据、训练、诊断或测试。

**身份与来源。** 作者为 Yilong Chen、Junyuan Shang、Zhenyu Zhang、Yanxi Xie、Jiawei Sheng、Tingwen Liu、Shuohuan Wang、Yu Sun、Hua Wu、Haifeng Wang。正式 PDF 首页将 Shang、Zhang、Wang、Sun、Wu、Wang 六人标为 Baidu；另有中科院信工所/国科大及北师大。arXiv v1 为 2025-02-19，v2 为 2025-02-23；正式发表于 ACL 2025，28241–28259 页。它研究语言模型，不是 CTR/CVR。[版本记录](https://arxiv.org/abs/2502.13842)、[ACL 原文](https://aclanthology.org/2025.acl-long.1369.pdf)。

**官方代码未确认。** 已查论文/ACL 页面及题名、arXiv ID、作者、百度和 GitHub 的组合检索，未找到可由作者或论文链接确认的 ITT 仓库。论文 §4.1 提及 Sheared-LLaMA、Composer 作为基础设施，不能将这些上游仓库当成 ITT 实现。以下是数学和伪代码审查，不是官方运行代码审查。

**文中定义的信息流。** §3.2–3.3、式 (4)–(7) 将同一个变换 `f` 反复用于当前隐藏状态；步编码 `φ` 逐通道加权各步结果，router 按当前状态选择 token。用统一记号概括正文：

\[
U_0=f(X),\qquad
V_{t,i}=\begin{cases}
\alpha_t w_{t,i}f(U_{t-1})_i&i\in M_t\\
U_{t-1,i}&i\notin M_t,
\end{cases}\qquad
U_t=U_0\odot\phi_0+\sum_{j=1}^{t}V_j\odot\phi_j.
\]

这里将正文混用的 token/step 下标重新命名；`f(U)_i` 仍是带上下文的 Transformer 运算，不能解释成独立 token MLP。首次全 token 前向，之后只让选中 token 执行额外 `f`。主变换共享参数，router 和步编码随步区分；不是严格零新增参数。[arXiv v2 §3](https://arxiv.org/html/2502.13842v2#S3)。

**最有用的实现信息在 Appendix A.1 Algorithm 1（正式 PDF 第 14 页）。** 首次 `f(X, KV_past, mask, Θ)` 返回 `U0, KV_new`；随后调用 `f(U_selected, KV_new, mask, Θ)` 并再次返回更新后的 `KV_new`。据此可确认循环接口携带当前状态和 cache，不能把它描述为“只换第二轮 Q，始终固定外部历史”。由标准 Transformer 推断，活动输入参与 Q/K/V 投影；但伪代码没有规定 cache 是原位覆盖还是追加、未活动位置保留何版 K/V、位置编码及因果 mask 怎样对齐。**这些细节未达到直接复现程度。** 原始 `X` 明确进入首次调用；之后保留首轮输出的累积路径，没有明写每轮再次注入原始 `X`。这也不是将 LM 输出概率反馈进工作状态。[Algorithm 1](https://aclanthology.org/2025.acl-long.1369.pdf#page=14)。

**梯度和监督。** §3.3–3.4 指定最终语言模型交叉熵，router 的连续权重参与反传；没有逐步答案监督/教师 CoT 损失。硬 Top-K 的索引选择本身不因此变成可微。正文及附录采用跨步链式梯度，没有写 detach；未有官方代码，不能进一步证明每个 cache 张量的实际 autograd 保留方式。§3.1 用真实目标损失描述 early exit 是概念说明，不能直接作为线上停机规则。[§3.1–3.4](https://arxiv.org/html/2502.13842v2#S3)。

**消融证据。** 正式原文 Table 3（PDF 第 7 页，ITT ×4，162M）：完整 PPL 10.25；去残差累积 11.02；去 router 10.44；去步编码 10.56。前者损害最大，因此不能把结果归结为只靠 router。Only Select 重加权 10.25，Symmetric 10.41。Table 2 同一训练配置在推理修改各步参与比例，70/70/70 为 10.52，70/70/50 为 10.26；它提示更多 token 更新并不保证更好，也不是淘宝上稀疏更新收益的证据。[§4.3、Tables 2–3](https://aclanthology.org/2025.acl-long.1369.pdf#page=7)。

**复现前必须保留的歧义。** Eq (6) 含 `w`，Algorithm 1 融合行未写该乘子；未选 token 的 bypass 分支与随后累计融合不能直接推出最终表示严格不变。正文的 percentile 记法与附录 `k=floor(C*S)` 的选择比例语义也需要落实。附录只提 `getCapacity(P,ρ,τ)`，没有其可执行公式。论文未提供独立的“cache 更新/跨层传递/原输入注入”消融。Table 3 的 10.56−10.25 实为 +0.31，括号印为 +0.22；FLOPs 4.70→3.29 实为约 30% 降低，正文“超过 50%”不能直接据表重现。正式 PDF 摘要的模型规模也与其正文表格不一致；这里避免引用那条跨规模宣传数字。这些问题不否定消融方向，但限制了精确移植和机制归因。

与本地结构的关系属于我们的分析。当前 [NoFeedbackStage](../recurrence-maturity-depth/source-snapshot/QFormerCross8NoFeedback.py) 的两轮为：

\[
e=L_0(q,H),\ z_1=L_1(e,H),\ y_1=L_2(e,z_1),\
x=(e+z_1)/2,\ z_2=L_1(x,H),\ y_2=L_2(y_1,z_2).
\]

`z1` 已进入下一工作层，信息流没有“循环完全看不到前次结果”的断路。区别是工作/预测双状态、固定历史以及固定入口混合；ITT 使用累计 token 状态和选择性同层重算。ITT 不能证明取消 `y→z` 隔离一定更好，也不能把文本 token 与四个任务或 4/8 个查询槽等同。

可检验的两个借鉴点（研究解释，不是新训练指令）：

1. **先检验状态累积的语义，再增加循环次数或 router。** 论文最强消融支持的是 RTC。若未来转成推荐结构，应把“原状态”和“本轮增量”明确定义，并在循环内部比较累积与替换，固定其他读入和预测路径。原层本身已有残差，不能把其完整输出再次相加后称为同一个增量更新。旧 post-readout 混合、只改检索投影并不自动等同这项机制；但须查重既有 core/state-routes，不能据论文假定未做过。若同预算的累积对照仍不改善最终四任务结果，这条迁移假设即不获支持；论文的条件性收敛推导不提供保证。
2. **“保留首轮表示、只加强部分状态”应与“所有状态同步变换”分开比较。** Only Select 对照给了这个问题直接但有限的依据。若移植，需比较相同参数/计算下的选择策略，并明确未选状态是否真的保持、是否仍可被当作 K/V 读取；不能只用 dense mask 宣称省算。4/8 槽很短，硬路由和 gather 的成本可能超过节省，而且它改变训练路径与优化器行为。若收益仅来自加权尺度或额外参数，或完整四任务指标无改善，就不能解释为识别了“更难的 token”。优先复用已有证据，不自动开启新的 router/比例扫描。

以上信息流问题与历史行为在目标时刻是否成熟是两条独立问题；架构改变不能补救输入可用性未定义，也不能用共同输入指纹证明线上可用。
