> Historical design evidence from the completed coverage round; this is not a field-role training result. Original: [NEXT_STRUCTURE_AUDIT.md](workspace/qformer-loop/taobao-transfer/recurrence-coverage-retrieval/NEXT_STRUCTURE_AUDIT.md).

本轮已进入“全部负且无接近候选”分支：八组正式训练成功、质量审核通过；相对同 q 原循环的 macro AUC 点差均低于 −0.0001，最好 `cov_q8_loo_positive` 仍为 −0.001069121358。来源为 [quality-summary.json](workspace/qformer-loop/taobao-transfer/recurrence-coverage-retrieval/reports/quality-summary.json)，SHA `ab3e9cad65194b7d34656032a6f18736c72826ecb77cb106797a588eb558bb05`。本说明未独立复核本轮配对区间或 latency；根代理已另报告 quality/latency 均不支持候选，16 项配对仍在闭环。不把点估计全负写成统计上已证明全面退步。原最佳单轮/原循环继续作为研究参考。

只提出以下两个未发现严格同等既往实现的结构假设；均未实现、未测量、未确认收益。它们来自本地源码和已完成记录的缺口，不借用“论文已经证明会有效”的说法。当前不启动任何训练或 probe，不创建 handoff，不扩展 strength/rank 标量扫描。

| 已做过的近邻 | 源码与结果 | 对新假设的限制 |
| --- | --- | --- |
| 淘宝改变第二轮 memory 来源 | [MemorySourcesStage](workspace/qformer-loop/taobao-transfer/recurrence-memory-sources/QFormerCross8MemorySources.py:16) 的 `work_entry` 拼接 `[H,e]`；另三种只在 prediction 读取处拼接 `[z2,H]`、`[z2,z1]`、`[z2,e]`。[八组结果](workspace/qformer-loop/taobao-transfer/recurrence-memory-sources/RESULTS.md) 全低同 q 原循环。 | 不能重提“加回历史/入口 token”作为新方向；该轮没有逐历史事件改写第二轮 K 或 V。 |
| 淘宝工作/预测状态、query 修正与路由 | [StateRoutes](workspace/qformer-loop/taobao-transfer/recurrence-state-routes/QFormerCross8StateRoutes.py:15)、[NoFeedback](workspace/qformer-loop/taobao-transfer/recurrence-q4-capacity/QFormerCross8NoFeedback.py:17) 仍反复读取相同 H；[QueryRoutes](workspace/qformer-loop/taobao-transfer/recurrence-query-routes/QFormerCross8QueryRoutes.py:46) 的 K/V 均从原 features 投影，只分流 query correction。directional 一轮改第二 Q，coverage 本轮改 score mask。 | `work state` 是少量 query slots，不是 L 个历史事件；变更 z 不等于写回每个历史 token。不得把旧 query gate 改名为 memory conditioning。 |
| KuaiRec 已实际写回历史 | [HistoryWriter](workspace/qformer-loop/model_zoo/QFormerCross11SourceUpdate.py:9) 做 `H←H+a·Attn(LN(H),LN(q或H),LN(q或H))`，在 query-between/query-after-field/self-between/self-before 四位置、两初始化跑过八组。 [round13解释](workspace/qformer-loop/reports/round13/interpretation.md) 全低 r7，最好 query-between-zero AUC 0.870694573、Δr7 −0.000539486；p50 15.16ms 对 r7 12.59ms。 | 一般的“两轮间用 query 反读/写 H”并非新探索；旧写回会同时改变后续 K/V，也有新 writer 参数。它是不同数据/父结构的反证，不能宣称已严格否定淘宝 V-only 条件化。 |
| 重复轮的 embedding/gate 与字段读取 | [QFormerCross11Conditioned](workspace/qformer-loop/model_zoo/QFormerCross11Conditioned.py:10) 的 `condition_scope=history` 实际变 query 输入或 query 更新幅度，不改历史 K/V。 | 名称含 history/field 不能当作逐事件 K/V 条件化或 item/action 分解已经做过的证据。 |

候选一：**只给第二轮逐事件 V 加入工作状态条件化的非线性残差，保留原 Q2、K2 与概率路径。**

原历史 token 为 H_l，原第二轮值为 V_l=W_fi(H_l)，`s1=mean_slots(z1−e)`，仅使用第一轮工作状态，完全不使用 y1 或标签。一个可具体审查的小结构是：

`u_l = GELU(A_H·functionalLN(H_l) + A_s·functionalLN(s1))`

`V2_l = V_l + B·u_l`；K2、Q2、mask、w_qi(q2)、残差输入和原 prediction/post 路径全部保留。

这里 u_l 必须依赖各历史事件 H_l；只有依赖 s1 的统一 channel gate 可移到 `P@V` 之后，不能宣称改变了逐事件内容。上述非线性联合项一般不能移出历史加权求和，与旧 query correction/slot gate 不同。rank16、D384、无 bias 时三个矩阵合计 18432 个新参数，可用 B=0 作为原函数初始化目标，构造 RNG 隔离；实际初始化输出/梯度、参数顺序和 native PSGDKron 合约仍须验证，不能由 B=0 推出优化器轨迹一致。

必要的结构对照是同参数、同历史 H 输入但以固定 RMS1 常量替代 s1 的静态条件版本，再与原循环比较；这区分额外逐事件容量和首轮工作状态条件。不是用旧 query-mixing 的 joint/constant 结果代替新的 V 位置验证。若只胜过静态控制却不达原循环质量/成本规则，仍不构成优化候选。若动态与静态差异不明，也不能以非零残差或大梯度代替收益证据。

测量缺口：暂无证据显示原第二轮 V 内容是瓶颈，也未测这个逐事件映射的额外训练开销。直接实现至少增加 O(B·L·D·r) 的历史计算及激活；8192×100×384 的一个额外 BF16 历史张量约 600MiB。可以先在单独实现审查中评估是否值得原预算，不能把“低秩/参数少”写成零成本。旧 accepted second-read probe 明确要求两次 V 相同，不能用于此结构的归因；新参数还触及已记录的 HeavyBall 布局风险，不能自动套用旧参数修复规则。

候选二：**第二轮地址 K 只由物品字段和原位置构造，V 继续保留物品与历史 action 的完整内容。**

当前 [feature_map](workspace/qformer-loop/data/Taobao_Action/feature_map.json) 有六个历史 item 字段（item_id、cate_id、campaign_id、customer_id、brand、price_bucket）及 action；其 embedding 在 [_build_sequence_tokens](workspace/qformer-loop/taobao-transfer/recurrence-coverage-retrieval/source-snapshot/QFormerCross8.py:1083) 先拼成一个向量、共同投影/加位置/LN，随后同一个 H 同时进入 w_k 和 w_fi。因此检索地址和被读取的行为内容目前没有字段角色分离。

记 `U_l = W_I x_item,l + W_A x_action,l + b`，原 H_l=LN(U_l+pos_l)。只在第二轮另构造 `H_key,l=LN(W_I x_item,l+b+pos_l)`，令 `K2=k_norm(w_k(H_key))`，而 `V2=w_fi(H)` 保持原值；Q2 和第一轮所有计算维持原路径。物品字段仍为六字段完整集合，不另挑某任务字段；action 仅从第二轮 K 输入移除，始终保留在第一轮与第二轮 V 内。假设是“按物品/上下文寻址、读取其行为内容”可能比混合字段同时决定地址和值更合适，不是“action 无用”。

可复用原 sequence_projection 的 item/action 列块、bias、position、sequence_input_norm 和所有 attention 参数，不增加参数或更改注册顺序；第二轮功能从开始就不同，不要求它与原模型输出相等。第一轮仍要走原完整投影，不能拆分再求和后仅凭数学等价声称 bitwise 不变。第二 key 分支会多投影/LN及历史激活，具体实现/成本尚待审查。直接同 q 原循环是必需对照；未看到改善就舍弃此字段分配，不从某任务局部正向切换主目标。

这与 memory 拼接不同：事件数和有效 mask 完全不变，每个事件的 K/V 有明确不同字段职责；也与历史 writer 不同：不生成 query→history 更新，更不恢复 prediction→work 反馈。检查两份已完成 episode 账本及上述实际模型后，未发现相同的“第二轮 item-only K / full-content V”训练；只是查重结论，不是收益证据。

时间方向暂不作为第三个候选：当前可见的是原序列顺序与有效事件序号位置嵌入，hour/is_weekend 属于当前 context，feature map 没有逐历史事件 timestamp/elapsed-gap。[collator](workspace/qformer-loop/runtime/UniRank/unirank/pytorch/dataloaders/unirank_dataloader.py:525) 取历史前缀并保留原顺序，padding 逻辑见579行；这不足以实现真实时间间隔建模。历史长度分箱诊断也只测长度，不能证明新近事件或特定时间尺度最有用。若以后要时间分解，先核数据中真正可用的时间字段与截断语义；不能把序号当 elapsed time 或悄悄增加数据输入。

两候选共同的前置边界：原两轮/no-feedback、同 q 原循环和最佳单轮、seed20262027/2193updates/PSGDKron/LR/clip1 保持；不把 query slot 视为任务。K-only/V-only 指固定同一次 forward 的局部路径；训练后梯度、共享投影权重和整体函数都会变化。需检查 mask 孔洞、全 padding、单历史事件、历史 action 不含目标动作（collator 目标槽 action 为0，模型用 `[:-1]`）、异常恢复及 BF16/native 行为。已有 attention/direction-norm/readout 诊断不重复调度；旧 probe 的 shared K/V 与源码路径约束不满足时应拒绝而非放宽 guard。只有新的明确决策价值才考虑隔离适配器，当前没有“confirmed”机制或可消费的 edit handoff。

只读查重时间：2026-09-09T12:00:57.072821+00:00。本次仅消费小型源码/元数据、既有账本及已完成结果；未加载原始数组/checkpoint，未运行 forward、bootstrap、训练、GPU 或改全局 memory。查重状态 SHA：

- `../memory/episodes.jsonl`: `977aea1f556512b5dfec4ae73fb8198d51a3e4e9b57313aef9880655b38aaf5f`。
- `../../memory/episodes.jsonl`: `363d2eddf98e3e47eb384a8a8283bfb138df2fac2ceb213342517ea8763ff116`。
- `../recurrence-memory-sources/QFormerCross8MemorySources.py`: `7da32d4b5fab79d0c3b79afec0cbaa8543eaff356b287982ca714637653a2ed7`。
- `../../model_zoo/QFormerCross11SourceUpdate.py`: `176194c1188b051e730f0eb06726a7b0985fec691849fa34f6b9f38928e8c8ae`。
- `source-snapshot/QFormerCross8.py`: `4ef5c5c78f5ef658f24a4786c3927cf254279cf560b9b4746a0a7d9145b42a7f`。
- `../../data/Taobao_Action/feature_map.json`: `60c168c763257dbc9d2c4041c1c6d67ae5bbe71357f9ba559f6aa7f119a774dd`。
