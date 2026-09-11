# 按任务读取两轮预测状态：查重与证据边界

研究日期：2026-09-09。本文只记录备选假设，没有实现模型、固定新模型实验协议或安排训练。Root 后续已完成已有预测的逐任务循环收益与任务间差异分析，见 [TASKWISE_RECURRENCE_EVIDENCE.md](TASKWISE_RECURRENCE_EVIDENCE.md)。以下保留查重时的证据边界，新的分析仍不证明读出混合会有效。

**未发现严格等价的已训练方案：保留原两轮全部计算及预测状态不反馈到工作状态的设计，仅在各任务 tower 前分别学习首轮／末轮预测状态的读出混合，并从原末轮读出精确恒等初始化。** 查重范围为读取时淘宝 169 条、KuaiRec 233 条 episode，以及对应模型和已审阅的配置；这是当前可追溯记录中的结论，不把源码中的可选 API 自动计作已训练实验。

## 目标假设及其位置

以现有 no-feedback 两轮为例，省略 mask 记号：

```text
e  = L1(q, H)
z1 = L2(e, H)
y1 = L3(e, z1)
z2 = L2((e + z1) / 2, H)
y2 = L3(y1, z2)
原任务 t 的 tower 输入 = flatten(T(y2))
```

`T` 表示原末尾 post self-attention、归一化和 FFN 后处理。`y1/y2` 是这组后处理之前的预测状态；不能把原始 `y1` 与已经后处理的 `T(y2)` 当作位置对齐的表示直接比较。

待研究的自由度只在最终读出：两轮工作／预测更新均保留，第二轮仍重新读取历史；各任务能够补充多少首轮表示，由训练决定。它不等于每任务减少循环深度、不提供计算早退，也不把任务标签或验证统计输入循环核心。原计算路径见 [q4 no-feedback 源码](../recurrence-q4-capacity/QFormerCross8NoFeedback.py:15)；默认模型只将共享末轮表示交给各任务 tower，见 [上游读出源码](../../upstream/model_zoo/QFormerCross8.py:1152)。

## 本地近似探索为何不等价

| 已有项目 | 实际改变及已有结果 | 与本假设的边界 |
|---|---|---|
| `route_prediction_half` | 两轮内部均执行 `y = y + .5 * (L3(y,z) - y)`，所有任务共享；macro AUC 0.820982528。[源码](../recurrence-state-routes/QFormerCross8StateRoutes.py:12)、[结果](../recurrence-state-routes/RESULTS.md) | 连首轮状态及第二轮 L3 输入都改变，未保留原 `y1/y2` 后再选择读出。这项负结果应保留，但不能当作任务专属读出已失败。 |
| `dual-state` 与 `q4-capacity` 深度对照 | 全部任务统一执行一轮或两轮。q4_r1 为 0.826512190，ctx_q4 为 0.828472565，宏增量 +0.001960375。[q4 源码](../recurrence-q4-capacity/QFormerCross8NoFeedback.py:15)、[深度结果](../recurrence-q4-capacity/DEPTH.md) | 一轮版本实际去掉第二轮计算，且是独立训练的 checkpoint。旧 dual-state 还把预测状态反馈进第二轮工作输入；两者都没有每任务轮次读出。[dual-state 源码](../recurrence-dual-state/QFormerCross8DualState.py:18) |
| 每任务查询槽门、残差门、post-FFN 适配 | upstream 支持 `task_query`／`task_query_channel`，只缩放最终查询槽／通道；任务残差门也只读取最终展平表示。post-FFN scales 混合末尾 FFN 更新，任务 adapter 适配最终表示。[槽门源码](../../upstream/model_zoo/QFormerCross8.py:1225)、[post-FFN 源码](../../upstream/model_zoo/QFormerCross8.py:1166) | 没有 `y1` 或轮次维度；审阅的训练配置未发现启用这些任务专属开关，不能给它们编造训练结果。已分配但 strength=0 的任务残差门参数，也不构成实际启用。 |
| 第二轮输出投影适配 | `middle_r2_lora8_every_step` 仅在第二次 middle 调用为 `w_o` 增加 rank8 增量；增加 6,144 参数，macro AUC 0.826446913。[源码](../recurrence-adapter/QFormerCross8MiddleAdapter.py:31)、[结果](../recurrence-adapter/RESULTS.md) | 是所有任务共享的轮次专属变换，基于旧纯 middle 循环；没有各任务混合两轮状态。 |
| query correction routes／query mixing 及其 gate 诊断 | 修正第二次 L2 的共享查询或指定分支；mixing 在 `e/z1` 之间调节工作查询，最后仍只读 `y2`。八组 mixing 均低于对应原循环。[mixing 源码](../recurrence-query-mixing/QFormerCross8QueryMixing.py:70)、[结果](../recurrence-query-mixing/RESULTS.md)、[分路源码](../recurrence-query-routes/QFormerCross8QueryRoutes.py) | `query` 粒度是每个查询槽的系数，不是每个任务的循环深度系数；现有 gate shuffle 诊断也不回答任务对 `y1/y2` 的读出偏好。 |
| 深度辅助监督／混合深度训练 | `supervised12` 训练时重算一／两轮并加入首轮损失，推理统一两轮；macro AUC 0.825484529。[源码](../recurrence-training/QFormerCross8DepthTraining.py:35)、[结果](../recurrence-training/RESULTS.md) | 改变训练目标或全局执行深度，没有任务专属读出混合。 |
| KuaiRec conditioning／DepthNorm | 按重复轮次调节共享状态，或按总执行深度选末尾归一化；DepthNorm 仍共享循环核心及 tower。[conditioning 源码](../../model_zoo/QFormerCross11Conditioned.py)、[DepthNorm 源码](../../model_zoo/QFormerCross11DepthNorm.py:7)、[账本](../../memory/episodes.jsonl) | 属于共享循环条件化／深度归一化，没有淘宝四任务分别读取两轮预测状态的等价训练。 |

上述 AUC 均为已有固定 seed 结果，不把不同模型家族之间的数值相减当作匹配消融。

## 需要先澄清的读出对照

一个表示位置对齐的候选写法是：

```text
h1 = T(y1)
h2 = T(y2)
r_t = h2 + a_t * (h1 - h2)
prediction_t = tower_t(flatten(r_t))
```

这里的 `T` 使用同一套原后处理权重，原 `T(y2)` 路径保留。计算 `T(y1)` 增加一次 post attention／FFN 前向及相应训练图，不增加完整历史循环，但也不是零 FLOPs／零延迟改动；参数少不能代替实测计算开销。若改成先对原始 `y1/y2` 混合、再逐任务调用 `T`，则是另一个假设，后处理调用数也不同，不能与上式混用。训练时若后处理含 dropout，还需审计新增调用的随机数消耗及原路径恒等性，不能只凭代数公式宣称训练行为完全一致。

若后续证据支持进入训练，至少需要区分以下三个读出条件，而不是直接把任务专属版本与原模型比较后归因：

1. 原末轮读出：`r_t = h2`。
2. 共享混合：四任务共用一个 `a`。
3. 任务混合：各任务分别使用 `a_t`。

共享混合检查“补充早期状态是否有用”；任务混合相对共享混合才检查“任务专属读出自由度是否有额外价值”。按任务槽门、按轮次 query 适配和按任务循环深度读出是三个不同问题。这里不预定训练矩阵或宣布某项会改善 AUC。

## 原 y2 的精确恒等初始化

直接令残差系数 `a_t = 0`，上式在有限值输入下代数上回到原 `h2`；系数本身仍可获得关于 `h1-h2` 的梯度。也可考虑 `a_t = b * tanh(theta_t)`、`theta_t=0` 的有界残差参数化，但这允许负系数，不能称作严格凸混合。

普通有限 sigmoid／softmax 参数无法精确到达只读 `y2` 的端点。使用非常小的首轮权重只能得到近似恒等；不能把它报告成精确恒等。若要求权重严格位于 `[0,1]`，需另行明确端点和有效梯度如何兼容，不能通过零点导数为零的平方参数化默默得到无法启动的门。

最终若实现，应实测初始共享参数、随机数状态、输出及原路径梯度一致性，并确认新参数有有效梯度和实际优化器更新；本文没有完成这些验证，也不引入新的优化器规则。

## 现有证据能支持到哪里

后续保存预测分析提供了“任务从循环中获得的收益不同”的探索性证据：q4 与 q8 各自对照中，加购、购买相对点击的 AUC 效果差区间为正，但不能先指定点击用 `y1`、购买用 `y2`。q4 的独立训练一／两轮对照与 q8 的同 checkpoint 整轮删除回答的问题不同；q8 四任务的整体第二轮收益均得到区间支持。全部六组任务差值使用共享用户重采样，95% 区间未校正多任务或模型选择，不包含训练 seed 方差。[完整任务证据](../recurrence-task-effects/RESULTS.md)

即便任务间循环收益差异得到支持，它也不自动证明同一两轮模型的 `T(y1)` 能补充 `T(y2)`，或证明一个可训练的任务读出门会改善。固定 checkpoint 的首轮／末轮表示互补性、共享混合与任务混合之间的区别，仍缺直接证据。

已完成的 attention 专属干预支持保留原 q4.buy 的第二轮概率重分配：复用首轮 attention 的 ΔBCE 为 +0.000014312，预定 99.375% 区间 [+0.000008108,+0.000021102]；其他七项为 inconclusive。这既不指定任务应该读取哪轮状态，也不把其他任务判为不需要循环。当前读出备选保留两轮计算及原第二检索；它尚未得到实际收益验证。[attention 结果](../memory/tool-evolution/candidates/taobao-second-read-attention-v1/RESULTS.md)

当前状态：仅研究说明；模型实现、读出干预、训练、成本测量和收益结论均未开展。保持用户既有无新增训练 seed 的约束。
