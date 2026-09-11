# 第二轮查询修正放在哪条路径

此轮要检验：第二轮工作输入的修正是否需要同时进入检索、显式交互和残差，还是只有部分路径值得保留。八个候选均从原 seed 初始化重新训练；复用已经完整训练的 v2 `query_q4_work` 和 `query_q8_entry` 作为 `all` 路线对照，不重复训练 `all`，也不加载这些 checkpoint 作为新候选的初始权重。

## 证据和归因边界

rec-harness 的固定权重分路干预显示，q4 工作状态修正移除检索修正后 fav/buy 损失上升，移除残差修正后 fav/buy 损失下降；仅保留检索也改善 fav/buy。但 q8 入口修正没有重复这一检索收益，移除检索修正反而改善 buy，移除残差修正也改善 buy。q4 与 q8 在本轮同时具有不同的查询槽数和修正来源，因此不能把两者差异单独归因于槽数。

两个模型的全部修正置零后 buy 损失均改善，shuffle 在四任务中的区间均跨零。q4_work 的完整验证 AUC 仅接近原 q4，未证明提升或等价；q8_entry 低于原 q8。当前证据不允许将近似持平的 AUC 解释为已证实的样本特定检索收益，也不能预设选择某条路径后重新训练一定改善 AUC。

诊断证据见 [分路结果](../query-correction-diagnostics/RESULTS.md) 与 [完整干预解释](../query-correction-diagnostics/reports/interpretation.json)。诊断采用固定 5,514 行、4,647 用户的 CPU FP32 IPW BCE；区间仅条件于固定模型和选中样本，不含训练 seed 方差、分层抽样不确定性或多项比较校正。训练参考和配对结果见 [v2 结果](../recurrence-query-correction-v2/PARTIAL_RESULTS.md)。本轮延续 toolkit 的“干预提出假设、重新训练验证收益”流程，将结果与失败如实回写 memory。

## 模型与八组对照

两轮的状态定义保持不变：`e=L1(q,H)`，`z1=L2(e,H)`，`y1=L3(e,z1)`，`q2=(e+z1)/2`。共享 rank16 修正为 `delta=B GELU(A LN(source))`，LN 无可学习参数，B 零初始化，A 与 v2 初始化及 RNG 保持一致。q4 使用 `source=z1`，q8 使用 `source=e`。第二轮 L2 的历史 memory 仍为原始 H；`y2=L3(y1,z2)`，最终 self-attention 和 FFN 各一次，不恢复预测状态到工作状态的反馈。

| 候选后缀 | 第二轮 w_q 输入 | 第二轮 w_qi 输入 | 第二轮残差输入 |
|---|---|---|---|
| retrieval | q2+delta | q2 | q2 |
| interaction | q2 | q2+delta | q2 |
| residual | q2 | q2 | q2+delta |
| attention | q2+delta | q2+delta | q2 |
| all（复用 v2） | q2+delta | q2+delta | q2+delta |

每个后缀各有 `route_q4_*` 和 `route_q8_*` 一组。所有路线参数名、形状和总量相同：width384/rank16 新增 12,288 参数，q4 总计 51,736,516，q8 总计 81,628,612。第一轮 L2 继续走原 forward；仅第二轮选择修正进入的路径，保留原 qk norm、mask、temperature、投影、residual scale 和 dropout。

## 固定训练条件与检查

每组均为 Taobao_Action，seed20262027，17,822,296 训练行、2,871,767 验证行，batch8192、1 epoch/2,193 次更新、BF16、max_len100、两轮、无反馈。Dense optimizer 为 HeavyBall.PSGDKron，`preconditioner_update_probability=1.0`，dense LR=1e-4；sparse Adagrad LR=.05；clip1、weight_decay0。除模型名及 `correction_route` 外，每项配置逐字段与对应的 v2 q4_work/q8_entry 相同。不增加 seed，不读取测试集。

保留 v2 的 deferred-zero-A 处理：仅第 1 步严格全零 A 梯度设为 None；第 2 步必须正常激活，不允许通过持续跳过梯度使 smoke 通过。真实数据两步 smoke 要验证 B 第一步更新、A 第二步更新、四任务头更新、参数覆盖和 L2 的两轮路径。此处未规定对其他近零参数修改优化器；如出现失败，保留原始证据并单独处理，不能赋予虚构 AUC。

CPU 七项模型测试已通过。`all` 对 v2 的参数、RNG、输出和梯度为 bitwise exact，包含非零 B 和 dropout。B=0 时所有路线的初始参数、RNG、输出 exact；部分分路改变 FP32 梯度累加顺序，因此共享梯度仅数值一致：最大绝对差 5.722e-6，整体相对 L2 2.230e-7；FP64 最大差 1.776e-14。最大逐张量相对差来自原梯度范数约 3.044e-8 的 k_norm.bias，不能用它声称训练轨迹相同。非零修正的手工展开、梯度、无反馈、padding、标签隔离和 checkpoint 均有测试。CUDA 七项模型测试也已通过；真实 PSGDKron 两步 smoke 仍是每组训练的前置检查。

`gpu-map.json` 将八组分配到八张不同的 GPU。启动由主流程协调；GPU0 上尚未结束的差值重试拥有优先权，不能与其重复占卡。配置、模型、运行器、计数与参考身份完成冻结和审计后才能正式训练；本准备步骤没有启动训练。

## 结果要回答的问题

主指标为完整验证集四任务 macroAUC，同时报告 click/cart/fav/buy AUC 与 logloss。预设每个新候选都与同查询数、同来源的 v2 `all` 比较；再分别对照无修正原循环 `ctx_q4` 或 `explore_dual_no_feedback`，并保留最佳单轮 `psgd_precond_every_step`。参考配置、结果、checkpoint/预测文件及 manifest 哈希保存在 `route-references.json`。

八个 route 对 all 的配对区间均应报告，不按正负筛选；区间只反映固定 checkpoint 的验证用户重采样，不覆盖单 seed 方差或候选选择。若某条路线改善 all，却仍低于原循环，应报告为局部路径优化，不晋升为整体最佳。若 q4/q8 方向不同，保留各自条件结论；若各路线都无提升，拒绝“移除特定路径足以转化为训练收益”的当前假设。保持近似持平但计算开销合适的候选，不能把跨零区间当作等价证明。

同硬件、BF16、batch1024、同一验证批次比较前向耗时和显存，另报训练耗时；排除 IO 的微小前向差异不自动构成可靠加速。结果与失败均追加 rec-harness memory，区分同权重机制证据与重新训练的 AUC 证据，不改写历史失败或冻结文件。

GPU安排优先启动q4检索修正；q8 residual使用GPU0，等待明确要求的q4差值补跑释放该卡。其余七组可先使用GPU1—7。启动器逐卡等待空闲，避免同卡训练重叠。

仅对第2次更新时尚未初始化的A，沿用差值补跑已通过CPU/GPU五项测试的狭义nonzero-init guard：原PSGDKron先计算，若对exact匹配的有限非零A梯度误报全零，返回与原异常栈中已算出值完全相同的自动尺度；函数在上下文退出时恢复，事件逐条记录。其他参数、真零、非有限值、手动尺度或HVP分支不绕过。学习率、clip1、预条件器选项和模型数学均不更改。
