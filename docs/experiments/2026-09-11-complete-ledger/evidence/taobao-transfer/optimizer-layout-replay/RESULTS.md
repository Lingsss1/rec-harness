# 新参数布局会改变原参数更新：合成诊断结果

12 次 CPU eager 优化器单步全部完成。只读观测与原生 baseline 的参数、完整优化器状态和结束 RNG 完全一致；两种新增参数布局在 C/S/R 全部固定后，六个原参数的更新和优化器状态均逐位恢复，新增参数仍有有限、非零更新。

这验证了本地 HeavyBall 3.2.1 的实现机制，没有解释八组淘宝 AUC 下跌，也没有改变其 discard 结论。输入由公式构造：旧梯度约 0.25–0.50，新增共享标量梯度特意设为 4，任务向量梯度为 [1,1,1,1]。下面的差异大小不是实际训练梯度效应的估计。

| 固定的条件 | 单独改变的因素 | 保存结果 |
|---|---|---|
| 原参数 Q 初值、随机探针 | C：原参数裁剪后梯度 | `shared_011→111`；pre.weight 更新相对 baseline 的差异由 65.47% 变为 0 |
| 原参数裁剪后梯度、随机探针 | S：原参数 Q 初值 | `shared_101→111`；四个输出 bias 的更新差异由约 20.38%–25.08% 变为 0 |
| 原参数裁剪后梯度、Q 初值 | R：原参数得到的随机探针 | `shared_110→111`；work.weight 更新差异由 14.40% 变为 0，四个 bias 同样恢复 |
| C、S、R 均固定 | 两种新增参数布局 | shared/task 的六个原参数更新与状态全部恢复；新增参数分别有 1/4 个元素更新 |

完整八格 shared 因子对照及 task 的两个端点均保留，没有假定三个因素的效果可相加。固定 C 时，新旧参数仍处于同一桶，bias 桶的原生初始化尺度为 1.114319，baseline 为 1.304106；S 仅替换原成员的 Q。R 先执行原生抽样，再替换原成员探针，保留新成员探针。布局内的结束 RNG 相同，但不同布局的结束 RNG 与 baseline 不同；外层 fork/restore 则正常恢复调用前 RNG。因此不能把该干预称为整个随机轨迹保持不变。

实际执行父进程观测墙钟 3.0219 秒，worker 最后记录 CPU 3.3424 秒，采样进程树 CPU 3.66 秒；观测最大 RSS 694075392 字节，产物 7440813 字节。完整父子进程退出阶段 CPU 没有单独累计，不把这些数值冒充完整总 CPU。执行未触及模型、真实数据、GPU、训练更新或新训练 seed。13 项元数据检查及保存结果独立复核通过；后者核对了 1381 个张量载荷和 35 个分桶记录，没有重跑优化器。

通过真实 rec-harness 生命周期记录，候选 `probe-302bf4e60eb7a4e126b2d1b1` 当前为 **synthetic_pass**：paper_fidelity、synthetic_correctness、robustness、首阶段 cost 通过；原生真实数据 shadow 与 incremental_utility 仍 pending。没有 accepted、工具推广或模型修改 handoff。[方法与原始来源](../memory/tool-evolution/candidates/probe-302bf4e60eb7a4e126b2d1b1/method-card.md)

方法边界与 PyTorch 官方文档一致：相同 seed 并不保证跨平台执行可重现。CPU eager 的阳性结果不能替代 CUDA compiled 或实际梯度、多步训练的验证。[PyTorch reproducibility](https://docs.pytorch.org/docs/2.9/notes/randomness.html)

下一步研究应回到第二轮查询本身。查重确认尚未试过归一化后查询方向的解析变换；同时，共享 K/mask/scale 时，log P1 先验与 `Q2+αQ1` 的 softmax 等价，不应拆成两个新机制。12 个确定性数值样例验证了该等价，但不代表 BF16 逐位一致或模型收益。由于这类组合会同时改变方向与长度，优先检查现有导出能否支持方向／长度分离诊断，再确定固定权重对照；暂不从合成优化器结果直接启动模型改动。[查重与代数条件](retrieval-alternatives-audit.md)

[完整解释](reports/interpretation.json) · [实际结果](reports/replay-once/result.json) · [独立复核](reports/replay-result-review.json) · [数值等价样例](query-prior-equivalence.json)
