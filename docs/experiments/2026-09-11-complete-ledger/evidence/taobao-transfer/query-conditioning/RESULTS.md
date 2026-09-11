# 第二轮查询初步诊断

已实际运行 rec-harness diagnose_collapse.py，比较固定四查询双轮checkpoint的第一/第二次工作层输入。4096行、375用户，CPU float32、无权重更新。

槽均值有效秩52.60→49.53，第一主方向方差占比10.00%→14.99%；slot0同方向变化。两轮对应查询平均cosine0.9094，相对L2变化0.5331。表示全部有限、无死维度；不能因工具的medium启发式标记宣称塌缩，更不能从几何指标推断检索质量。

子集click193/cart11/fav11/buy4例，不能据此排序稀疏任务机制。下一步固定分层子集，做第二轮入口/工作信息shuffle与zero，使用工具的paired分析。保留正在运行的八组实验。

memory已补三条有作用域的循环结论；历史episodes未改写。回滚快照见memory-before/。

## 配对干预完成

5514条/4647用户的固定分层子集，逆抽样概率加权，两个checkpoint。打乱第一轮工作信息在两个模型的click/fav/buy上均有正的损失增量95%区间；打乱入口信息在fav/buy上也为正。cart区间跨零。不变对照严格为零。

保留两种信息；下一步仅测试零初始化低秩查询修正，使用入口/工作单独条件作匹配对照，不能预设提高AUC。置信区间仅条件于固定子集和checkpoint，不覆盖抽样设计及训练随机性。完整表见reports/paired-query-parts/toolkit/perturbation_report.md。
