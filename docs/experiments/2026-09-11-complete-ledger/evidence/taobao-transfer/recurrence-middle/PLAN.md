# 中间层循环对照

源于 recurrence-research/INSIGHTS.md 的首轮可验证假设。保持默认PSGDKron、dense LR1e-4、sparse Adagrad0.05、clip1、width384和原seed20262027；2193更新、相同数据和验证协议，不评估test。

middle_r2: L1→L2→L2→L3→post。
middle_r2_anchor010: 第二次L2前以0.1混合固定入口e=L1(q,H)。

两个候选无新增参数，不改变原始历史K/V。R=1已通过与原最佳单轮模型等价测试。每组先真实batch8192 smoke，再完整训练；不重跑已有基线psgd_w384_single (0.8274661363436118)。同时参考原Cross8及已有tail/post循环模型。

独立模型文件位于本目录，不改变在跑的优化器实验所冻结的model_zoo源文件。GPU4/5用于本轮，GPU1–3旧实验继续完成。完成后检查同checkpoint的深度曲线及表征变化，再决定是否进行轮次独立归一化/辅助监督。
