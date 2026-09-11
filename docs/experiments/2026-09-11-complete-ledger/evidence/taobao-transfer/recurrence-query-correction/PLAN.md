# 第二轮查询修正

依据 query-conditioning 的固定权重 Shuffle/Zero 诊断：入口与工作信息均有条件贡献，但尚未证明平均融合为瓶颈。此轮为可证伪的结构探索。

固定 q2=(e+z1)/2，加 B GELU(A LN(source))；rank16，B零初始化。source取z1-e、e、z1、(e+z1)/2，四查询和八查询各四组。mean为同参数非线性变换对照，避免把通用额外容量误归因为差分信息。层归一化无可学习参数，修正每个查询槽共享。width384时仅增加12288个参数。保留两轮、不反馈预测状态、不改变memory来源。

测试要求：原权重/RNG/初始输出和基础参数梯度一致；B首步可学习、后续A有梯度；非零修正后仍满足无反馈、padding隔离和checkpoint恢复；CUDA及真实PSGDKron smoke。初始化相同不意味着优化轨迹相同，新参数可能改变优化器随机数消费。

主指标完整验证macroAUC，四任务AUC/logloss及前向开销并列报告。固定seed20262027/2193updates/clip1/PSGDKron每步预条件/LR。原q4与q8双轮复用历史对照。当前memory-source八组完成并审计后再安排GPU；不覆盖已有冻结代码。不读取test。

现有rec-harness工具足以支持当前条件依赖测量；无tool-evolution ready handoff，故不伪造工具接受或handoff记录。训练结果另行决定保留与否。
