# 循环核心更新对照

只改变第二轮更新幅度或中间层交互门控；默认PSGDKron、clip1、学习率、宽度和seed固定。

| 结构 | 固定两轮平均AUC | 循环平均AUC | 增量 |
|---|---:|---:|---:|
| middle_repeat_half | 0.827503 | 0.824857 | -0.002646 |
| middle_bounded_interaction | 0.827503 | 0.827147 | -0.000356 |

完整四任务AUC/logloss、显存和训练耗时见[结果](reports/summary.json)。[审计](reports/final-audit.json)。单seed结论，未评估测试集。两组训练和验证均固定两轮，末轮损失；参数量一致，实际耗时单独报告。
