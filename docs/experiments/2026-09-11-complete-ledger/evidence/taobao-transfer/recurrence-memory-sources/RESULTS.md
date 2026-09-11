# 两轮循环的读取来源

四查询和八查询各四组，均保持两轮，只改变第二轮工作或预测更新的memory来源；PSGDKron预条件器每步更新设置、clip1、学习率和seed固定。

| 结构 | 最佳单轮平均AUC | 循环平均AUC | 增量 |
|---|---:|---:|---:|
| memory_q4_work_entry | 0.828624 | 0.825296 | -0.003327 |
| memory_q4_prediction_history | 0.828624 | 0.827172 | -0.001451 |
| memory_q4_prediction_states | 0.828624 | 0.826791 | -0.001833 |
| memory_q4_prediction_entry | 0.828624 | 0.826079 | -0.002544 |
| memory_q8_work_entry | 0.828624 | 0.824274 | -0.004349 |
| memory_q8_prediction_history | 0.828624 | 0.824428 | -0.004196 |
| memory_q8_prediction_states | 0.828624 | 0.822584 | -0.006039 |
| memory_q8_prediction_entry | 0.828624 | 0.825531 | -0.003093 |

完整四任务AUC/logloss、显存和训练耗时见[结果](reports/summary.json)。[审计](reports/final-audit.json)。单seed结论，未评估测试集。八组均使用末轮损失；相同查询数的参数量不变，实际耗时单独报告。

同查询数原循环对照及耗时见 [解释](reports/interpretation.json)。八组均未超过匹配基线；保留原读取路径，下一步检验第二轮查询修正。仅限当前单 seed 和已测试的拼接位置。
