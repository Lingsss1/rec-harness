# 并行PSGDKron对照

每组仅修改一个 PSGDKron 参数，结构、clip1、学习率和seed均保持最佳384单轮模型设置。

| 优化器变体 | 默认PSGDKron平均AUC | 候选平均AUC | 增量 |
|---|---:|---:|---:|
| psgd_beta095 | 0.827466 | 0.827994 | +0.000528 |
| psgd_raw_precond | 0.827466 | 0.821979 | -0.005488 |
| psgd_precond_every_step | 0.827466 | 0.828624 | +0.001157 |

完整四任务AUC/logloss、显存和训练耗时见[结果](reports/summary.json)。[审计](reports/final-audit.json)。单seed结论，未评估测试集。推理图与对应默认PSGDKron基线相同，本轮未单独测速。
