# 并行PSGDKron对照

每组仅替换稠密优化器，结构、clip1、学习率和seed均保持各自基线设置。

| 结构 | AdamW平均AUC | PSGDKron平均AUC | 增量 |
|---|---:|---:|---:|
| taobao_cross8_clip1 | 0.823049 | 0.825738 | +0.002689 |
| t2_cross8_w384_loop1 | 0.824055 | 0.827466 | +0.003411 |
| t3_post_half | 0.824970 | 0.827088 | +0.002118 |

完整四任务AUC/logloss、显存和训练耗时见[结果](reports/summary.json)。[审计](reports/final-audit.json)。单seed结论，未评估测试集。推理图与对应AdamW基线相同，本轮未单独测速。
