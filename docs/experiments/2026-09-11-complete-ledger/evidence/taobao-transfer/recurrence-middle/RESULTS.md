# 中间层循环对照

只改变循环位置及入口注入；默认PSGDKron、clip1、学习率、宽度和seed固定。

| 结构 | 单轮平均AUC | 循环平均AUC | 增量 |
|---|---:|---:|---:|
| middle_r2 | 0.827466 | 0.827503 | +0.000037 |
| middle_r2_anchor010 | 0.827466 | 0.826447 | -0.001020 |

完整四任务AUC/logloss、显存和训练耗时见[结果](reports/summary.json)。[审计](reports/final-audit.json)。单seed结论，未评估测试集。本轮推理增加一次中间层计算，尚需单独测速。
