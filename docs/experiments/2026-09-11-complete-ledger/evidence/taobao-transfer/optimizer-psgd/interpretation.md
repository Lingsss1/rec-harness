# PSGDKron 首组结果

固定 t3_tail_half 结构、clip1、dense LR 0.0001、sparse Adagrad LR 0.05 和 seed 20262027，稠密优化器替换为 HeavyBall.PSGDKron。2193 步及完整验证已完成，审计通过。

Macro AUC 从 0.825441344 提升至 0.826217786（+0.000776442），是目前观察到的最高值。点击、收藏、购买 AUC 提高，加购略降；四项 logloss 均降低。

500 次用户配对 bootstrap 的 macro AUC 增量 95% 区间为 [-0.000191282, 0.001694871]，跨零，因此保留为优化候选，不能认定稳定优于 AdamW。该诊断固定 checkpoint，不覆盖训练随机性或候选选择偏差，没有增加训练 seed。

隔离依赖包含 HeavyBall 3.2.1、opt-einsum 3.4.0、NumPy 2.3.4；原共享运行环境未修改。训练含验证耗时 2114.39 秒，对照 1850.63 秒，非专门控制的速度实验。未重新测量推理速度，未使用 test。

另三组 Cross8 clip1、384 单轮和 post-half 的并行对照仍在运行，待完整结果后统一归档。
