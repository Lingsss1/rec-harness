# 第二轮输出投影 LoRA

候选 `middle_r2_lora8_every_step` 在纯中间 R2 基础上，仅将第二次中间层调用的输出投影改为 W_o + B A。rank=8，alpha=8，主权重仍训练并共享。A 为 Kaiming 初始化，B 为零；不改变已有参数的初始化随机流。width384 增加 6,144 参数，预期总参数 81,622,468。

这是 Relaxed Recursive Transformers 的从头训练、单投影改编，不是论文的预训练模型转换或全投影 LoRA 复现。固定两轮，不声称支持任意深度外推。函数式替换只作用于第二次调用，无持久轮次状态；当前会物化 width×width 的权重差矩阵，耗时比较需包含此成本。

固定最佳单轮设置：PSGDKron 每步更新预条件器、dense LR1e-4、sparse Adagrad0.05、clip1、seed20262027、2193 更新、batch8192、相同 train/valid。基准为最佳单轮 0.828623533 和正在训练的同优化器纯中间 R2；Cross8 保留参考。不访问 test，不增加训练 seed。

CPU 三项和 CUDA/BF16 三项契约已通过：零初始化参数/RNG/输出/原参数梯度精确一致；非零 adapter 与显式独立第二层结果及输入梯度精确一致，原共享权重不被永久修改；adapter 梯度、优化器唯一完整覆盖、四任务更新、掩码与检查点恢复。测试中的优化器为小型 fixture 的 AdamW；仍需真实 PSGDKron batch8192 smoke，不能将此测试作为该优化器训练已经通过的证据。

尚未开始质量训练。下一步接入隔离 runner 与审计，再做 smoke 和完整验证。当前 recurrence-best 两组继续运行。若 adapter 带来增益，需要同有效深度非共享对照解释收益来源；若持平，按用户要求保留候选，但结合参数和耗时判断价值。
