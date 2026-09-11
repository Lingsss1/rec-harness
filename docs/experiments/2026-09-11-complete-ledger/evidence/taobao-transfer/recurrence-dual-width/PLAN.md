# 双状态两轮的宽度探索

利用完成诊断后空出的 GPU4/5，补充 width256 与 width512 两组。复用冻结的 QFormerCross8DualState，固定两轮；相对已完成的双状态 width384，只改变 token_dim。对应参数量54,763,844和118,757,956，width384参考81,616,324。仍使用原四任务tower和embedding_dim16。

两组均固定 PSGDKron每步预条件更新、dense LR1e-4、sparse Adagrad0.05、clip1、seed20262027、2193更新、batch8192、相同train/valid。不评估test，不增加seed。每个实验独立编译缓存，避免共享缓存状态影响启动；编译缓存目录记录在worker输出。

主要比较最佳单轮0.828623533，另比较同结构width384的0.827393887，以检验容量变化。这不是循环贡献的单变量对照；若某宽度获益，再补同宽度一轮对照。已有同结构1/2轮初始化、手工展开、输入和权重梯度、标签隔离与检查点测试；本轮额外在实际256/512宽度检查CPU/CUDA四任务更新和padding不变性。随后真实batch8192 smoke验证PSGDKron、参数量和状态读取路径，通过后完整训练。
