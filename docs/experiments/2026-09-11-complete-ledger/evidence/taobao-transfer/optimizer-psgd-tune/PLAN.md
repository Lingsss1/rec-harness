# PSGDKron 单因素优化

固定当前最佳 384 单轮结构、dense LR 0.0001、sparse Adagrad LR 0.05、clip1、batch8192、2193 更新、seed20262027。对照是已完成 psgd_w384_single（AUC0.827466136），不重跑对照。

三组仅增加一个优化器参数：beta=0.95；momentum_into_precond_update=False；preconditioner_update_probability=1.0。原库默认 beta0.9，使用动量更新预条件器，更新概率前1000步为1后衰减。本轮不修改 precond_lr。

仅在独立 runner 中用 partial 传入选项，模型和共享工厂不变。记录并断言实际参数；每组先真实 batch8192 smoke，然后完整训练。任意失败停止对应组，不自动重试。全程不评估 test，无新 seed。
