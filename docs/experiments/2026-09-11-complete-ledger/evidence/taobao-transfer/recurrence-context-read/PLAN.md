# 上下文回读 × attention位置

原控制复用已训练的explore_dual_no_feedback，平均AUC0.828521213；主指标参考最佳单轮0.828623533。不重复训练同一baseline，不增加seed。新实现无回读、末尾attention必须在CPU/CUDA精确复现原控制的参数、RNG、预测和梯度。

令N为已构建的用户/物品/上下文字段token，H为历史；C共享入口ns_qformer最后一层的全部权重。e=L1(q,H)，z1=L2(e,H)，y1=L3(e,z1)。有回读时 z1'=C(z1,N)，否则z1'=z1。第二轮 z2=L2((e+z1')/2,H)，y2=L3(y1,z2)。中间attention作用于y1；末尾attention作用于y2；两处均做时共享原post_self_attn和post_self_norm。最终post FFN仅一次。y从不参与z更新；不重建embedding，不新增参数、不创建第二份共享模块注册。

GPU0回读+末尾；GPU1无回读+中间；GPU2回读+中间；GPU3无回读+两处；GPU4回读+两处。原控制加前三组构成用户指定2×2对照，后两组检验保留末尾attention是否必要。组合不预设收益相加。GPU5/6/7补充原路径查询4/6槽、入口1层，所有形状调整只限全1融合数组的对应长度。

固定两轮、seed20262027、PSGDKron每步预条件更新、dense LR1e-4、Adagrad0.05、clip1、batch8192、2193更新、BF16、原train/valid；不读取test。五组核心实验参数均81616324；额外C读一次字段，middle只迁移一次attention，both执行两次，需分别报告训练耗时及推理开销。正确性测试检查手工展开、共享层梯度、mask与标签隔离、checkpoint恢复以及改动y不影响第二轮工作状态的结构约束。真实smoke检查历史/工作读取顺序、字段读取次数与attention调用数。
