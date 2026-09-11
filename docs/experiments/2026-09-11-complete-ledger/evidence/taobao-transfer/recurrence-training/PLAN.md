# 循环训练目标实验

主对照 middle_r2=0.827502882；额外保留 Cross8、默认 PSGDKron 单轮和当前全局最佳单轮结果。

- middle_mixed123：按更新序号循环深度1/2/3，2193次更新分别731次；平均历史层调用数4，与固定两轮相同，但不声称实测FLOPs相等。
- middle_supervised12：共享全部权重，损失 (L_R2 + 0.25 L_R1)/1.25，每批仅一次反传和optimizer更新。零dropout下重算前缀产生相同中间读出；当前为两次完整前向，历史层调用数每批7，另有重复的embedding/输入和输出计算，不是同计算量对照。

两组固定两轮验证。原数据、seed20262027、width384、默认PSGDKron、dense LR1e-4、sparse Adagrad0.05、clip1、batch8192、2193更新不变。不评估test。
CPU与CUDA/BF16各3项测试通过：原生验证输出/参数一致，训练循环调用次数与一次clip/update，归一化辅助损失与梯度。每组先真实8192 smoke再完整训练。

论文依据与推论区分见 ../recurrence-research/INSIGHTS.md；上一轮深度诊断见 ../recurrence-middle/DEPTH.md。
