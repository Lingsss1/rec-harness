# 四查询匹配消融与循环容量分配

复用 ctx_q4（AUC 0.828472565）和原八查询循环（0.828521213）作为对照，不重复训练。q4_r1 只去掉第二次工作/预测更新；它的预测层读取工作状态，与历史 Cross8 单次模型不同。其余七组保持两轮、entry_weight=.5、不反馈预测状态、末尾一次 attention。

四查询：单轮、post FFN ratio 2/8、全局 FFN ratio 2、入口三层。八查询：post FFN ratio 2/8、全局 FFN ratio 2。其余设置及 seed20262027、PSGDKron 每步预条件、dense lr1e-4、sparse lr.05、clip1、batch8192、2193次更新固定。容量改变会影响初始化随机数消费，不能宣称所有对应权重完全相同。

先完成 CPU/CUDA 精确对照和手工展开梯度检查，再冻结源码和配置。每卡独立 smoke、训练及编译缓存。审计四任务及数据哈希，保留 validation predictions，未读取 test。质量完成后同批量测前向开销。单 seed、不预设收益与统计等效。
