# 第二轮读取来源

四查询/八查询各四组，复用已有同查询数双轮基线。固定两轮、无预测到工作反馈、末尾一次 attention、entry_weight=.5。第一轮保持 e=L1(q,H), z1=L2(e,H), y1=L3(e,z1)。第二轮仅改变一个 memory 来源：

- work_entry：L2 读取 concat(H,e)，历史 mask 后追加有效入口槽。
- prediction_history：第二次 L3 读取 concat(z2,H)，工作槽有效，历史保留 mask。
- prediction_states：第二次 L3 读取 concat(z2,z1)。
- prediction_entry：第二次 L3 读取 concat(z2,e)。

默认 control 两轮完全复现原 no-feedback。z2 查询始终为 (e+z1)/2；第二次 L3 查询始终 y1。参数量/初始化不变，不增加训练 seed。PSGDKron 每步预条件、dense lr1e-4/sparse lr.05、clip1、batch8192、2193更新不变。注意 concat 引入 token 数量差异，收益不能直接归因为语义互补；报告前向开销。

先测试手工展开、梯度、padding隔离、无反馈以及checkpoint恢复，再freeze、真实smoke、八卡完整训练。未使用test。
