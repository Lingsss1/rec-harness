# 工作与预测循环路径对照

基准为去反馈width384两轮候选0.828521213。主对比最佳单轮0.828623533。CPU/CUDA控制模型应精确复现候选的参数、RNG、预测与梯度。每组仅启用一种模式；固定width384、参数81616324、seed20262027、PSGDKron每步预条件更新、dense LR1e-4、Adagrad0.05、clip1、batch8192、2193更新、BF16、末轮四任务损失、train/valid数据，不访问test。

| GPU | 模式 | 变化 |
|---|---|---|
|0|anchor0|第二轮工作状态输入仅用上一轮z，取消入口混合|
|1|work_half|仅第二轮工作更新改为z+0.5*(F(x,H)-z)|
|2|prediction_half|两轮预测更新均改为y+0.5*(G(y,z)-y)|
|3|prediction_reset|每轮预测都从固定入口e读取z；取消预测状态跨轮传递|
|4|work_norm|仅第二轮工作输入做无参数LayerNorm|
|5|prediction_norm|两轮预测query均做无参数LayerNorm|
|7|inner2|每次预测更新前执行两次工作更新，共4次F和2次G|

GPU6另跑固定工作记忆对照。prediction_reset第一轮预测输出不参与最终损失，保留调用以明确路径消融；inner2增加计算但不增加参数。各组不具备相同FLOPs，报告实际耗时，不将参数相同误称计算相同。归一化是待检验干预，不预设状态崩塌已经发生。正增量做500次配对用户bootstrap，单seed、多候选选择的局限保留。
