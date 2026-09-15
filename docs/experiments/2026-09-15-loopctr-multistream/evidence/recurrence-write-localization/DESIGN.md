# 动态写入：轮次定位与状态保留消融

用户要求进一步研究，并且每次实验更新血缘图。八张空闲GPU各运行一个新模型；原始Cross8与dynamic_write使用上一轮保存结果作固定参照，不重训，不新增训练seed。结构父模型是dynamic_write；Cross8是指标对照。输入沿用原始发布数据、17动作、无mature处理、width256、q8、两个工作轮次、预测状态不反馈工作状态。PSGDKron每步预条件、dense LR1e-4、sparse Adagrad .05、clip1、BF16、batch8192、2193步、seed20262027均固定。

| 新模型 | 唯一控制项（相对dynamic_write，组合组明确列出） |
|---|---|
| first_only | 仅首轮写入含输入相关投影，次轮保留静态写入偏置 |
| second_only | 仅次轮写入含输入相关投影，首轮保留静态写入偏置 |
| collapsed_write | 首轮预测之后、次轮之前合并两流为均值 |
| tied_write | 每轮两流写入门系数取均值，限制两流接收相同的额外写入 |
| no_static_routes | 将读取与旁路静态系数固定为零；两轮动态写入保留 |
| first_only_no_routes | first_only 加读取/旁路固定为零 |
| second_only_no_routes | second_only 加读取/旁路固定为零 |
| collapsed_no_routes | collapsed_write 加读取/旁路固定为零 |

保留相同517维扁平路由参数以控制初始化和PSGDKron参数布局。部分变体禁用部分参数的作用，不能宣称有效容量相等；tied_write使两组投影只通过组合发挥作用。减少的是信息流自由度，未必减少运行开销。所有原始参数必须与前轮Cross8/dynamic_write初始化摘要一致；full模式必须与前轮非零路由前向完全一致。

离线验证显式两流更新oracle的非零前向和全参数梯度、零初始化原生一致性、预测干预不改变工作输入。所有8组先各做2步原生8192行smoke，BF16零初始化原生输出一致、梯度有限、原attention及head和router有效更新，才允许正式训练。不自动重试训练；失败保留，血缘仍更新。

统计固定10个保存模型（2旧参照+8新模型）、全部45个两两差值、每组21个macro/任务AUC/BCE/任务间差值指标。500次共享用户聚类重采样、seed20260906；先检查全部用户/物品/长度/标签逐行一致和源AUC一致。只衡量本训练seed验证用户不确定性；不做多重校正，不称独立确认，绝不读取test。

决策：macro AUC为主；逐任务AUC/BCE和同卡同1024行推理成本为guardrails。相对dynamic_write的区间上界<0则质量退化；区间包含0且成本不增加可列近似候选；区间下界>0才是同seed下质量提升候选。相对Cross8单列，不将恢复旧基线误称超过最佳写入。轮次控制与合并消融用于机制排除，不能将门值大小当成收益证据。

每个worker启动及终止都刷新qformer-loop/lineage/update.py，收尾追加memory后再次刷新。数据协议不同的历史节点不得直接按AUC连成提升链。
