# 原发布协议 Cross8：两流读写对照

用户明确要求取消mature。使用原17类action、原始UniRankDataloader、原始标签/历史构造；不调用maturity_dataloader，不增加24小时掩码。协议与旧原生Cross8相同，不把无成熟处理等同于已证明线上输入可用性。

固定8查询、width384、4heads、原Cross8所有其他组件、seed20262027、1epoch/2193更新/batch8192、BF16、clip1、dense HeavyBall.PSGDKron每步预条件LR1e-4、sparseAdagrad .05。所有模型重训一次，原Cross8和两轮去反馈同时作为当轮匹配对照。没有额外训练seed。

八组：cross8、no_feedback、static、dynamic_read、dynamic_write、dynamic_bypass、dynamic_all、collapsed_all。前两组没有router；后六组相同双流组织，5个静态标量。动态组另加对应行的输入投影；动态参数跨轮共享。新增有效参数分别5/389/773/773/1925/1925。动态vs静态不是完全等有效容量，报告参数/耗时；dynamic_all与collapsed_all参数相同。

每槽两个工作状态，不增加查询数量。a0=b0=e；第一轮共享工作子层读取两流融合，默认写入a，b保留入口。第二轮默认读(a+b)/2并将更新写入两流。第一轮预测读取a1，第二轮预测读取更新后两流均值；y1只流向y2，不流向任何工作状态。读取、写回、旁路混合系数分别有界，依赖当前两流均值的RMS归一化表示；没有使用标签作为路由输入。

collapsed_all在两轮之间将a1/b1都替换为它们的均值，第一轮预测已经生成。此对照保留router容量而移除可区分的跨轮双流状态。固定路由和所有路由零初始化时与no_feedback在数学上相同，表达式安排保证原生浮点回放精确一致，因此不额外训练固定等价模型。

范围是原工作attention残差边界上的最小多流实现。没有新增工作FFN/MoE、历史编码器或深度监督，不宣称完整复现LoopCTR全子层HCR。源码快照保存原Cross8及NoFeedback。

离线测试：非零显式两流前向和梯度oracle；零路由与原生精确一致；预测干预不改变工作输入；完整模型原参数、构造RNG与稠密参数注册。实际smoke：每组2个原生8192行更新；新模型额外在第一batch检查BF16原生预测回放；所有head、原attention、新router有效更新，梯度有限。全8组通过后才开始正式训练。

GPU0..3已有其他任务，不触碰；用空闲4..7按队列分两批。独立GPU锁；每cell外层worker监控资源并复用已验证的owned-process-tree/subreaper/pidfd清理。失败保留，不自动重跑；停止派发未启动cell，已运行独立cell允许正常结束。

训练后固定全部28个两两差值，macroAUC、4任务AUC、4任务BCE及直接任务效果差，共21指标/对照。复用500次共享用户聚类重采样，seed20260906，不读取test；先核对全部验证行ID、标签、原始输入和初始化。另在同卡同1024行batch下10预热+40计时/模型。macro为主，逐任务和成本为guardrails；不只报告正结果。

后续条件化诊断需要未用于上轮设计的新用户组，不能把已使用确认组再次称作新确认。当前先执行训练与完整配对，不把路由权重大小当作收益证据。
