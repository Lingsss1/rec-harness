# 固定工作记忆的循环预测对照

GPU6，nf_fixed_memory，seed20262027。相对去反馈两轮候选仅将 entry_weight 从0.5改为1，参数、初始化、优化器、学习率、clip1、batch8192、2193更新、四任务末轮损失均固定。

e=L1(q,H)，z=L2(e,H)，y1=L3(e,z)，y2=L3(y1,z)，最后执行共享post attention/FFN与tower。复用已测试的NoFeedback实现；dropout=0使两次L2(e,H)严格相同。为保持对照调用预算仍计算两次L2，不声称提速。测试额外验证两次工作记忆精确相同，输出与复用同一个z的手工表达式相等，输入及L2参数梯度相符。

相对entry_weight0.5候选判断工作记忆变化是否必要；相对已完成dual_state_r1可判断固定记忆下第二次预测更新的效果，但两者计算预算不同。主对比仍为最佳单轮0.828623533。历史Cross8/Cross11只提供背景，不作为当前结构增益归因依据。

CPU/CUDA检查与真实batch8192 smoke通过后全量训练。独立编译缓存，不新增训练seed，不访问test；正增量做配对用户bootstrap。
