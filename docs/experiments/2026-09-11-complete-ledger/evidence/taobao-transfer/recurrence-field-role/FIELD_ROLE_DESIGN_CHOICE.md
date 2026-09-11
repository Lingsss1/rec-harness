> Historical design evidence from the completed coverage round; this is not a field-role training result. Original: [FIELD_ROLE_DESIGN_CHOICE.md](workspace/qformer-loop/taobao-transfer/recurrence-coverage-retrieval/FIELD_ROLE_DESIGN_CHOICE.md).

# 下一轮字段分工的条件设计

当前 coverage 八组已完成训练、质量审核和延迟独立复核，全部未达原定候选门槛；16组配对尚在进行。本文记录 root 根据本地 memory、源码查重和独立设计意见形成的下一轮优先方案，尚未创建下一轮协议、实现或训练。先完成当前配对与记忆归档；已有研究授权继续有效，无需另行确认。

优先零新增参数的第二轮字段分工，暂缓新增18432参数的逐事件V残差。原因是前者直接检验检索地址与读取内容的职责，同时保留原参数集合；后者还涉及尚未在真实梯度下验证的PSGDKron布局作用，不能把这一混杂当作模型机制。零新增参数也不表示梯度、优化器动态或计算量相同。

每个原q4/q8父模型各四组：

| 名称意图 | 第二轮 K 输入 | 第二轮 V 输入 | 问题 |
|---|---|---|---|
| 物品地址 | item | full | 只改变定位历史的字段是否有益 |
| 行为内容 | full | action | 只改变取回内容的字段是否有益 |
| 组合分工 | item | action | 两项改动能否组合 |
| 反向分工 | action | item | 分工方向是否有区别；反向若胜出也按原门槛保留 |

原full/full父模型复用。前两个单改、组合和原模型形成2×2比较；反向组同时改变两条路径，不能独立归因。不得由组合只胜单改就称它优于原模型。

字段表示沿用原权重，但在原sequence_input_norm之前分别构造：

- Hfull = LN(W_item·x_item + W_action·x_action + bias + scaled_position)。首轮必须调用原完整构造代码，不能拆分矩阵求和后声称bitwise相等。
- Hitem = LN(W_item·x_item + bias + scaled_position)。
- Haction = LN(W_action·x_action + bias + scaled_position)。

三者均应用原有效历史mask。LayerNorm是非线性的，Hfull不等于Hitem+Haction；对照改变字段和表示尺度/方向，不能称纯信息删除。当前六物品字段各16维共96列，action16维；共享投影为384×112。action词表含padding共17类，Haction还含位置和共享bias，不能把它误当四个任务标签或单个常量。源feature map和_build_sequence_tokens已只读核对；完整可行性仍由FIELD_ROLE_FEASIBILITY.md补齐。

仅第二次work层cross-attention（stage.layers[1]）替换keys和/或values原始输入。原接口已有独立(queries,keys,values,mask)，优先局部pre-hook，避免重写SDPA、w_qi和残差路径。入口、首轮work、首轮prediction、Q2形成、两轮/no-feedback和末尾post均保持原结构。训练后共享权重与所有预测自然改变，不能把局部路径控制写成全模型数值不变。

可行性审查和原生验证要覆盖：两次work读取计数、仅第二次选定路径替换、参数/缓冲区/构造RNG/初始优化器布局保持、孔洞/全padding/单历史事件、position累计有效事件语义、target槽action不进入history、原生BF16/FP32的实际dtype、异常恢复及缓存清理。第一轮必须与同权重原模型匹配；新增角色的第二轮允许从初始时就不同。组合和反向组需要两个alternate历史表示，必须报告额外投影、LN、激活和延迟。

保持seed20262027、2193updates、PSGDKron every-step、原学习率、clip1、BF16和原四任务macro AUC。每q建议保留四组对父模型、两个单改彼此、组合分别对两个单改、反向对组合，共8项预定配对；两q合计16项。若正式采纳，协议须在新测量前明确角色、顺序和条件best-single比较，不从当前文档直接执行。

可另预声明组合量 I=AUC(item,action)−AUC(item,full)−AUC(full,action)+AUC(full,full)，使用这些固定模型共同用户抽样的保存draw逐项计算，无须增加训练seed或重新抽用户。不能相减区间端点，不能将I解释为同权重机制；四任务AUC/BCE和六任务差值仍全部报告。该统计在下一轮协议中正式冻结前不运行。

已有反证继续有效：淘宝八组memory拼接没有改逐事件K/V；KuaiRec r13的八组query/self-history写回同时改变K/V且增加writer参数、全部低r7。本方案与它们不同，但没有已验证收益。旧accepted attention工具要求共享K/V或原源码路径，不可放宽guard直接套用；只有会影响下一步决策的新测量才另行验证适配器。

来源：[NEXT_STRUCTURE_AUDIT.md](workspace/qformer-loop/taobao-transfer/recurrence-coverage-retrieval/NEXT_STRUCTURE_AUDIT.md)、[DIAGNOSTIC_FOLLOWUP.md](workspace/qformer-loop/taobao-transfer/recurrence-coverage-retrieval/DIAGNOSTIC_FOLLOWUP.md)、本轮[点估计](workspace/qformer-loop/taobao-transfer/recurrence-coverage-retrieval/POINT_RESULTS.md)。根代理条件选择记录；不是训练结果、ready handoff或已冻结实验协议。
