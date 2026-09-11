**预执行统计审查：40 列设计足以比较两种指定替换，但不能直接证明“方向主导循环收益”。** 本文件依据根代理预定设计与已接受的 `bootstrap_user_contributions` API 编写；当前尚未读取新 `protocol.json`，未运行数据、前向或 bootstrap，未修改 memory。后续协议、导出器和聚合实现仍需独立核对。

固定原 q4/q8 checkpoint、5514 行、4647 位用户和原 16 strata/IPW。令四条件的逐行 BCE 为 `N=native`、`D=direction_first`、`R=norm_first`、`B=both_first`；Q 与 L2 范数均指原 q_norm 后的逐样本/head/slot 向量。条件分别为 `Q2`、`Q1*n2/n1`、`Q2*n1/n2`、`Q1`，其余检索输入、交互、残差及模型权重保持原样。

| 每个 checkpoint/task 的预定列 | 定义 | 正值的条件含义 |
| --- | --- | --- |
| direction−native | `d=D−N` | 在第二轮原范数下，替换为第一轮方向更有害。 |
| norm−native | `r=R−N` | 在第二轮原方向下，替换为第一轮范数更有害。 |
| both−native | `b=B−N` | 同时替换方向和范数更有害；受下述 fallback 限定。 |
| direction−norm | `c=D−R=d−r` | 指定方向替换比指定范数替换更有害。 |
| interaction | `i=B−D−R+N=b−d−r` | 两项联合替换的 BCE 增量大于两个单项增量之和。 |

这不是五个独立实验。两个模型×四任务×五列共 40 列，必须保留全部正、负和跨零结果。`c`、`i` 要在逐行/用户 contribution 层构造并共享每次抽样，不能用两个区间端点相减或“一项显著、另一项不显著”替代配对比较。负 interaction 可能是补偿/饱和，不能自动认作模块冲突；这是输出 BCE 上的交互，不是架构参数交互的证明。

统一使用 [已接受的公共函数](../memory/tool-evolution/candidates/taobao-second-read-attention-v1/src/aggregate_numpy.py:52)，不修改原 `SEED=20262027` 或 `REPLICATES=10000`。固定用户排序，把全部 40 列放入同一个 `4647×40` contribution 矩阵，调用一次并记录 column mapping/user-order/draw SHA。每列用户 contribution 为 `U/N_population * sum_selected_rows(weight * contrast)`；保持原固定总体分母，不能按 draw 重新归一化权重，也不能把同用户跨 strata 拆开重采样。原 IPW 样本和标签、ID、mask 顺序必须完全一致。

95% 区间仅探索描述。决策使用预定 family 40 的 **99.875% 双侧 percentile 区间**，线性分位数 `[0.000625, 0.999375]`，ε=`1e−6` BCE。外层按本轮水平取区间，不能继承旧 attention wrapper 的 99.375% 或伪造 native/identity 对照。Bonferroni 提供的是以各 bootstrap 区间有效为前提的名义多重比较控制，不是精确有限样本覆盖保证；10000 draws 每侧极尾仅约 6 个抽样值，贴近门槛时需显示未舍入端点，不因结果边缘而换 seed/扩 draw。

| 允许的陈述 | 必要联合条件（全部使用上述 family 区间） |
| --- | --- |
| “该 checkpoint/task 上，指定方向替换比范数替换更有害” | `lower(d)>ε` 且 `lower(c)>ε`。只证明本次替换比较，不能直接上升为整个第二轮的来源。 |
| “原第二检索的联合替换损失中，方向替换体现较大的贡献” | 前项成立，且 `lower(b)>ε`；同时完整报告 `r`、`i`，不忽略补偿或依赖另一因素的效应。 |
| “在这组替换和 ε 尺度内，方向贡献占主导、范数单项及非加性交互很小” | 再要求 `CI(r)` 与 `CI(i)` **整体包含于** `[−ε,+ε]`。这是预定尺度下的条件性界定，不是普遍等效、范数无用或方向唯一有效。 |

若 `r` 明确为负，原范数可能在此比较中有害；即使 `c` 为正，也应描述方向/范数的异号补偿。若 `i` 跨零但区间宽，不能当作可加；若 `d`、`r` 同为正且 `c` 跨零，只支持两项替换各有害，不能排序。点估计近零或区间跨零都不是等效；95% 排零而 family 区间未过门槛，只作探索线索。不得把结果不确定解释为方向、范数或循环“没有效果”。

上述方向与范数替换**没有按相同几何扰动大小校准**。即使联合门槛通过，也只涉及 Q1/Q2 的实际方向差和范数差，不能证明单位扰动的方向敏感性更强。尺度替换在固定 K 下相当于改变该 query 的 logits 尺度；它不是检验是否需要训练中的 q_norm 参数。两边相反或只单个任务通过时必须保留 checkpoint/task 范围，不能宣称“只有这一类任务获得信息”，也不能推断 AUC、重训练收益或新增检索信息。

fallback 应逐 `(row,head,slot)` 使用同一个联合条件 `n1<=1e−12 OR n2<=1e−12`；四模式同位回退 `Q2`，避免用各模式不同排除规则选择样本。记录分子/分母近零的交集与并集、fallback vector 数、涉及 row 数和这些 row 的 IPW 质量；一个 row 的部分向量回退不意味着该 row 全部干预为零。全 padding 和长度 1 的行仍留在固定估计总体中，不另行筛除。

因此 `both_first` **不自动等价旧 reuse-first**：旧 probe 对第二次 Wq 输入换成第一轮输入，没有这个小范数回退。仅在无 fallback（或所有 fallback 位的 Q1/Q2 本来完全相同），且模型/数据/精度/路径身份及输出核验一致时，才可作为旧行为的复现证据；不得新增第五次真实前向来临时补对照。若有实质 fallback，本轮估计针对含 fallback 的干预，不能把旧 q4.buy 结论当作已通过的本轮 `b` 门槛。

执行链复核应核：仅 q_norm 后第二 Q 被替换；非目标输出与 state/RNG 恢复；两家单次 loader traversal、四模式各 5514 行（合计 44112 condition-row forwards）；原样本/标签/IPW及所有任务一致；40 列一次公共 bootstrap、draw/结果身份绑定、固定预算和幂等失败边界。当前仅形成判读约束，不授权或启动任何执行。
