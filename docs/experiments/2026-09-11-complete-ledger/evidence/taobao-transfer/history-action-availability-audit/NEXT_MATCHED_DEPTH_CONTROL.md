# 最小匹配深度对照：成熟历史下的任务循环收益

**建议仅保留 q4 四格作为下一项待决策设计；目前不实施、不冻结协议、不启动训练。** 它能回答“在共同严格历史边界下，保守隐藏尚未成熟的行为标签，是否改变独立训练 R2 相对 R1 的任务收益”。它不能识别真实泄漏造成的 AUC 份额，也不是完整线上因果评估。当前 N 四组继续按原冻结协议运行，不能改输入或借此追加预算。

| 新训练格 | 深度 | 历史 action |
|---|---|---|
| released-R1 | 一次 work＋prediction | 原完整组合，采用下述共同时间 mask |
| released-R2 | 两次共享 work＋prediction | 同上 |
| mature-R1 | 一次 work＋prediction | 窗口未成熟则三行为 unknown |
| mature-R2 | 两次共享 work＋prediction | 同上 |

## 两个变量以外必须共同

四格均先取原 `[max(0,l−100):l]`，由同一用户 `timestamps[l]` 推定目标时间 `t0`；保留 `item>0 且 th<t0` 的 token。**同秒及更晚 token 全部屏蔽，不从更旧历史回填，不改原数据行的 seq_len/目标索引。** 原位置编码按有效 mask 的 cumsum 重算，四格使用同一路径；分别记录排除数、有效长度、空历史及时间边界。这个 released 是“共同严格历史条件下仍用 released action”的新对照，不是旧原样数据。

mature 仅在 `th+86400<t0` 时保留完整 cart/fav/buy 组合，包括成熟真零；等于边界仍视为未成熟。其余保留历史 item 与**假设已可知的** click，但把三行为整体标为 unknown，正值和零值都隐藏。共同词表保留原 1–16 类与 padding0，追加 click0/1＋三行为未知两类：**共19类**。unknown 既不是 exposure、已知行为零，也不是 padding；只改第四项 task masks 对原 QFormer 无效，必须改历史 action embedding 的输入编码。目标 action 仍为0，四项目标监督和原 train/valid 行身份不变，旧发布文件保持只读；转换随目标时间而变，不能永久改写整条用户序列代替逐样本转换。[可行性与时间限制](MATURITY_CONTROL_FEASIBILITY.md)

四格共同使用同一19类 feature map、action embedding 和 q4 构造路径。最小模型采用已验证的 [NoFeedbackStage](../recurrence-q4-capacity/QFormerCross8NoFeedback.py:7)，仅 `work_steps=1/2` 不同、`entry_weight=.5`；R2严格沿 `(entry+work)/2`，prediction 更新 `y1→y2`，不反馈到 work；两深度均保留原末尾 post 与四个完整原任务 tower，不加入 N task bias/四分支或新读出。19类比旧17类预计多32个稀疏 embedding 参数，最终须实际计数；四格全部初始参数/buffer、顺序和构造 RNG 应来自同一未训练初值并核验。不能称与旧17类模型具有同初始化、同分桶或同优化轨迹；各格实际梯度/clip/预条件器也会不同。

共同沿用真实旧 q4 R1/R2 的训练约定：seed20262027、2193更新、batch8192、单epoch、PSGDKron每步预条件、dense LR1e−4/weight decay0、Adagrad LR.05、clip1、BF16、原四BCE损失和等权macro主指标，原q4宽384/4head/4槽、相同入口/FFN/tower容量；无compile/checkpointing，无额外seed/test。旧两份 [manifest身份](../recurrence-q4-capacity/reports/q4-depth-identities.json) 实际均 `shuffle=True/drop_last=False`；沿用同一loader随机顺序，核对四格训练行/批次指纹，不误改为新shuffle策略。相同验证/保存时点、完整训练预算和数据转换分别固定；预计仅4×2193=8772正式更新，正式计算代价分别报告而不强称R1/R2同FLOPs。

## 决定性比较与判读

对任务 t 及 macro，定义 `D_released=metric(released-R2)−metric(released-R1)`，`D_mature=metric(mature-R2)−metric(mature-R1)`；**主交互量 `I=D_mature−D_released`**。四个新 checkpoint 在完全相同验证行/用户/目标标签上比较，复用原500次用户配对、seed20260906的共同抽样，直接由每次四模型的指标计算这三个量；保留四任务AUC/BCE、macro AUC与全部六项直接任务效果差。不要从“某任务显著、另一任务不显著”推出任务差异，不把两个独立区间简单相减当 I 的区间；区间仍仅为固定checkpoint的探索性95%，不含训练seed方差或多重/选择校正。

- `D_mature`仍有正向AUC证据，才支持这个保守输入协议下对应任务的循环收益仍可观察；`I<0`同时成立只说明收益减弱，并不表示完全消失。
- `I<0`不能量化实际泄漏影响：保守隐藏也会丢掉本来已在目标前发生的阳性，表示/学习轨迹随之变化。I跨零不是“完全不受影响”；D_mature跨零也不是“任务没有信息”。
- 任务异质性需看 D_mature 的直接任务差，以及 I 的任务差；点击不是无需审计的完美负控制。旧 q4 分数只作历史背景，**released两格也必须重训**：共同19类初始化与严格同秒 mask 已改变原对照；不能用旧R2−R1代替新 D_released。新协议分数也不直接晋升或取代旧协议 incumbent。

四格足以识别上述“严格历史前提下、成熟编码×训练深度”的统计交互；不分解同秒排除效应，也不需要为此扩成八格。目标时间仍来自 user_info[l]，缺独立事件时间交叉验证；保留click只是无点击事件/到达日志下的明确假设，窗口结束也不保证日志已到齐。类目/品牌代理标签、其他特征可知性和保守unknown造成的信息舍弃继续存在，故不得命名为“完整线上去泄漏”对照。

本次只读既有配置/manifest、源码、metadata与可行性说明，未读取数据数组或权重、执行测试/模型/统计/GPU/训练，未写全局memory；只新增本设计说明。实际实现、数据版本、输入闭包、统计/资源预算和准入检查须由根节点另行决定并冻结。
