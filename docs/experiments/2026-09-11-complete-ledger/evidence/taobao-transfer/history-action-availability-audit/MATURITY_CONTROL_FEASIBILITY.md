# 历史 action 成熟度控制：实现边界

**状态：仅源码可行性分析；A 的实际样本结果尚未用于本说明，也未决定新训练。** 若 A 发现大量未成熟窗口，可构造“完整窗口成熟后才释放行为标签”的保守控制；它检验固定离线协议对窗口信息的依赖，不能直接识别线上无偏效果或实际泄漏率。

**时间规则。** 当前 [metadata](../../data/Taobao_Action/meta_data.json) 给出秒级时间、86400 秒窗口、`user_id/cate_id/brand` 匹配；[预处理:812](../../runtime/UniRank/data/Taobao/preprocess_Taobao_seq_action.py:812) 的 `searchsorted(side='left')` 加 `<= window` 实际包含两端，即 `[t_h,t_h+86400]`。令目标曝光时间为 `t_0`，仅当 `t_h+86400 < t_0` 时释放历史 cart/fav/buy 的完整组合，包括成熟后的真实零；等于边界仍视为未成熟，避免把目标同秒发生的终点行为当作请求前已知。仅在另有可靠事件/到达顺序时，才能论证放宽为 `<=`。按 [排序与 seq_len:885](../../runtime/UniRank/data/Taobao/preprocess_Taobao_seq_action.py:885)，同秒历史只是行号在前，不证明已发生：严格时序控制应排除 `t_h>=t_0` 的整条历史 token，保留真正较早的 item，且不回填更旧事件；这会改变有效长度和原模型位置编号，必须单独计数，并在比较双方使用相同规则。若选择保留同秒 item，只能称“released 行序前缀控制”。

**未知不能编码成负例。** 当前 action 有 16 个标签组合加 padding 0；`exposure=6` 是四标签全零，`is_click=9` 是其余三标签全零，二者都不是“未知”。最低改动是在保留原 1–16 编码的基础上，增加“延迟三行为未知、click 为 0/1”的两个非 padding 类别；共同 24h 窗口使这三个未知位同步。未成熟行隐藏三行为正值与零值，保留 item 和确实可用的 click；成熟行保留原完整编码。若 click 本身可知性未确认，还需独立的 click-unknown 状态，不能硬归 0。发布序列只有曝光时间及原 `clk` 标签，没有 click 发生/到达时间，故“历史 click 已可用”必须明确作为假设或由额外日志证实。仅两个新类别时 vocab 17→19，在当前 16 维 action embedding 下增加 32 个稀疏参数；这不是零参数或原优化轨迹不变的控制。

**模型和 loader 的真实改动面。** [collator:417](../../runtime/UniRank/unirank/pytorch/dataloaders/unirank_dataloader.py:417) 当前只加载 item/action 序列；需额外加载既有 `full_timestamp_seq`，按 `seq_len=l` 从同一用户全序列取得目标时间 `timestamps[l]`，并以完全相同的 `[max(0,l-100):l]` 切片对齐历史。目标 data 行没有原时间列，实施前必须验证序列长度、排序、目标 item 及映射一致性，禁止用目标标签推导可知性。同一历史事件随目标时间改变状态，因此不能永久改写一条 `full_action_seq` 充当所有目标的控制。保持目标 action=0、目标监督四标签、原行/用户身份和 split 不变。

当前 [task masks:539](../../runtime/UniRank/unirank/pytorch/dataloaders/unirank_dataloader.py:539) 只是 action 名称中的正行为指示，未知和成熟零都为 0；若供别的模型使用，应另传 knownness，不能当成带负例语义的可用性 mask。原 q4/q8 的 [get_inputs:325](../../runtime/UniRank/unirank/pytorch/models/rank_model.py:325) 默认忽略第四项，随后 [历史 builder:1083](../recurrence-task-query/source-snapshot/QFormerCross8.py:1083) 将 action embedding 与 item 字段拼接、投影、加位置并做 LN。**只改第四项 masks 不影响这两个 QFormer；必须改变历史 action 编码。** 普通历史 mask 仍表示 item token 是否存在，不应用它删除所有未成熟窗口，否则同时丢失 item 和 click。

**公平比较与不可识别项。** 冻结新 loader/编码、feature map、metadata、时间边界和转换指纹，连同实际用到的 timestamp 列所属 train/valid 文件纳入新输入闭包；旧发布文件保持原样。有关原父模型与未来候选都必须在相同新输入协议重新训练/验证，固定原 seed、优化器、预算和四任务主指标；旧 checkpoint 直接裁剪后的推理只能说明输入分布改变。若要分离“隐藏未成熟标签”的效应，父模型的 released-action 与成熟控制也应采用同一扩展词表、同秒处理及共同参数初始化方案，记录新增 embedding/优化器布局和成本，不能把旧父分数当作完全匹配对照。本说明不指定训练组数或启动顺序。

未成熟阳性可能早已发生；仅最终标签不能恢复它的实际事件时间，保守控制会舍弃部分本可用信息。成熟零也依赖日志完整性与到达延迟；曝光时间加 24h 只保证事件时间窗口结束，不能证明日志已到齐。类别/品牌代理标签、相邻窗口共享行为、click 时序、全数据词表与静态特征等限制仍在。A 的窗口统计无论高低，都不能单独量化 AUC 污染或证明新的检索机制。

范围：只读小 metadata、既有说明与源码；未读数组/权重/global memory，未 import/运行模型、测试、审计、GPU或训练，仅新增本文件。
