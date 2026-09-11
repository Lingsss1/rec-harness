# 历史任务 prior：先核窗口标签在目标时刻是否可知

**当前不建议启动新的 action-prior 训练。** 新颖性不是眼下的主要缺口：历史曝光排在目标之前，不等于这条曝光的未来窗口标签已在目标时刻可知。先完成 N 四组既定训练与原定对比，下一步如获授权，只做预先限定的小样本 CPU 时间审计；本笔记没有执行或派发该审计。

## 已确认的代码与数据事实

- 实际 [meta_data.json](../../data/Taobao_Action/meta_data.json) 的 `vocab_size.action=17` 表示 **16 个组合类别＋padding 0**，不是 17 个非空类别，更不能按 action ID 的二进制位解码。例：`buy=1`、`exposure=6`、`is_click=9`。组合先由四标签形成、再按语义名称排序编码；见 [build_action_maps:285](../../runtime/UniRank/data/Taobao/preprocess_Taobao_seq_action.py:285)。metadata SHA 为 `10ae1cc1a019007b44c8e76ec0ffa35535ed5e3e972b89e9821ebd008ba3f971`，与 N 原参考输入身份一致。
- metadata 明示 cart/fav/buy 来自同一 `user_id/cate_id/brand` 的曝光后 **86400 秒**行为窗口。当前 [预处理函数:771](../../runtime/UniRank/data/Taobao/preprocess_Taobao_seq_action.py:771) 与此一致：789 行按上述键分组，812–821 行取不早于曝光的首个相应行为，时间差不超过窗口则置 1。`behavior_log` 不含广告 ID；这是类目/品牌级代理监督，**不是用户直接购买了该广告 ID 的事实**。
- [869–914 行](../../runtime/UniRank/data/Taobao/preprocess_Taobao_seq_action.py:869) 先标注整表，按用户/曝光时间/行号排序，再将四标签编码成 action；`seq_len=cumcount`，并保存 `full_action_seq` 和 `full_timestamp_seq`。因此相邻曝光可以共用一个更晚的行为作为窗口标签依据；目前只确认这种构造可能性，未统计实际发生情况，也未重放数据生成。
- [dataloader:417](../../runtime/UniRank/unirank/pytorch/dataloaders/unirank_dataloader.py:417) 只读用户 item/action 序列，没有读 timestamp；[530–551 行](../../runtime/UniRank/unirank/pytorch/dataloaders/unirank_dataloader.py:530) 先切历史，再由词表名称 `split('|')` 生成四个长 100 的任务 mask。目标 item 另追加，目标 action 固定为 0；[579–598 行](../../runtime/UniRank/unirank/pytorch/dataloaders/unirank_dataloader.py:579) 取 `seq[:l]` 或 `seq[l−100:l]`，排除了当前序号。**当前行标签没有直接进入 task masks，但这里没有按目标时刻检查历史 action 的窗口是否结束/阳性是否已发生。** 同一时间戳也仅按行号确定序列顺序。

## 这意味着什么，尚不能断言什么

若历史曝光时间为 `t_h`、目标时间为 `t_0`，`t_h < t_0` 不推出 `t_h + 86400 <= t_0`。窗口未结束时，最终 action 类别可能依赖目标之后的行为；一个阳性也可能早已发生在目标之前。**仅凭源码或窗口重叠，不能给出实际泄漏率、受影响 AUC 或归因当前任务收益差异。** 零标签同样是完整窗口后的结果，不能只审阳性 token 就证明全部历史特征在线可得。

原 q4/q8 与 N 都通过 [原历史 builder:1083](source-snapshot/QFormerCross8.py:1083) 使用 `[:, :-1]` 的 action embedding；[N forward:252](QFormerCross8TaskQuery.py:252) 使用同一输入，默认不读取第四项任务 masks。故已做比较仍描述**这个固定离线特征协议下**的预测效果，但“都用同一协议”不能推出线上因果收益，也不能把 cart/buy 的循环增益直接解释为真实可用购买历史带来的信息。无需据此抹去旧结果，必须保留作用范围。

## 新颖性查重与当前决策

149 次已入账训练的 family/change 与相关源码未找到“仅 R2 按任务 action 类型加有限 score prior”的同一实现。F 改历史 K/V 的字段投影供给；coverage 改首轮 attention 的跨槽统计；N 改投影后、q_norm 前的任务偏置，均不同于直接用 token 类型加 score。仓库 [INFNet:124–137](../../runtime/UniRank/model_zoo/INFNet.py:124) 已用第四项 mask 做任务历史池化，所以任务 mask 本身也不是新工具/新信息来源。本次停止展开 prior 公式和控制，不凭名称新颖追加训练。

**下一项需要的是小样本时间证据。** 如另行授权，应预先固定 train/valid 样本、用户和最大读取量；纯 CPU 对齐目标行 `seq_len` 与用户 timestamp/action 序列，复核目标排除、同时间戳、最近 100 条及 `t_h+86400>t_0` 的比例。这个比例只标记“窗口尚未成熟”，不能命名为实际泄漏率；若要判阳性在目标之前/之后发生，还需对应行为时间与原匹配键，缺失就记 unknown。不要顺手改数据或重训基线。

N 最终若 task 胜 shared 且原父对照/成本仍有候选价值，只能提高进一步研究任务条件的优先级，不能跳过上述可知性问题；若仅内部改善、两者仍负于原父，或任务差异无明确方向，不应继续这一支。当前 N 尚在训练，本笔记未读取其结果。任何后续仍保持原两轮、无 prediction→work 反馈、seed20262027、PSGDKron/原学习率/clip1/2193 更新；当前四组预算不变，无新 tool candidate/handoff/global-memory 修改。

范围：只读 metadata、源码和既有 notes/账本说明；零数组/权重读取、测试、forward、bootstrap、GPU、训练或外部浏览。只新增本说明文件。
