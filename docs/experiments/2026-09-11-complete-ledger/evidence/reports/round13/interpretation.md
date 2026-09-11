## 结果与取舍

本轮八组均未超过父模型，AUC 首选仍为 r7_small_attention_first。r13_query_between_zero 达到 0.870695，较父模型低 0.00053949，较 Cross8/Cross11 高 0.00114755/0.00072116。其 UAUC=0.791353，较父模型高约 0.000516，但 Logloss=0.188660、NDCG=0.408685 均不占优。

历史 self-attention 的 warm between 方案 NDCG=0.410352，比父模型高约 0.000496，但 AUC=0.870188。保留这一排序取舍，不能把局部指标提升当作整体最优。新 writer 总参数 8.379M，比父模型增加约 3.25%。

目前 48 个后续结构未刷新 r7 AUC。下一步计划保持循环位置，改共享 block 的 FFN 为参数预算接近的门控变换，分别比较字段、历史及两者。该方向尚未训练。

同卡 BF16、batch1024 的 p50：r7_small_attention_first 12.59 ms；r8_first_loop1 8.45 ms；r13_query_between_zero 15.16 ms；r13_self_between_zero 16.78 ms。新分支的额外成本需与有限的指标取舍一起考虑。
