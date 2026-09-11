# KuaiRec 第十三轮：两轮之间的历史源更新

八组均沿用已有 seed20262027、冻结 kuairec-loop-v1 与一轮训练预算；没有新增 seed 或测试集评估。Cross8 为主基线，Cross11 全 Query 为优化参考。

保留父模型 N S A N S 两次共享 Query 循环。额外 writer 仅更新本次 forward 的历史表示，不修改数据或缓存。

## 冻结结构

| 模型 | 写入位置/来源 | 初始强度 |
|---|---|---:|
| r13_query_between_zero | query_between | 0.0 |
| r13_query_between_warm | query_between | 0.1 |
| r13_query_after_field_zero | query_after_field | 0.0 |
| r13_query_after_field_warm | query_after_field | 0.1 |
| r13_self_between_zero | self_between | 0.0 |
| r13_self_between_warm | self_between | 0.1 |
| r13_self_before_zero | self_before | 0.0 |
| r13_self_before_warm | self_before | 0.1 |

query_between 使用第一轮 A 后的 Query；query_after_field 使用第二轮字段读取后的 Query；self_between 对两轮之间的历史做 self-attention；self_before 在首次读取前更新历史并供两轮使用。写入只执行一次，诊断第三轮复用更新后的历史。
Writer = history + scale × masked attention(LN(history), LN(source), LN(source))。padding 的写入为零，历史 self-attention 屏蔽无效 key；输入只含严格早于当前样本的事件。注意力和 norm 从父模型复制并独立训练，无额外 RNG 初始化。
每组新增 263937 参数，总计 8379410。零初始强度保持父模型初始函数及共享参数梯度，scalar 先学习、注意力随后获得梯度；0.1 初始强度改变初始函数。

## 同 seed 完整验证

| 模型 | AUC ↑ | Δ父模型 | ΔCross8 | ΔCross11 | Logloss ↓ | UAUC ↑ | NDCG@10 ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| cross8_original | 0.869547 | -0.001687 | +0.000000 | -0.000426 | 0.189132 | 0.788347 | 0.404062 |
| baseline_all | 0.869973 | -0.001261 | +0.000426 | +0.000000 | 0.189433 | 0.786362 | 0.402129 |
| r7_small_attention_first | 0.871234 | +0.000000 | +0.001687 | +0.001261 | 0.188067 | 0.790837 | 0.409856 |
| r8_first_loop1 | 0.870096 | -0.001138 | +0.000549 | +0.000122 | 0.188919 | 0.792501 | 0.412650 |
| r9_feedback_omit | 0.869712 | -0.001522 | +0.000165 | -0.000262 | 0.188932 | 0.788871 | 0.405604 |
| r13_query_between_zero | 0.870695 | -0.000539 | +0.001148 | +0.000721 | 0.188660 | 0.791353 | 0.408685 |
| r13_query_after_field_warm | 0.870311 | -0.000923 | +0.000764 | +0.000337 | 0.188634 | 0.789468 | 0.405682 |
| r13_self_between_zero | 0.870214 | -0.001020 | +0.000667 | +0.000240 | 0.189123 | 0.790487 | 0.408627 |
| r13_self_between_warm | 0.870188 | -0.001046 | +0.000641 | +0.000215 | 0.188870 | 0.790413 | 0.410352 |
| r13_query_between_warm | 0.869829 | -0.001405 | +0.000282 | -0.000145 | 0.188862 | 0.790011 | 0.409168 |
| r13_self_before_zero | 0.869720 | -0.001514 | +0.000173 | -0.000253 | 0.189065 | 0.790019 | 0.409198 |
| r13_query_after_field_zero | 0.869507 | -0.001727 | -0.000040 | -0.000466 | 0.189176 | 0.790804 | 0.409072 |
| r13_self_before_warm | 0.869198 | -0.002036 | -0.000349 | -0.000776 | 0.189062 | 0.789274 | 0.406719 |

同一 kuairec-loop-v1 输入哈希、1 epoch/1240 更新、batch8192、AdamW1e-4/Adagrad.05、BF16、clip1、全 Query。原始 Cross8 保留 clip10。

## 同卡前向与训练成本

| 模型 | 参数/M | p50/ms | p95/ms | 训练含验证/s | 峰值显存/GiB |
|---|---:|---:|---:|---:|---:|
| cross8_original | 9.232 | 13.73 | 13.82 | 448.7 | 16.67 |
| baseline_all | 11.408 | 15.10 | 15.23 | 495.1 | 18.88 |
| r7_small_attention_first | 8.115 | 12.59 | 12.67 | 441.6 | 15.74 |
| r8_first_loop1 | 8.115 | 8.45 | 8.47 | 337.0 | 9.81 |
| r9_feedback_omit | 8.115 | 11.19 | 11.23 | 396.1 | 13.70 |
| r13_query_between_zero | 8.379 | 15.16 | 15.21 | 499.4 | 19.43 |
| r13_query_after_field_warm | 8.379 | 15.18 | 15.24 | 501.4 | 19.37 |
| r13_self_between_zero | 8.379 | 16.78 | 16.96 | 541.0 | 21.16 |
| r13_self_between_warm | 8.379 | 16.77 | 16.86 | 546.8 | 21.16 |
| r13_query_between_warm | 8.379 | 15.16 | 15.21 | 506.5 | 19.43 |
| r13_self_before_zero | 8.379 | 16.76 | 16.82 | 537.7 | 21.16 |
| r13_query_after_field_zero | 8.379 | 15.15 | 15.19 | 499.5 | 19.37 |
| r13_self_before_warm | 8.379 | 16.77 | 16.92 | 535.5 | 21.16 |

同一 A800、BF16、batch1024，10 次预热、40 次固定真实批量前向；排除 IO，不代表服务端到端延迟。训练墙钟受同时运行的任务影响。

## 正确性与复现

CPU/GPU 各 5 项测试通过：零强度父模型输出/梯度兼容含 BF16，新增参数唯一注册及两阶段实际学习，遮罩/空历史/Query 反馈梯度，真实写入时机和第三轮源复用，最终输出干预与非法配置拒绝。
[计划](reports/round13/PLAN.md) · [配置](reports/round13/configs.json) · [模型](model_zoo/QFormerCross11SourceUpdate.py) · [结果](reports/round13/summary.json) · [训练状态](reports/round13/training/status.json)

## 结果与取舍

本轮八组均未超过父模型，AUC 首选仍为 r7_small_attention_first。r13_query_between_zero 达到 0.870695，较父模型低 0.00053949，较 Cross8/Cross11 高 0.00114755/0.00072116。其 UAUC=0.791353，较父模型高约 0.000516，但 Logloss=0.188660、NDCG=0.408685 均不占优。

历史 self-attention 的 warm between 方案 NDCG=0.410352，比父模型高约 0.000496，但 AUC=0.870188。保留这一排序取舍，不能把局部指标提升当作整体最优。新 writer 总参数 8.379M，比父模型增加约 3.25%。

目前 48 个后续结构未刷新 r7 AUC。下一步计划保持循环位置，改共享 block 的 FFN 为参数预算接近的门控变换，分别比较字段、历史及两者。该方向尚未训练。

同卡 BF16、batch1024 的 p50：r7_small_attention_first 12.59 ms；r8_first_loop1 8.45 ms；r13_query_between_zero 15.16 ms；r13_self_between_zero 16.78 ms。新分支的额外成本需与有限的指标取舍一起考虑。

## rec-harness 诊断

2 个候选导出完成，18 个工具调用成功；JSON 与 Markdown 均已读取。每个候选使用固定 57,408 行验证子集，表示/梯度为其中 7,232 行。checkpoint 未变化，未使用测试集。

| 模型 | 子集原始 AUC | 关闭额外 A | 推理一次循环 | 推理三次循环 | 最终 Q0 有效秩 |
|---|---:|---:|---:|---:|---:|
| r13_query_between_zero | 0.861855 | 0.858125 | 0.818278 | 0.819705 | 11.33 |
| r13_self_between_zero | 0.862038 | 0.860515 | 0.819785 | 0.807515 | 11.23 |

- `r13_query_between_zero` 关闭 A 的分块平均 Logloss 增量 +0.003252，95% block CI [+0.002597, +0.003911]；八个 Query 中 8 个 Shuffle 后子集 AUC 下降。最终 Q0 的工具标记为 medium，有效秩为 11.33/256。
- `r13_self_between_zero` 关闭 A 的分块平均 Logloss 增量 +0.001450，95% block CI [+0.000917, +0.002050]；八个 Query 中 8 个 Shuffle 后子集 AUC 下降。最终 Q0 的工具标记为 medium，有效秩为 11.23/256。

推理深度扰动只反映固定权重对训练结构的依赖；与本轮从头训练的结构对照分开解释。有效秩和局部梯度是机制线索，不能直接证明坍缩或保证重训收益。区间条件于固定 checkpoint，沿用既有工具 bootstrap 设置，不涉及新增训练 seed。
[完整诊断摘要](reports/round13/diagnostic-evidence.json) · [工具状态](reports/round13/toolkit-candidates-status.json)

## 历史更新分支诊断

固定 checkpoint、57408 行验证子集；仅用于机制诊断。Query roll 在当前 batch 内确定性错配反馈 Query，没有新增训练 seed。

| 候选 | 学到的 scale | 原始 AUC | 关闭 writer AUC | 错配 Query AUC |
|---|---:|---:|---:|---:|
| r13_query_between_zero | 0.00003553 | 0.861854504 | 0.861868856 | 0.861853469 |
| r13_self_between_zero | -0.00010710 | 0.862037922 | 0.862051748 | 不适用 |

两个零初始候选的最终 scale 绝对值均小于 0.00011。关闭 writer 后 AUC 均微升约 0.000014；Query 错配影响也很小。未观察到强的正向历史写入依赖。该证据不覆盖 warm 初始化候选，关闭分支也不等同于从头训练无分支模型。

checkpoint 未改变，原始 AUC 与标准导出逐项一致。


保留接近 Cross8/Cross11 或存在成本/排序取舍的候选，同时保存较差对照。固定 seed 的筛选结果不等于已证明稳定泛化收益。
