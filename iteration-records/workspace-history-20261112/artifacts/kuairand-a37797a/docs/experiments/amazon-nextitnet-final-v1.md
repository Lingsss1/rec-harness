# 第二 GELU 候选的最终测试尝试

确认通过后的比较已封存：`legacy_incumbent` 对 `amazon-next-v3-round-1-model`，四类别、全部确认初始化 17/83 的 16 个 checkpoint。最终评估代码冻结于 `934e7a8`，eval_seed 为 20260927，每类至多 1000 用户。输出目录为 `/home/liuzhishan/lhr/rec-harness-runs/amazon-nextitnet-final-v1/`；selection hash 为 `6c741b259b6eedf9af991a1b25d1320cd496a356858f8ff12050fc6af4239bb3`。该 study 已封存，不再允许返回模型优化。

首次读取测试数据后，执行器按原协议拒绝：`final interaction outside frozen vocabulary`。没有启动最终模型推理，也没有产生最终分数；没有修改 checkpoint 或静默删除用户。

| 类别 | 冻结 checkpoint 的 itemnum | 预处理元数据 itemnum | 测试交互中超出词表的条数 |
| --- | ---: | ---: | ---: |
| Movies_and_TV | 17457 | 17490 | 48 |
| Industrial_and_Scientific | 25660 | 25764 | 195 |
| Electronics | 31345 | 31533 | 294 |
| CDs_and_Vinyl | 30920 | 30951 | 55 |

训练/验证交互全部在冻结词表内。训练 worker 使用训练/验证中出现的最大物品 ID 构造 embedding；预处理 `dataset_stats.json` 则记录完整物品数，与测试最大物品 ID 一致。因此当前 checkpoint 无法直接评分这些未知物品。该检查只输出本机聚合与文件哈希，不将测试信息发送至研究角色。

`failed-final-acceptance.json` 核验了原封存、终止进程身份、原始失败、零推理任务及词表/元数据证据。`vocabulary-audit.json` 保存全部切分的聚合核验。原测试协议仍为未知物品直接拒绝，未更改为过滤用户、扩充 embedding 或将未知正例计为未命中。

已请求用户选择是否新增明确标注的补充评估，将未知正例计为未命中并报告覆盖率；在该选择明确前，不执行依赖新口径的测试评分。现有确认增益保持其原范围，不冒充最终测试成绩。
