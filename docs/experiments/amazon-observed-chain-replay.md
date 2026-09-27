# 两轮真实开发研究的顺序回放

新增 `amazon-replay-chain OUTPUT CAMPAIGN... --gpu-second-weight 1 --role-call-weight 1`。每个输入须是已审计的单轮开发 world，后继须有唯一已提交的 `next_campaign_lineage`，其 CAS 内容、前序 campaign/study 路径、事件头及累计成本均匹配。前一轮读出必须在后一轮设计上下文中，incumbent 源码和可见基线必须连续；缺少这些证据不拼接。

真实验收连接 `amazon-input-offset-review-v4/research` 和 `amazon-autoresearch-next-v1/research`。world 位于 `/home/liuzhishan/lhr/rec-harness-runs/amazon-observed-chain-replay-v1/world.json`，hash 为 `b7470ef29a46997b322bea54d8e4f696a78816793d3e0a944bc43688904e94c8`。两轮均未达到开发门槛，选中分数维持基线 0.5535。

| 成本项 | 加权历史成本 |
|---|---:|
| 根状态已有 GPU 成本 | 6477.860062330961 |
| 第一轮候选及累计角色 | 1205.4143040701747 |
| 第二轮新增对照、候选及角色 | 2179.1062788441777 |
| 合计 | 9862.380645245314 |

权重都是 1，单位为 GPU 槽位秒与角色提交次数的显式加权和，不是金额、总墙钟或新增 GPU 使用。第二轮以累计成本差额收费，继承的 11 次角色与前序训练不重复计算；CPU 与 token 成本仍不包含。

`acceptance.json` 记录两次独立隔离策略运行结果一致、world 不挂载、首轮没有未来结果、第一轮揭示后仍没有第二轮结果、第二轮预算不足则拒绝揭示、重复转换一致，以及源账本不变。新增 Codex、训练和模型推理均为零。初次验收在策略启动前因 Python 整数权重与 CLI 冻结浮点权重的 JSON 哈希不同退出，按原冻结权重 1.0 重试通过；首次命令另有后继路径拼写错误，也未进入策略执行。

该适配器支持已有顺序的继续/停止，要求所有前序依赖先揭示。它不支持改变提示、代码、工具、轮内决策或 GPU 并行调度；也未完成世界开发/确认划分、反事实新策略效用或最终测试。其他缺少显式血缘的历史 campaign 不能仅凭时间顺序获得回放资格。
