# 公开观测进入独立 shadow 条件

shadow 规格现在支持 `tool_contract: public-observation-measurement-v2`。每个条件显式给出来源研究及观测 ID：

```json
{
  "tool_contract": "public-observation-measurement-v2",
  "conditions": [
    {
      "id": "state-a",
      "model_version": "model-a",
      "seed": 42,
      "observation_source": {
        "campaign": "../model-a-research",
        "observation_ids": ["activation-...", "block-score-..."]
      }
    }
  ]
}
```

片段只展示一个条件；实际套件仍要求至少两个独立的模型源码/seed 条件，同一模型和 seed 的别名、不同观测或重复运行不能充数。观测须对应所选模型及开发条件。`campaign` 相对规格文件解析，`cutoff_seq` 可选；省略时冻结当前事件前缀。

冻结时只读核对 Amazon 控制器锁、绑定 study、协议、事件链与对应图快照。所选观测须存在于该前缀中，且公开摘要 CAS、图节点和主机 summary 事件一致。回滚掉的节点、截止点之后的节点、额外字段、跨模型和非开发条件均拒绝。源路径、摘要引用和事件头保留在主机套件中，工具仅收到 v2 公开请求。

运行前重新构造同一前缀并核对全部输入。源历史的后续追加不改变冻结输入；摘要损坏、锁变化或对套件重新计算 hash 后篡改数值仍被拒绝。原有两个条件各运行两次、范围及重复稳定性门槛、结果复用和费用保留继续生效。这里验证只读适用性，不代替在线决策收益。

reference、shadow、online 三个套件必须声明相同输入合同才能组合为工具验证 pipeline。reference 用顶层 `tool_contract` 声明，online 用冻结 `tool_runtime.contract`；没有声明的旧套件按 v1 处理。项目预检还核对计划与内置验证套件的合同，防止用旧 v1 验证器支持 v2 晋升。声明相同合同不证明测量的科学适用性；各阶段仍需实际的独立输入和参考结果。

本次27项相关测试通过，覆盖新来源链、修改拒绝、截止/回滚、两条件执行及恢复、pipeline合同一致性和已有工具流程。测试使用模拟训练收据和CPU启动器，不算真实影子实验。

真实只读核验位于 `rec-harness-runs/observation-shadow-input-v1`：从既有 activation-role 与 block-score-context 研究提取5条公开观测，重复前缀重建一致，源账本不变。两个来源共享同一模型与seed，不声称两个独立条件；未新增Codex调用、工具执行或训练。在线套件自动产生这些模型观测的能力仍待补齐。
