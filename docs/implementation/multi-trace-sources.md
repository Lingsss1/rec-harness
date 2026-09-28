# 多来源 Harness 轨迹入口

演化配置可用 `trace_sources` 代替单个 `trace_source`：

```json
{
  "trace_sources": [
    {"campaign": "../source-a", "cutoff_seq": 54},
    {"campaign": "../source-b", "roles": ["evidence", "hypothesis", "evaluation", "lead"]}
  ]
}
```

相对路径以所属配置文件为准；两种字段不能同时声明。每个来源仍通过已有角色输入、CAS、配置锁和事件前缀检查。未指定cutoff时在演化初始化时冻结当前前缀，不追随后续追加。重复路径或克隆账本中的重复角色证据拒绝，以免把同一次调用重复计入诊断。

多来源返回 `research-decision-prefix-set-v1`：每条轨迹的顶层trace_id是来源前缀与本地ID共同生成的唯一引用；`source_trace_id` 保留本地ID，`source_trace_hash` 绑定原始完整轨迹，`source_prefix_hash` 指向来源task/protocol/cutoff/event-head。角色可见输入、输出和模型引用不重写。不同来源按前缀稳定排列，每个来源内部保留提交事件顺序，不用墙钟时间重排。无损共享压缩仍可恢复完整集合。

Harness候选须将所有实际来源task_id放入plan.development_ids，不能只排除其中一个任务。多个来源恰好使用同一个task_id时，仍保留各自独立前缀，不把task数量当作独立实验数量。单来源入口保持原schema、ID和输出语义。

项目的Harness模板也可声明trace_sources：调度器在未包含当前模型campaign时追加当前回合，并将全部来源任务加入排除集合。模板不声明时沿用原单来源路线。若外部优化器转交MeasurementGap，每条证据仍带task/protocol/prefix；不生成任意一个任务代表所有来源的诊断摘要。工具分支保留来源集合，并继续要求排除全部来源任务。

51项相关测试通过，覆盖重名ID、原轨迹逐项恢复、相反输入顺序的一致输出、截止后的追加、重复/克隆拒绝、排除集合、项目组合及多来源测量转交。阶段验证和工具执行的组合测试使用夹具，不算在线效用。

真实只读证据在 `rec-harness-runs/multi-trace-source-acceptance-v1`：取已完成异任务迁移实验的源A与两个开发组，不取确认组；3个前缀、2个task_id、18条轨迹，原本仅6种本地角色ID，合并后18个唯一引用。各原轨迹hash与此前终态审计一致，压缩恢复及反序收集完全一致。没有新增Codex或训练；尚未在一次真实多来源优化器调用中检验补丁收益。
