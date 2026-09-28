# Amazon 评审测量身份修复验收

修复8a26d96后，使用v4已有工具组和候选工具组各自已公开的开发结果、候选源码与基线诊断，分别执行一次evaluation和一次lead。四次真实Codex调用全部成功、零重试、4 BCU，无新增模型训练。两组均返回discard，不产生新的模型效果估计。

独立目录 `rec-harness-runs/amazon-review-scope-v1` 保存prepare/run脚本、原始输入引用、源事件前缀和新角色收据。原控制器前十二个开发诊断/观测事件作为明确标注的历史前缀导入新账本，未复制旧角色完成事件作为新调用。新评审输入将基线诊断和观测置于`incumbent_measurement_context`，保留基线身份；顶层候选身份与其执行证据不变。lead使用本次新evaluation返回。

`transport-acceptance.json`为passed：两组各两次真实传输，原诊断CAS、精确角色输入、基线/候选绑定及新增费用核对一致。`resume-acceptance.json`为passed：强制禁止后端调用后再次执行恢复，零新增调用，两份账本和验收结果不变。源码与安装环境各645项完整回归零跳过，见[v33安装验收](../implementation/install-acceptance-v33.md)。

这证明修复后的评审传递与恢复合同在真实Codex调用中成立。输入来源是既有开发实验，不是一次新的三组在线研究，也不能修复或替代v4冻结历史中的失败审计。v4完整送达验收仍为失败；其62份费用账本另独立核对通过，499.711171 BCU保留。两者累计503.711171 BCU，不包含未定价CPU墙钟时间，也不等于GPU实际利用量。
