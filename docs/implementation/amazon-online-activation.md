# Amazon在线对照的激活观测与费用

Amazon工具在线套件现在允许控制器配置activation_diagnostic。每个比较组从自己的incumbent开发checkpoint采集原始激活摘要，原始摘要同样供三组研究角色读取；已有工具和候选工具可用公开观测合同进一步计算测量。继承已有的开发集限定、私有权重/历史留在主机端、摘要入图和角色送达路径。此接入不会把确认或最终测试数据提供给工具。

配置见[观测版套件](../../examples/amazon-tool-online/spec-activation.json)及[控制器](../../examples/amazon-tool-online/controller-activation.json)。示例settings使用历史NextItNet公开源码的默认值，必须按实际训练设置核对；默认值本身不能证明与某份checkpoint兼容。其余模型、数据和运行环境路径同基础示例需要替换。datasets/seeds省略时，采集该条件所有数据集和开发seeds；显式种子必须属于该条件开发分区。

启用采集必须在cost_weights中声明正数activation_units权重。冻结阶段验证参数字段、MAXLEN及开发条件，并要求采集额度能覆盖最坏情况下每轮更换incumbent的次数，避免基线训练之后才发现已知额度不足。实际费用仍按实际执行记录，缓存复用不重复计费。

三组研究报告新增独立activation_units费用轴。主机逐项核对probe的冻结参数、研究/模型/数据集/seed身份、事件账本、结果投影、CAS和成功结果绑定的开发checkpoint；成功测量必须与档案一致。费用阶段将这些子账本纳入去重集合，并按冻结权重计入；已放弃验证尝试的probe账本也识别为同一轴。原有无采集套件继续接受原三轴费用配置。

该费用轴是已记录probe的BCU，不是完整CPU硬件成本；训练准入与独立复算的wall time仍未计价。这里只接入activation_diagnostic，block-score和交互诊断的在线采集配置仍未接通。真实Codex/GPU端到端效用和自主发现测量缺口仍需实际验收，不能由这些工程测试推导。

新增7项测试覆盖开发分区、显式权重、预算覆盖、只读账本审计、checkpoint/配置/结果篡改拒绝、终态读出和三组费用汇总。带PyTorch解释器下49项相关测试通过，零跳过，4.634秒，包含既有probe、checkpoint、历史采样及角色观测送达测试。新在线场景的训练和角色仍为夹具，本次没有新增Codex或训练。

真实Amazon源码/数据/配置另外完成激活启用的两阶段冻结及只读预检，历史研究和study账本未变；未运行采集或在线效用实验。证据位于rec-harness-runs/amazon-online-activation-v1，晚于v27安装包。
