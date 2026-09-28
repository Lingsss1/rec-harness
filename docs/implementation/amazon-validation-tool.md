# Amazon 验证专用候选工具入口

AmazonAutoresearch.init 增加主机内部参数 `_validation_tool`，采用与小型任务相同的 freeze_validation_tool 合同。它冻结候选源码、worker、输入合同和运行配置，设置控制器及工具的 validation_only 标记，复用原有隔离诊断执行、开发聚合输入、角色送达及独立诊断费用账本。

该入口不需要也不创建正式工具晋升。正常配置仍通过 diagnostic_registry 加载已晋升工具；候选注入不能同时覆盖 registry，也不能进入 review_only 控制器。validation_only 是控制器保留字段，不能由普通配置自行声明。注入前必须明确 diagnostic_budget，候选必须提供 diagnostic.py 和受支持的输入合同。冻结后原作者目录变化不影响已归档工具。

这是 Amazon 三组在线工具验证所需的运行接口，不是三组验证器本身。后续新增[Amazon独立三组入口](amazon-tool-online.md)，实现study创建/冻结、组绑定、训练收据评分和分轴费用；真实执行与pipeline接入仍待完成。既有tool_online保持合成任务合同。不要把此接口当作正式发布入口，或将同一可变 study 共享给比较组。

4项新增测试覆盖冻结源码执行、开发聚合进入工具和 evidence、缓存恢复不重复测量/角色调用、诊断与角色成本分离、study账本不变、互斥入口及缺失合同拒绝。包含已有 Amazon 项目、观测、工具送达和角色恢复的34项相关测试通过，5.585秒。测试调用实际 diagnostic_worker.main，但CPU启动器和角色为夹具，训练读出被替换；本次没有OS隔离、Codex或GPU训练的新增验收。证据位于rec-harness-runs/amazon-validation-tool-v1，变更晚于v27安装包。
