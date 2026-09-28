# 在线比较的组间恢复

外层Evolution现在允许在线比较在已完成研究组之间恢复。范围是直接 `online_validation` 的development/confirmation、`tool_online` 的utility/confirmation，以及 `tool_pipeline` 的utility。

恢复前只读核对已有组的冻结配置、事件链、费用投影、controller_stopped事件和state.json。每个已初始化组必须以budget或max_rounds结束、有有效incumbent且没有未结算费用。满足条件后，在原目录恢复验证器：原组由Research终态缓存返回，只执行剩余组；新增验证器进程仍单独计费。

组内未完成、只有可修改的终态文件、配置变更、部分初始化或未知目录均不授权重启。组内中断不能被自动转换成失败回合后继续比较，那会改变既定实验轨迹。Harness pipeline涉及控制器费用前缀，尚未接入本项组间恢复；任意自定义验证器也不据此获得重启资格。

24项定向测试通过，覆盖两组策略比较、三组工具比较、pipeline入口分派、已有诊断不重复、未知组内结果不新增费用、状态篡改拒绝及项目调度。角色、模型评估及诊断执行为工程夹具，报告保持online=false；没有新科学效用结论。

`rec-harness-runs/online-group-recovery-v1` 对三份真实历史在线实验的7个研究组完成只读检查，前后事件链完全一致：tool-authoring-online-v1、joint-project-acceptance-v1、harness-evolver-acceptance-v2。没有启动验证器、Codex、CPU作业或训练；它证明真实终态记录可通过组间恢复预检，不证明真实故障后续接已运行。

本项晚于v23安装包及527项完整回归；新增路径由上述定向测试验证，安装交付需随后更新。
