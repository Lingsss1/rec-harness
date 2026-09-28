# 在线比较的组间恢复

外层Evolution现在允许在线比较在已完成研究组之间恢复。范围是直接 `online_validation` 的development/confirmation、`tool_online` 的utility/confirmation、`tool_pipeline` 的utility，以及 `harness_pipeline` 的development/confirmation。

恢复前只读核对已有组的冻结配置、事件链、费用投影、controller_stopped事件和state.json。每个已初始化组必须以budget或max_rounds结束、有有效incumbent且没有未结算费用。满足条件后，在原目录恢复验证器：原组由Research终态缓存返回，只执行剩余组；新增验证器进程仍单独计费。

组内未完成、只有可修改的终态文件、配置变更、部分初始化或未知目录均不授权重启。组内中断不能被自动转换成失败回合后继续比较，那会改变既定实验轨迹。Harness pipeline的prefix_audit及任意自定义验证器不据此获得重启资格。

每次恢复调用在费用预留之后冻结独立的 `controller-cost-resume-N.json`，将CAS引用和实际argv写入启动事件。原始阶段费用快照保留不变；后续启动检查各次调用的费用前缀和命令绑定。Harness pipeline通过本次调用的快照计入所有已发生的控制器费用，不再读取首次启动时的旧总额。

新增真实控制器工程测试在baseline组完成后连续中断两次，再继续candidate组：角色总数仍为12，子研究成本16，三次验证器调用费用3，pipeline报告总额19；三个独立费用快照分别为1、2、3，当前调用预留均为1。角色及评估仍为夹具，online=false。费用、pipeline、Evolution及在线恢复相关37项测试通过。

24项定向测试通过，覆盖两组策略比较、三组工具比较、pipeline入口分派、已有诊断不重复、未知组内结果不新增费用、状态篡改拒绝及项目调度。角色、模型评估及诊断执行为工程夹具，报告保持online=false；没有新科学效用结论。

`rec-harness-runs/online-group-recovery-v1` 对三份真实历史在线实验的7个研究组完成只读检查，前后事件链完全一致：tool-authoring-online-v1、joint-project-acceptance-v1、harness-evolver-acceptance-v2。没有启动验证器、Codex、CPU作业或训练；它证明真实终态记录可通过组间恢复预检，不证明真实故障后续接已运行。

Harness pipeline费用恢复完成后，534项完整源码回归全部通过、零跳过，200.911秒，包含Torch与Linux隔离。`rec-harness-runs/harness-pipeline-recovery-v1`保存全量及37项定向测试日志、源码清单和验收范围。本项仍晚于v23安装包，安装交付需随后更新。
