# 新 worker 条件下的独立工具在线比较

启动检查点：此文件记录冻结设计和启动状态，尚无完整效用结果。运行目录 `/home/liuzhishan/lhr/rec-harness-runs/tool-authoring-online-v1`，源码 `17c6075`；实际进度以只读 `progress.py`、各组事件账本和最终 `acceptance.json` 为准。

此前工具完成真实生成、三个离线阶段及两个真实开发条件的影子验证。旧在线套件与当前 worker 不兼容，本次创建独立新套件，未改写原 Evolution 计划或旧验证结果。

新套件 hash 为 `f4400c52e4668a4ff1b7e59f87de0d7e7a066c774fdb72d691b9776205962449`，worker hash 为 `97f24cf409dbf92ecea5bfbddc9f68a585a2e0ee334302ac61032448edf3204e`。候选源码从原 CAS 核验后复制，实现 hash 保持 `e72ec5ed45d39d5fb44a5af9a6e93fa496cd60c615d5dba0664af1a065b17389`，没有新增生成调用。

三组为 no_tool、existing_tool、candidate_tool，共用起始模型、评分协议、基础 LLM、图起点和每组预算 12，使用独立 overlay。开发 seeds 为 1581/1582，确认 seeds 1681/1682 仅冻结而不在本次运行。每组最多一轮，预计合计 18 次顺序角色调用；实际失败/重试及费用从账本统计，不以预计数冒充实际数。

沿用来源计划的效用门槛：同时优于两个对照，最小平均收益 0.005，最大退步 0.01，阶段最大声明费用 70。旧生成/离线/影子阶段的账本身份和费用均保存在 `source.json`，控制器阶段费用与子验证费用分别呈现，不重复求和成“实际训练费用”。

启动后已观察无工具组 evidence 完成、hypothesis 提交，另两组尚未启动；这不是实验完成证据。运行过程不启动真实推荐训练、确认或最终测试，不直接晋升旧 campaign 候选；零收益和失败如实留存，不重跑直到出现正结果。
