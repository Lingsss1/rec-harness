# 结构化插入合同端到端验收 v3

状态：真实端到端及终态精确恢复验收通过。运行目录 `/home/liuzhishan/lhr/rec-harness-runs/generated-control-e2e-v3`，冻结源码 `896a8e5`。使用与 v2 相同的 32 用户合成数据和 2 epoch × 2 batch 训练协议，继承 v2 全部 326.55892564356327 GPU 秒历史成本。

本轮新增验收新 campaign 默认启用的方法范围合同：真实 hypothesis 在生成代码之前声明允许改动的方法，experiment 同一次调用生成候选和新对照，主机在角色输出检查与原子准入中分别检查每份源码没有越界改动。评估和负责人必须收到两份模型各自的方法范围核验、操作图摘要和整段序列语义检查。预期仍为六角色调用，无新增评审角色。

终态审计保留 v2 的三源码身份、开发对比、独立评分、归档/CAS 与零新增执行恢复检查，并新增上述合同和角色实际可见证据字段断言。结果以 `controller-result.json`、`terminal-acceptance.json`、`watch-result.json` 为准；启动不构成通过，合成结果不证明真实推荐机制或性能收益。未改写此前 campaign 或最终测试配置。


## 终态证据

`controller-result.json` 为 completed，`terminal-acceptance.json` 和 `watch-result.json` 均通过。六次 Codex 角色调用、无重试、一次完整周期、56 份归档引用核验通过。基线、候选及生成对照均完成 GPU 训练和独立 CPU 评分；三份评分各 32 用户，排名差异及正样本并列均为零。

hypothesis 冻结的合同只允许 `NextItNetModel.encode_sequence`。候选及对照的 AST 实际变动均只包含该方法，合同 hash 同为 `d88b8decb9d2beebf72aa1e9e38624d3068098b7386bf37c652271054bbc1c11`。终态审计读取 evaluation/lead 的真实角色上下文，确认两份模型各自的范围核验、操作图和序列语义摘要均以 verified 交付；这些字段不是事后追加到角色记录。

三模型开发 HR@10 都为 0.09375，候选不进入确认、基线保持不变。这里验证完整 discard 路径，不构成机制或真实推荐收益证据。总 GPU 账本 592.9974888265133 秒，包含前驱全部 326.55892564356327 秒；新增 266.43856318295003 秒。

周期身份 `89921159da99cdb5e4d1ce07424f3d4c5b6804ecbbef8540a6d2a87456514e06`，开发结果 hash `9c905079a1fe79c507c3a15310eb82ffeeb00c9fb003c1433b942655f3a842d7`。终态在禁止新增角色、训练与推理函数执行的情况下重复 controller.run 和评分读取，输出、事件和费用不变。v11 安装代码另完成真实准入投影复核，见 `package-install-acceptance-v11/live-insertion-evidence.json`。
