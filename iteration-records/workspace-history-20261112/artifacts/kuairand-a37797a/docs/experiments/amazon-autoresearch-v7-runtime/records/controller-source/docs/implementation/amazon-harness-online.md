# Amazon Harness 在线配对验证

`python -m rec_harness.amazon_harness_online` 在相同 Amazon 模型起点、协议、初始化种子和各轴额度下，分别运行旧策略和候选策略。两组使用独立 study、记忆、角色目录及训练账本，复用完整训练、CPU 独立复评分、内部确认和终态 incumbent 选择规则。正式角色直接调用隔离 Codex。

## 使用

替换 [spec.json](../../examples/amazon-harness-online/spec.json) 和 [controller.json](../../examples/amazon-harness-online/controller.json) 中的主机路径、运行环境和模型设置。候选目录含 `policy.json`；[示例策略](../../examples/amazon-harness-online/candidate/policy.json) 只展示格式，不代表已验证的改进。

```bash
python -m rec_harness.amazon_harness_online freeze /path/to/spec.json /path/to/suite
python -m rec_harness.amazon_harness_online run --suite /path/to/suite/suite.json --candidate /path/to/candidate --output /path/to/development/report.json --stage development
python -m rec_harness.amazon_harness_online run --suite /path/to/suite/suite.json --candidate /path/to/candidate --output /path/to/confirmation/report.json --stage confirmation
python -m rec_harness.amazon_harness_online audit --suite /path/to/suite/suite.json --output /path/to/development/report.json
```

外层开发和确认条件在同一 suite 中冻结，两个阶段使用不同输出目录。不能跨条件复用同一数据身份及训练 seed。数据仍在主机；不同 seed 可能共享验证用户，因此这不是最终测试。是否进入确认、是否晋升由外层计划决定。

策略仅允许现有 `prompts`、`retrieval_limit` 和固定角色路由合同，不能修改训练、评分或预算。两组固定使用内置观测，可配置相同的开发激活采集；当前不支持混合诊断工具晋升或继承研究记忆。源 study 只提供冻结模型、worker、协议和数据身份，旧任务、结果及费用不导入。

每个条件每组每轮通常六次角色调用，一轮配对约十二次；外层开发和确认各一个条件约二十四次。实际调用受各组角色额度约束。

## 结果与恢复

主机复核有效策略、初始源码、协议、执行版本、完成事件和实际隔离传输收据，独立复算终态开发 HR@10。两组评分人群与候选集合必须相同。`gain` 为条件间候选减基线的平均值，`regression` 为最大退化；`passed` 表示执行合同有效，数值门槛仍由外层判定。测试注入角色时强制 `online: false`。

费用按角色调用、GPU 秒和可选激活费用分别核对，再按冻结权重折算；未定价费用轴拒绝。CPU 准入/复评分耗时、外部优化器开发和前缀审计费用未在此独立入口计价，不能把结果当成全部研究成本。

已完成组重新审计后复用；开始但未提交完成记录的组停止，不自动重发。内置验证预检和外层阶段恢复已识别新入口。`audit` 只读核对收据与报告，不启动研究。[三阶段 pipeline](amazon-harness-pipeline.md) 已接入 Amazon 前缀、开发、确认及累计费用；目前前缀只支持有历史覆盖的提示词改动，真实新策略效用尚待验证。

## 工程验收

八项新增测试覆盖配置、条件去重、策略权限、独立组与精确策略送达、真实控制器初始化和复用、正式路径传输要求、报告复算、篡改拒绝、未定价费用与未知结果恢复。与 Amazon 工具、激活、预检及阶段恢复相关共 43 项测试全部通过，零跳过，6.621 秒。训练及角色为明确夹具，没有新增真实 Codex 调用或 Amazon 训练；运行中的工具 v2 使用原冻结内核，不受这次改动影响。此实现晚于 v28 安装包。

真实配置另完成两阶段只读冻结和预检，包含相同范围激活采集，使用新的初始化条件；源配置、study 和锁文件哈希不变。证据为 `rec-harness-runs/amazon-harness-online-v1/preflight-acceptance.json`，新增角色调用和训练均为零。公开树边界、diff 格式和命令入口检查也通过。初次扩大测试命令误写了两个测试模块名，产生两项导入错误；改用实际存在的恢复测试模块后得到上述 43 项通过结果。
