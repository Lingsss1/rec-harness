# 项目验证器启动预检

新的联合项目在任何模型研究调用前检查已启用的工具/Harness 验证计划。此前项目只检查工具执行参数，旧冻结套件的 worker 不兼容可能等到模型研究或工具生成完成后才暴露。

`validation_preflight.preflight` 首先校验并计算验证器文件身份，再对已知内置模块检查冻结套件。工具 pipeline 递归核对 reference、shadow 和 online 的套件身份与工具文件；reference 校验参考例合同，shadow 复核开发来源，online 校验冻结 episode、既有工具产物和当前 worker。Harness pipeline 复用其前缀、独立在线输入和条件隔离检查。未知自定义验证器明确只检查文件依赖，不执行外部代码。

独立命令 `python3 -m rec_harness.validation_preflight CONFIG` 接受含 plan 的演化配置或 plan 本身，输出各阶段检查范围。新项目初始化自动调用；禁用更新槽或没有后继模型回合的槽不触发。预检不改变已有冻结项目的恢复逻辑，也不自动改写旧验证套件。

检查不创建候选、预留预算、调用 Codex 或执行训练。它只保证检查时点的一致性，不能替代验证阶段的再次核验，更不证明运行环境权限、服务可用性、工具对任意自主缺口适用或科学收益。

真实配置只读检查：`tool-authoring-codex-v1` 在调用前被拒绝，原因是 diagnostic worker 与冻结在线套件不匹配；`harness-evolver-acceptance-v2` 的已知套件检查通过。配置文件 hash 未变化，收据位于 `validation-preflight-v1/acceptance.json`。没有新 Codex 调用或真实推荐训练。

针对性回归 22 项通过，覆盖套件无执行/无文件改动、重新计算 suite hash 也不能掩盖旧 worker、缺失产物、计划身份不一致、项目创建前失败、禁用槽与联合调度恢复。

完整源码回归：465 项测试全部通过、零跳过，88.702 秒，启用 Torch 和 Linux OS 检查。日志和摘要保留在上述运行目录。本项及近期转交修复尚未纳入 v19 wheel。
