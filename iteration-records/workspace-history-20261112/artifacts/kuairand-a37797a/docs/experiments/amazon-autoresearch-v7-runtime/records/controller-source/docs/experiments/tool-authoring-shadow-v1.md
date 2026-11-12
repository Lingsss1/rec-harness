# 新生成工具的真实开发影子验证

接续 [tool-authoring-codex-v1](tool-authoring-codex-v1.md)，使用同一冻结候选及原 shadow 阶段计划。验收目录 `/home/liuzhishan/lhr/rec-harness-runs/tool-authoring-shadow-v1`，`preflight.json` 保存准入结果，`acceptance.json` 绑定阶段报告、脚本与账本身份。

执行前重新核验冻结影子套件与原始 Amazon 开发作业收据。两个不同模型条件、seed 42，各运行两次真实 CPU 隔离测量，共四次，均通过预声明范围与重复性要求：

| 条件 | mean_metric | minimum_metric | metric_spread | 两次输出 |
|---|---:|---:|---:|---|
| incumbent-dev42 | 0.5535 | 0.445 | 0.192 | 完全相同 |
| channel-gate-dev42 | 0.5545 | 0.446 | 0.182 | 完全相同 |

报告中的 gain=1 是该阶段通过比例，不是模型增益或决策效用。候选进入 shadow_validated，未晋升。禁止启动验证器后重复验证，返回原接受记录，事件和费用不变。新增 Codex 调用、训练和确认任务均为 0。

控制器累计费用由 4 增至 5（含本次阶段声明费用 1），独立影子账本记 4；两层费用分列，不混称实际 GPU 成本。控制器事件 18，链头 `0465b2c8bb1067955040361715089eaa82c7926d063468b88f4f675481927903`；影子账本事件 12，链头 `b16fc1ddf8f507e00edc0b574d97db2ede4dc6b67f28b7892d0fa20a290ccf0b`。此前离线收据保留为当时检查点。

在线套件准入发现其 worker hash 为 `e9bceb476699095356a5c96e570c3830e830b1f6ca1e98279026d85684ccf85b`，当前为 `97f24cf409dbf92ecea5bfbddc9f68a585a2e0ee334302ac61032448edf3204e`，不兼容。因此未启动在线效用调用，也未修改旧套件；后续在线验收须建立新的冻结条件，不能将这次影子通过当成在线晋升证据。
