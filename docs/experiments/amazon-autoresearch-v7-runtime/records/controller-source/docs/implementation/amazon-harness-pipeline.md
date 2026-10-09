# Amazon Harness 三阶段接入

`harness_pipeline` 现可组合 Amazon 前缀审计、独立开发和冻结确认。它按 online suite 的类型选择小型任务或 [Amazon 配对执行器](amazon-harness-online.md)，旧小型任务配置保持兼容。

## 冻结与运行

先按配对执行器说明冻结 online suite。然后在 `harness_prefix freeze` 的规格中提供已封存 Amazon campaign、角色、事件截止点、明确的 Codex 模型、预算和每次调用成本；`identity` 使用 online suite 的完整 identity，`baseline_policy` 从源 campaign 读取。源模型源码及角色上下文来自提交时的 CAS 快照，原始回答、事后反馈和确认结果不进入重演输入。

```bash
python -m rec_harness.harness_prefix freeze /path/to/prefix-spec.json /path/to/frozen-prefix
python -m rec_harness.harness_pipeline freeze /path/to/pipeline-spec.json /path/to/pipeline
```

`pipeline-spec.json` 的格式：

```json
{
  "prefix": "/path/to/frozen-prefix/suite.json",
  "online": "/path/to/frozen-online/suite.json"
}
```

三个阶段的外层验证命令均使用下面的形式，将 stage 分别设为 `prefix_audit`、`development`、`confirmation`。目录及费用前缀由现有 Evolution 控制器提供，配置沿用 `harness-validation-pipeline-v1`。

```text
{python} -m rec_harness.harness_pipeline run --suite /path/to/pipeline/suite.json --stage development --candidate {candidate} --output {output} --cost-context {cost_context}
```

在运行前冻结各阶段门槛。前一阶段必须存在匹配候选源码、pipeline 和报告哈希的接受记录；不能跳过前缀或开发门槛直接进入确认。

## 前缀合同与独立性

支持 evidence、hypothesis、evaluation、lead 的提示词审计。校验 incumbent、协议、提案 ID、配对 seed、对照、方法及缺口引用、预算、插入合同，以及原始开发报告引用和确认准入门槛。成功只表示响应满足可检查合同，不证明机制归因或反事实模型收益。未准入的旧角色输出排除；缺少调用前源码快照时拒绝。

Amazon 历史输入只保留当时实际提供的记忆，不能从当前图反推未记录的历史检索。因此改变 `retrieval_limit` 会在调用前拒绝；未覆盖角色和改变路由也按既有前缀规则拒绝。当前三阶段自动验收支持有历史覆盖的提示词改动，不将独立配对入口能执行的所有策略都说成可通过前缀审计。

前缀与在线阶段必须使用相同 identity、旧策略和明确的基础模型。源 study 的全部开发/确认初始化条件进入排除集，禁止在线阶段复用相同数据哈希、数据集与训练 seed。不同初始化可能共享验证用户，不代表独立用户或最终测试；跨模型族效用也不能由此推导。

## 费用与恢复

每个已执行 Amazon 在线阶段都从原始收据重新审计，要求真实隔离 Codex 角色。费用从角色、GPU 秒、可选激活账本复算，按 online suite 的冻结权重转换；前缀审计和外层控制器费用已是 BCU，按原数值累计。中断尝试保留，重复账本、待结算费用、未知费用轴或配置变化拒绝。CPU 准入及复评分耗时仍没有定价轴，不能宣称覆盖全部硬件成本。

已完成组复核后复用，任何初始化但没有完成记录的组仍禁止自动重发。前缀恢复沿用提交绑定与结果归档规则，pipeline 的恢复已按 Amazon 目录识别。

## 验收

七项新增测试覆盖原始可见输入、原响应排除、恢复不重复调用、真实 Amazon 响应合同、来源排除、GPU 权重、中断费用保留、三阶段路由及正式传输审计要求。相关 67 项测试通过后，真实历史只读验收发现确认准入标志在 `development.summary` 内；修正校验并补充准入正例后重新冻结 v2。v1 失败记录保留，不改写旧套件。

`rec-harness-runs/amazon-harness-pipeline-v2/preflight-acceptance.json` 记录四个真实历史角色原响应的检查和三阶段配置预检，源事件链、controller 与 study 配置均未改变。没有新增真实 Codex 调用、训练、策略晋升或最终测试。

修正后完整源码回归 **637 项全部通过、零跳过，171.730 秒**，包含 Torch 与 Linux 隔离。`full-regression-acceptance.json` 绑定日志、全部测试文件及与两份冻结 suite 一致的实现哈希。公开树边界检查通过。本实现晚于 v28 安装包；源码回归不冒充新安装包验收。
