# 证据图 v2：真实角色运行验证

2026-10-12。在独立 `outputs/evidence-graph-live-v2-*` 目录中运行真实 Codex 研究员、实现者、独立审稿员、分析员。使用登录环境的默认模型，不调用训练任务。模拟数据不能导入生产研究记忆。

## 验证边界

最终配置使用 `--cpu-python`：模型调用、源码修改、独立审稿、AST 修改范围检查、CPU 合成输入数值合同、计划注册、开发比较、知识补丁验证与提交、SQLite 事件链审计、恢复均实际执行。训练（含 training smoke）、评分及相应工作器回执/底层证据摄入由测试替身提供。上层开发读出与解释写图仍使用生产代码。

CPU 合同执行前向与反向传播以检查梯度，不执行优化器更新。它不是训练，也不能代替训练后检查。模拟分数固定为 baseline 0.50、candidate 0.48、generated control 0.47，只检验未达标时不晋升、不执行确认。没有验证真实性能、真实独立重评分或正向晋升后的确认执行。

最终 `run4` 完成 1 个完整周期：4 个真实角色调用均一次通过，审计确认 4 个角色与 1 个周期，失败轮数为 0。候选和生成对照通过实际 admission CPU 合同，各有 68 个参数梯度检查为有限值；覆盖因果前缀、padding、空历史、读出一致性和 batch 独立性。开发结果未选中候选，未晋升、未执行确认。故障注入后复用已接受角色，终止后再次恢复没有新增请求。

整个排错过程实际调用模型 19 次，不能把成功角色数等同总调用数：`run2` 为 8 次（2 个失败轮，含格式重试），`run3` 为 5 次周期调用（分析员重试 1 次）加 2 次独立读回，`run4` 为 4 次。更早准备阶段曾因 macOS 缺少 `/proc` 在模型启动前失败，不计入真实模型调用。

macOS 验证适配器替代了 Linux `/proc` 进程身份读取；真实进程启动、等待与超时仍执行。本次只在已结束角色调用之间注入中断，不声称覆盖 Linux 孤儿进程恢复。

## 真实调用发现的问题

| 场景 | 实际结果 | 处理 |
|---|---|---|
| 径向归一化对照 | 独立审稿指出 padding 上 `sqrt(0)` 导数奇异，阻止进入计算 | 保留拒绝记录；下一研究轮引用故障并换成反向序号权重对照 |
| 复核上述拒绝 | 对归档代码运行真实 CPU 合同，报 `nonfinite gradient: blocks.0.conv1.weight` | 验证审稿发现有实际数值依据，不把它写成科学机制反证 |
| 正面审稿说明写入 findings | 两次 `approve` 带非空 findings，被控制器拒绝 | 明确 findings 只允许阻塞问题；成功检查放 limitations，未放松验证器 |
| 模拟预算填写 0 | 正数约束拒绝并触发一次格式修复 | 在输出约束说明中明确模拟预算同样必须大于零 |
| Claim 缺少 scope | 分析员首个补丁被拒绝，重试后才提交 | 将节点和 Claim 必填属性显式加入 patch_contract |
| 新知识已提交却读不回 | 真实分析员省略可选 visibility；初次真实读回没有原生 MeasurementGap | v2 解释检索沿用 public_only 的默认 public 约定；保留显式私有、协议、分区和引用闭包检查 |

这里最关键的新增修复来自真实输出：手写测试中的解释节点原本都带 `visibility: public`，没有暴露写入允许省略、读取却要求显式填写的不一致。回归测试现在覆盖省略该字段的合法解释、显式私有节点、确认分区以及缺失引用。

## 知识写入与实际读回

`run3` 分析员经校验提交了一个临时 Claim 和一个 MeasurementGap：

`amazon-research-6185db4bf52dd2b9-simulated-validation-gap-r1`

同一 study 的新研究员在修复前看不到该原生节点；修复后的独立调用既在 `evidence_refs`、也在 `gap.observation_refs` 中引用它。回答明确保留以下限制：

- 模拟候选相对 baseline 为 -0.02、相对 control 为 +0.01，不能据此确认或否定真实机制。
- 需要共享冻结的上游表示、配对用户与候选集合、输出范数、排序边界和实际保留历史长度分组，才能进一步区分时间顺序与其他解释。
- 原开发读出、机制比较和分组观察不是三份独立实验。
- 执行失败不等于机制反证；历史压缩实验的类别负向对照差异仍然保留。

这证明了本次样例中的“真实分析写图 → 检索 → 新研究员引用并解释”。它没有证明模型在任意后续任务中都会正确理解全部证据，或新设计能产生收益。

原始本地证据：

- [首轮失败及重试审计](../../outputs/evidence-graph-live-v2-run2/validation-report.json)
- [知识写入周期](../../outputs/evidence-graph-live-v2-run3/validation-report.json)
- [修复前读回](../../outputs/evidence-graph-live-v2-run3/readback-report.json)
- [修复后真实读回](../../outputs/evidence-graph-live-v2-run3/readback-probe-fixed-report.json)
- [真实 CPU 独立复核](../../outputs/evidence-graph-live-v2-run3/real-cpu-contracts-verified/summary.json)
- [接入真实 CPU 合同的最终完整周期](../../outputs/evidence-graph-live-v2-run4/validation-report.json)

这些原始记录位于忽略的本地输出目录；可移植摘要见 [JSON](evidence-graph-live-v2.json)，含每次请求与响应的哈希、审计结果、真实 CPU 合同结果以及修复前后的引用列表。早期失败输出保留，不覆盖成成功记录。

## 复现

解释器需预装 torch、numpy。运行下列脚本会真实调用角色模型并消耗账户额度；省略 `--cpu-python` 会连数值合同也模拟，只适用于控制器测试。

```sh
python3 scripts/validate_evidence_live.py --output outputs/my-live-check --rounds 1 --cpu-python /absolute/path/to/python
python3 scripts/validate_evidence_live.py --output outputs/my-live-check --probe-only
```

`--seed-campaign` 可载入此前验证的导出记忆；`--probe-name` 可保留独立的修复前后读回目录。禁止将这些模拟 campaign 当成真实训练证据。

相关测试 51 项通过，包含控制器、知识闭环、可选字段、引用闭包、20 轮回放、审稿和输出绑定。20 轮历史前缀回放仍为最近读出 37/37、未达开发门槛的读出 87/87，未来泄漏与悬空引用均为零；详见 [历史回放报告](evidence-graph-prefix-v2.md)。

## 合入 main 的兼容验证

合并时 main 已包含审稿返修、单种子晋升与监控更新。保留 main 的 `checks` / `blocking_findings` 审稿格式和节点属性模板，接入本次历史证据视图；上文 `findings` 与多种子验证配置描述的是原始运行记录，不代表新任务默认配置。模拟角色测试已适配新审稿格式。合并后的 71 项相关测试通过，覆盖证据图、历史回放、模拟闭环、输出绑定、审稿返修、晋升及最终选择；此次合并未重新调用角色模型或执行训练。
