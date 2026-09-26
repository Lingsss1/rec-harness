# Amazon NextItNet：有效 token 位置编号

开发集结果未达到确认门槛：四类别平均 HR@10 从 0.5535 降至 0.5435，差值 −0.0100。四次独立 CPU 复评分的主机分数重算、来源绑定与开发读出核对已通过，4,000 个排名无差异。控制器尚在归档和研究角色收尾，本文不构成整轮终态验收或已发布的研究记忆。

## 干预与比较条件

候选 `amazon-frozen-followup-v1-round-1-model` 由 Codex 提出并实现，方法为 `causal_valid_token_ordinals_v1`，来源标记为 `agent_hypothesis`。它将 `encode_sequence` 中的绝对 tensor 位置索引改为 `(seq != 0).long().cumsum(dim=1) * (seq != 0).long()`：有效 token 按保留历史顺序编号，填充位置编号为零。该变化不等价于整个网络对左侧填充保持不变，也不能凭性能差值确认“历史长度意识”这一解释。

对照为 `legacy_incumbent`。沿用冻结训练协议 `54fbd25c87110ea1b813f57c205b1725ceb0fb77b1e61ed9a331e3fe90dc6d31`、执行版本 `execution-f40bbccd81ea5824320c803d`、开发 seed 42，每类别 1,000 个固定验证用户。候选源码 SHA256 为 `f2f526bbe59d224d0b9016780dec170a563fd0b3825fd67745c57b786b29b3aa`，对照为 `d32d162182c2c156dd24573819361d513e49960f44429ce6f005afa21ba4d553`。源码准入、CPU 数值检查及训练压力检查收据成功。

| 类别 | 对照 HR@10 | 候选 HR@10 | 差值 |
|---|---:|---:|---:|
| Movies_and_TV | 0.601 | 0.604 | +0.003 |
| Industrial_and_Scientific | 0.531 | 0.532 | +0.001 |
| Electronics | 0.445 | 0.435 | −0.010 |
| CDs_and_Vinyl | 0.637 | 0.603 | −0.034 |
| 非加权均值 | 0.5535 | 0.5435 | −0.0100 |

主机开发选择要求平均增量严格大于 0.005。本轮未满足，`eligible_for_confirmation=false`。两类轻微正差不能抵消其余类别的退步；这是单一开发初始化上的候选筛选结果，不是跨 seed 的机制确认或真实最终测试结论。保留全部类别的次要指标和反例，不据此修改冻结评价规则。

## 可追溯证据

根目录为 `/home/liuzhishan/lhr/rec-harness-runs/amazon-frozen-autoresearch-followup-v1`：

- `research/rounds/round-1/hypothesis/response.json`：机制、替代解释、适用条件及证伪标准。
- `research/rounds/round-1/admission.json`：源码及 CPU 准入收据。
- `research/rounds/round-1/development-readout.json`：全部开发指标、源文件及训练收据绑定；读出 hash 为 `5f29a4679ebc7a8c870c1a2167009275245588a9cd804ec66029a97ddfd4e6c7`。
- `research/rounds/round-1/independent-rescores/*/acceptance.json`：四类别各 1,000 用户的独立排名核对。
- `development-preliminary-audit.py`、`development-preliminary-audit.json`：使用原冻结评分器重新计算分数并核验上述绑定，仅输出聚合结果，无新训练、推理或 Codex 调用。

费用终态对账、全部角色、周期完整性和精确恢复以随后产生的 `research-acceptance.json` 为准。在该文件通过前，不将本报告当作完整自主研究验收，也不将候选记忆自动发布到下一轮。
