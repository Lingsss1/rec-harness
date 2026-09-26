# Amazon NextItNet：有效 token 位置编号

开发集结果未达到确认门槛：四类别平均 HR@10 从 0.5535 降至 0.5435，差值 −0.0100。四次独立 CPU 复评分的主机分数重算、来源绑定与开发读出核对已通过，4,000 个排名无差异。控制器已正常结束，7 次 Codex 调用、无失败回合；候选淘汰，incumbent 不变，未执行确认或最终测试。独立终态审计尚在进行，本文不构成该审计通过或已发布研究记忆的证明。

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

## 探索性历史长度诊断

开发阶段专用诊断已完成，并核验每类别分箱人数合计为 1,000、净命中差与独立复评分的整体增量一致。以下展示人数最多的 6–20 次训练交互分箱；全部五个分箱，包括空分箱和小样本分箱，均保留在原报告中。

| 类别 | 分箱用户数 | 新增命中 | 丢失命中 | 分箱 HR@10 差值 |
|---|---:|---:|---:|---:|
| Movies_and_TV | 675 | 72 | 64 | +0.01185 |
| Industrial_and_Scientific | 671 | 54 | 55 | −0.00149 |
| Electronics | 679 | 52 | 56 | −0.00589 |
| CDs_and_Vinyl | 677 | 57 | 89 | −0.04727 |

CDs_and_Vinyl 的整体净损失为 34 次命中，其中该分箱净损失 32 次，1–5 次交互分箱净损失 6 次，21–50 次分箱净增加 4 次。因而本轮退步集中在占多数的较短历史用户，但这种事后观察不能把原因唯一归于位置编码或历史长度。分箱依据完整训练历史，不是截断后的输入长度；长历史分箱样本很少，不作稳健收益判断。

证据为 `development-history-diagnostic/report.json`（hash `5f5238dc7c8f805927ddee5689ed4532bee81e1b6fcd7ded9ee976ac45a1c396`）和 `history-diagnostic-acceptance.json`。诊断没有读取确认结果、最终测试，也没有新增模型训练或 Codex 调用。

## Codex 实际调用

事件账本记录 7 次外层调用，每个角色一次，无重试。以下耗时包含隔离启动等角色开销，不包含训练/研究图归档，也不是底层模型 API 请求次数。

| 角色 | 次数 | 耗时（秒） |
|---|---:|---:|
| evidence | 1 | 92.86 |
| measurement | 1 | 39.92 |
| hypothesis | 1 | 88.52 |
| experiment | 1 | 58.54 |
| evaluation | 1 | 45.87 |
| lead | 1 | 47.26 |
| knowledge | 1 | 44.92 |
| 合计 | 7 | 417.89 |

可复核记录为运行根目录中的 `codex-call-summary.json`。

## 可追溯证据

根目录为 `/home/liuzhishan/lhr/rec-harness-runs/amazon-frozen-autoresearch-followup-v1`：

- `research/rounds/round-1/hypothesis/response.json`：机制、替代解释、适用条件及证伪标准。
- `research/rounds/round-1/admission.json`：源码及 CPU 准入收据。
- `research/rounds/round-1/development-readout.json`：全部开发指标、源文件及训练收据绑定；读出 hash 为 `5f29a4679ebc7a8c870c1a2167009275245588a9cd804ec66029a97ddfd4e6c7`。
- `research/rounds/round-1/independent-rescores/*/acceptance.json`：四类别各 1,000 用户的独立排名核对。
- `development-preliminary-audit.py`、`development-preliminary-audit.json`：使用原冻结评分器重新计算分数并核验上述绑定，仅输出聚合结果，无新训练、推理或 Codex 调用。

费用终态对账、全部角色、周期完整性和精确恢复以随后产生的 `research-acceptance.json` 为准。在该文件通过前，不将本报告当作完整自主研究验收，也不将候选记忆自动发布到下一轮。
