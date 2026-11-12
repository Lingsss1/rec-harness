# 第 16 轮 小幅正负历史注入

[返回总览](../../README.md) · [机器可读汇总](../../rounds.json)

状态：**未晋升**。当轮基线：`amazon-next-v9-round-15-model`。

## 假设与实验设计

较小正向注入可能保留历史收益又降低干扰。

| 版本 | 改动 |
| --- | --- |
| 候选 | 归一化前注入 +0.05 倍前缀均值。 |
| 机制对照 | 注入 -0.05 倍。 |

基线和机制对照用途不同：前者判断是否值得保留，后者帮助区分解释。所有已训练版本使用 seed 42；并不承诺新增结构与基线逐参数初始化完全一致。

## 实验结果

| 品类 | 基线 HR@10 | 候选 HR@10 | 对照 HR@10 | 候选减基线 | 候选减对照 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Movies_and_TV | 0.66500 | 0.65300 | 0.65000 | -0.01200 | +0.00300 |
| Industrial_and_Scientific | 0.57200 | 0.58000 | 0.57000 | +0.00800 | +0.01000 |
| Electronics | 0.47700 | 0.45900 | 0.47600 | -0.01800 | -0.01700 |
| CDs_and_Vinyl | 0.68500 | 0.65400 | 0.68000 | -0.03100 | -0.02600 |
| **四品类平均** | **0.59975** | **0.58650** | **0.59400** | **-0.01325** | **-0.00750** |

Host 晋升：`无`；选择策略 `baseline-single-seed-v1`；confirmation_executed = `False`。

## 中文归因与审阅关注点

两者均不如零注入基线；不能支持恢复该分支，符号还可能影响 LayerNorm 和优化。

分析员结论标签：`unknown`；建议：`discard`。这里的 keep 是保留开发集记录供 host 选择，不等于机制确认。

### 分析员原始理由

Audited seed-42 development mean HR@10 was 0.5865 for the +0.05 candidate, 0.59975 for the zero-injection incumbent, and 0.594 for the -0.05 control. The candidate trailed both comparators and was ineligible for confirmation. It improved over both on Industrial_and_Scientific, but lost to both on Movies_and_TV, Electronics, and CDs_and_Vinyl. Source admission, completed frozen-protocol training, and synthetic checks support the recorded implementation and execution; they do not establish a causal mechanism or held-out benefit.

### 尚未排除的解释

- A sign change may affect optimization or LayerNorm statistics without adding useful historical information; the existing post-normalization context blend may account for any useful history signal.
- Only seed 42 was evaluated. Initialization, checkpoint selection, and category heterogeneity remain competing explanations for the mixed category effects.
- The supplied checks do not establish the proposal's exact fixed-state CPU prefix calculation, equal-and-opposite injection at that calculation, or forward and score_candidates identities after an update. Synthetic causality and sequence probes have finite coverage.
- No candidate history-length readout, independent replication, or confirmation result is supplied. A discriminating follow-up would verify the fixed-state calculation and scoring path, then compare zero, positive, negative, and history-shuffled injections across independent seeds under the same frozen evaluation protocol, with the existing post-normalization blend separately ablated.

## 审核和返回修改


### reviewer

原始判定：`approve`。[完整记录](../../records/research/rounds/round-16/reviewer/response.json)

阻塞项：



已检查：

- For implementation_hash 433cb75e1869c4874cb980d0ab9d8af0100dd296e72d5ffc8679e076c5136cb3, independently computed SHA-256 hashes match all three supplied source texts.
- Relative to baseline.py, model.py changes only the coefficient in NextItNetModel.encode_sequence from 0.0 to +0.05; control-1.py changes that same coefficient to -0.05. This matches the frozen insertion contract and author’s one-line-change claim.
- The existing prefix sum excludes the current and future positions, masks padding, and yields zero injection when there is no predecessor. The two arms apply equal-magnitude, opposite-sign coefficients before the final LayerNorm.
- The final normalization, 0.85/0.15 context blend, forward use of encode_sequence, and score_candidates use of the final encoded state are unchanged.

限制：

- The workspace supplies the sources in context.json and has no model/ directory to inspect separately.
- Host CPU contracts, execution manifests, and frozen evaluation are pending. This source approval does not establish runtime correctness, ranking benefit, or causal isolation.

本轮没有 role_rejected 事件。

## 实际调用与训练成本

| 调用目录 | 输入 token | 缓存输入 token | 输出 token | 用时秒 |
| --- | ---: | ---: | ---: | ---: |
| [analyst](../../records/research/rounds/round-16/analyst/request.json) | 248,353 | 203,648 | 3,327 | 53.3 |
| [experiment](../../records/research/rounds/round-16/experiment/request.json) | 174,715 | 151,168 | 1,957 | 57.7 |
| [researcher](../../records/research/rounds/round-16/researcher/request.json) | 537,742 | 464,640 | 5,632 | 78.2 |
| [reviewer](../../records/research/rounds/round-16/reviewer/request.json) | 339,405 | 307,200 | 4,054 | 71.1 |

缓存输入是输入 token 的子集；reasoning token 是输出的子集，不重复相加。这些计数不能直接换算 Pro 订阅额度。

| 训练任务 | 最佳 epoch | 完成 epoch | 分配 GPU 秒 |
| --- | ---: | ---: | ---: |
| [amazon-next-v9-round-16-control-1-CDs_and_Vinyl-42](../../records/study/jobs/amazon-next-v9-round-16-control-1-CDs_and_Vinyl-42/completed.json) | 95 | 115 | 729.7 |
| [amazon-next-v9-round-16-control-1-Electronics-42](../../records/study/jobs/amazon-next-v9-round-16-control-1-Electronics-42/completed.json) | 40 | 60 | 391.9 |
| [amazon-next-v9-round-16-control-1-Industrial_and_Scientific-42](../../records/study/jobs/amazon-next-v9-round-16-control-1-Industrial_and_Scientific-42/completed.json) | 110 | 130 | 799.0 |
| [amazon-next-v9-round-16-control-1-Movies_and_TV-42](../../records/study/jobs/amazon-next-v9-round-16-control-1-Movies_and_TV-42/completed.json) | 65 | 85 | 556.8 |
| [amazon-next-v9-round-16-model-CDs_and_Vinyl-42](../../records/study/jobs/amazon-next-v9-round-16-model-CDs_and_Vinyl-42/completed.json) | 45 | 65 | 485.7 |
| [amazon-next-v9-round-16-model-Electronics-42](../../records/study/jobs/amazon-next-v9-round-16-model-Electronics-42/completed.json) | 5 | 25 | 225.9 |
| [amazon-next-v9-round-16-model-Industrial_and_Scientific-42](../../records/study/jobs/amazon-next-v9-round-16-model-Industrial_and_Scientific-42/completed.json) | 70 | 90 | 601.6 |
| [amazon-next-v9-round-16-model-Movies_and_TV-42](../../records/study/jobs/amazon-next-v9-round-16-model-Movies_and_TV-42/completed.json) | 60 | 80 | 494.2 |

完整训练曲线在对应任务的 curve.jsonl；此表只计本轮新版本，复用基线不重复计费。

## 提案原文与源码差异

### 机制

Compare positive pre-normalization incorporation of preceding encoded states with the zero-injection incumbent and an equal-magnitude negative-injection control. Fixed-state CPU checks establish the intended computation; frozen ranking comparisons test usefulness without proving causal attribution.

### 预期观测

Provisional prediction: candidate seed-42 unweighted four-category mean HR@10 exceeds 0.60475, strictly more than 0.005 above the observed 0.59975 incumbent mean, and exceeds the generated negative-injection control. Report all four category effects and available history-length slices with counts; no slice gain is established.

### 证伪条件

Reject source admission if the formula, masking, causality or scoring contracts fail. Reject development advancement if candidate mean HR@10 is at most 0.60475, does not exceed the generated control, or fails host eligibility. A passing aggregate remains provisional mechanism evidence.

### 计划正确性检查

- Verify candidate model.py and generated control-1.py differ from the incumbent only inside NextItNetModel.encode_sequence, with coefficients exactly +0.05 and -0.05 respectively.
- Against an explicit CPU prefix calculation, verify equal-magnitude opposite-sign injections for fixed nonzero predecessor states and identical behavior when there is no predecessor.
- Check empty, singleton, dense and right-aligned histories; causality, finite outputs and gradients, zero padding gradients, batch independence and permutation equivariance before and after a synthetic update.
- Verify forward and score_candidates identities after an update and audit independent initialization and frozen execution manifests.

- [候选相对基线的 diff](candidate.diff) · [候选完整源码](../../records/research/rounds/round-16/experiment/model/model.py)
- [对照相对基线的 diff](control.diff) · [对照完整源码](../../records/research/rounds/round-16/experiment/model/control-1.py)

## 原始证据导航

- [研究员完整提案](../../records/research/rounds/round-16/researcher/response.json)
- [研究员实际上下文](../../records/research/rounds/round-16/researcher/context.json)
- [实验员返回](../../records/research/rounds/round-16/experiment/response.json)
- [本轮全部归档](../../records/research/rounds/round-16)
- [拒绝事件摘录](rejections.json)
- [六项指标与机制比较](../../records/research/rounds/round-16/development-readout.json)
- [分析员完整返回与知识写入](../../records/research/rounds/round-16/analyst/response.json)
- [晋升记录](../../records/research/rounds/round-16/promotion.json)
