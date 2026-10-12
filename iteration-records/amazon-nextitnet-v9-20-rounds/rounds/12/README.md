# 第 12 轮 最终读出的历史强度

[返回总览](../../README.md) · [机器可读汇总](../../rounds.json)

状态：**未晋升**。当轮基线：`amazon-next-v9-round-8-control-1`。

## 假设与实验设计

移除读出阶段历史混合可减少信息干扰。

| 版本 | 改动 |
| --- | --- |
| 候选 | 原 0.85/0.15 混合改为 1.00/0.00。 |
| 机制对照 | 改为 0.70/0.30。 |

基线和机制对照用途不同：前者判断是否值得保留，后者帮助区分解释。所有已训练版本使用 seed 42；并不承诺新增结构与基线逐参数初始化完全一致。

## 实验结果

| 品类 | 基线 HR@10 | 候选 HR@10 | 对照 HR@10 | 候选减基线 | 候选减对照 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Movies_and_TV | 0.67400 | 0.63500 | 0.65800 | -0.03900 | -0.02300 |
| Industrial_and_Scientific | 0.56400 | 0.53800 | 0.57100 | -0.02600 | -0.03300 |
| Electronics | 0.46400 | 0.44500 | 0.46800 | -0.01900 | -0.02300 |
| CDs_and_Vinyl | 0.67900 | 0.67400 | 0.66900 | -0.00500 | +0.00500 |
| **四品类平均** | **0.59525** | **0.57300** | **0.59150** | **-0.02225** | **-0.01850** |

Host 晋升：`无`；选择策略 `baseline-single-seed-v1`；confirmation_executed = `False`。

## 中文归因与审阅关注点

移除历史后明显下降；支持在本轮保留原配置，不支持历史权重越高越好。注意这不是第 15 轮删除的归一化前注入。

分析员结论标签：`provisional`；建议：`discard`。这里的 keep 是保留开发集记录供 host 选择，不等于机制确认。

### 分析员原始理由

The host-audited seed-42 development mean HR@10 was 0.573 for the candidate, 0.59525 for the incumbent, and 0.5915 for the higher-context control. The candidate was 0.02225 below the incumbent and 0.0185 below the control; host eligibility for confirmation is false. The audited execution and synthetic checks support that the development comparison ran under the frozen protocol, but the ranking result does not establish why performance fell.

### 尚未排除的解释

- The preceding-context contribution may help ranking; removing it may have caused the observed loss.
- Changing the blend may have changed representation magnitude or optimization. The paired development results do not isolate these effects from context information.
- Only one development seed is supplied. Category and history-length slices are dependent parts of that comparison, and checkpoint selection may affect the contrast.
- The supplied checks do not document an explicit fixed-state comparison of all three readout formulas for zero, one, and multiple predecessors, or every proposed scoring identity after a synthetic update. Finite synthetic probes and successful training do not establish all-input semantics or a causal mechanism.
- No confirmation results or independent replication are supplied.

## 审核和返回修改


### reviewer

原始判定：`approve`。[完整记录](../../records/research/rounds/round-12/reviewer/response.json)

阻塞项：



已检查：

- The supplied baseline, candidate, and control source texts match their declared SHA-256 hashes.
- Relative to baseline.py, model.py changes only the final blend in NextItNetModel.encode_sequence from 0.85/0.15 to 1.00/0.00; control-1.py changes only that line to 0.70/0.30. This matches the frozen insertion contract and the author's change claim.
- The shifted masked prefix sums use strictly preceding valid positions. Both arms retain the current-state fallback when there is no predecessor and mask padding outputs.
- forward and score_candidates both consume encode_sequence; their source expressions preserve the stated score identities. The training helpers and other source code are unchanged.

限制：

- Host CPU tensor checks and execution-manifest admission are pending. This source review does not establish runtime correctness, ranking benefit, or causal isolation.
- No /workspace/model directory was supplied; the review used the source texts embedded in /workspace/context.json.

本轮没有 role_rejected 事件。

## 实际调用与训练成本

| 调用目录 | 输入 token | 缓存输入 token | 输出 token | 用时秒 |
| --- | ---: | ---: | ---: | ---: |
| [analyst](../../records/research/rounds/round-12/analyst/request.json) | 231,417 | 186,624 | 4,211 | 48.4 |
| [experiment](../../records/research/rounds/round-12/experiment/request.json) | 244,184 | 216,960 | 1,577 | 50.6 |
| [researcher](../../records/research/rounds/round-12/researcher/request.json) | 699,549 | 629,376 | 4,890 | 67.9 |
| [reviewer](../../records/research/rounds/round-12/reviewer/request.json) | 337,929 | 304,000 | 3,573 | 51.5 |

缓存输入是输入 token 的子集；reasoning token 是输出的子集，不重复相加。这些计数不能直接换算 Pro 订阅额度。

| 训练任务 | 最佳 epoch | 完成 epoch | 分配 GPU 秒 |
| --- | ---: | ---: | ---: |
| [amazon-next-v9-round-12-control-1-CDs_and_Vinyl-42](../../records/study/jobs/amazon-next-v9-round-12-control-1-CDs_and_Vinyl-42/completed.json) | 45 | 65 | 338.4 |
| [amazon-next-v9-round-12-control-1-Electronics-42](../../records/study/jobs/amazon-next-v9-round-12-control-1-Electronics-42/completed.json) | 25 | 45 | 292.8 |
| [amazon-next-v9-round-12-control-1-Industrial_and_Scientific-42](../../records/study/jobs/amazon-next-v9-round-12-control-1-Industrial_and_Scientific-42/completed.json) | 55 | 75 | 491.1 |
| [amazon-next-v9-round-12-control-1-Movies_and_TV-42](../../records/study/jobs/amazon-next-v9-round-12-control-1-Movies_and_TV-42/completed.json) | 55 | 75 | 497.5 |
| [amazon-next-v9-round-12-model-CDs_and_Vinyl-42](../../records/study/jobs/amazon-next-v9-round-12-model-CDs_and_Vinyl-42/completed.json) | 70 | 90 | 635.8 |
| [amazon-next-v9-round-12-model-Electronics-42](../../records/study/jobs/amazon-next-v9-round-12-model-Electronics-42/completed.json) | 15 | 35 | 283.6 |
| [amazon-next-v9-round-12-model-Industrial_and_Scientific-42](../../records/study/jobs/amazon-next-v9-round-12-model-Industrial_and_Scientific-42/completed.json) | 25 | 45 | 340.6 |
| [amazon-next-v9-round-12-model-Movies_and_TV-42](../../records/study/jobs/amazon-next-v9-round-12-model-Movies_and_TV-42/completed.json) | 65 | 85 | 608.8 |

完整训练曲线在对应任务的 curve.jsonl；此表只计本轮新版本，复用基线不重复计费。

## 提案原文与源码差异

### 机制

Compare a lower-context readout against the audited incumbent and a symmetric higher-context control under the same frozen protocol. The CPU contrast establishes which formulas run; paired development and conditional confirmation test the ranking prediction, while causal attribution remains provisional.

### 预期观测

Falsifiable prediction: seed-42 unweighted four-category mean HR@10 exceeds 0.60025, more than 0.005 above the observed 0.59525 incumbent mean, and exceeds the generated higher-context control. Report all category effects and available history-length slices with user counts. No slice benefit is established in advance.

### 证伪条件

Reject implementation admission if the formulas, causality, masking or scoring contracts fail. Do not advance if candidate mean HR@10 is at most 0.60025, does not exceed the generated control, or fails host eligibility. Comparable gains from the higher-context control weaken the directional explanation. Apply frozen confirmation only if development qualifies; an aggregate gain alone does not prove the mechanism.

### 计划正确性检查

- Verify edits are confined to NextItNetModel.encode_sequence in model.py and control-1.py; preserve imports, signatures, configuration, training helpers and unrelated code.
- Compare candidate, incumbent and control outputs with explicit formulas on fixed states, including zero, one and multiple predecessors.
- Check dense and right-aligned histories, future-token invariance, finite outputs and gradients, zero padding gradients, batch independence and input immutability before and after a synthetic update.
- Verify forward score identities and the score_candidates tail dot-product identity after the update.
- Audit fresh per-category initialization and frozen execution manifests.

- [候选相对基线的 diff](candidate.diff) · [候选完整源码](../../records/research/rounds/round-12/experiment/model/model.py)
- [对照相对基线的 diff](control.diff) · [对照完整源码](../../records/research/rounds/round-12/experiment/model/control-1.py)

## 原始证据导航

- [研究员完整提案](../../records/research/rounds/round-12/researcher/response.json)
- [研究员实际上下文](../../records/research/rounds/round-12/researcher/context.json)
- [实验员返回](../../records/research/rounds/round-12/experiment/response.json)
- [本轮全部归档](../../records/research/rounds/round-12)
- [拒绝事件摘录](rejections.json)
- [六项指标与机制比较](../../records/research/rounds/round-12/development-readout.json)
- [分析员完整返回与知识写入](../../records/research/rounds/round-12/analyst/response.json)
- [晋升记录](../../records/research/rounds/round-12/promotion.json)
