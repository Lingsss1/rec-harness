# 第 05 轮 逐特征分配历史信息

[返回总览](../../README.md) · [机器可读汇总](../../rounds.json)

状态：**未晋升**。当轮基线：`amazon-next-v8-round-1-control-1`。

## 假设与实验设计

按特征调整当前状态与历史状态的分配可减少冲突。

| 版本 | 改动 |
| --- | --- |
| 候选 | 在原混合上加入 q*(c-u)，q=0.15*tanh(a)。 |
| 机制对照 | 加入 q*(0.85u+0.15c)，只校准整体混合。 |

基线和机制对照用途不同：前者判断是否值得保留，后者帮助区分解释。所有已训练版本使用 seed 42；并不承诺新增结构与基线逐参数初始化完全一致。

## 实验结果

| 品类 | 基线 HR@10 | 候选 HR@10 | 对照 HR@10 | 候选减基线 | 候选减对照 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Movies_and_TV | 0.64800 | 0.63200 | 0.64000 | -0.01600 | -0.00800 |
| Industrial_and_Scientific | 0.56100 | 0.55700 | 0.56800 | -0.00400 | -0.01100 |
| Electronics | 0.48600 | 0.45100 | 0.45200 | -0.03500 | -0.00100 |
| CDs_and_Vinyl | 0.68400 | 0.68200 | 0.66400 | -0.00200 | +0.01800 |
| **四品类平均** | **0.59475** | **0.58050** | **0.58100** | **-0.01425** | **-0.00050** |

Host 晋升：`无`；选择策略 `baseline-single-seed-v1`；confirmation_executed = `False`。

## 中文归因与审阅关注点

两者几乎相同且均退步；匹配参数量并未证明隔离了信息分配机制。

分析员结论标签：`unknown`；建议：`discard`。这里的 keep 是保留开发集记录供 host 选择，不等于机制确认。

### 分析员原始理由

Audited seed-42 mean HR@10 was 0.5805 for the candidate, 0.59475 for the baseline, and 0.581 for the matched control. The candidate missed the proposed 0.60425 threshold, and host eligibility is false. Its mean contrast with the control was nearly null (-0.0005), while category contrasts differed in sign. These development outcomes do not identify the causal mechanism.

### 尚未排除的解释

- The incumbent fixed blend may already be calibrated.
- Independent initialization, optimization, or validation checkpoint selection may explain the small candidate/control contrast.
- Matched parameter counts do not establish that the control isolates disagreement allocation.
- Fixed-state mechanism oracles after a parameter update, score identities after save/load, and independent seed replication are not documented as completed.
- History-length slices are exploratory, with sparse long-history groups; no confirmation result is supplied.

## 审核和返回修改


### reviewer

原始判定：`approve`。[完整记录](../../records/research/rounds/round-5/reviewer/response.json)

阻塞项：



已检查：

- Reviewed implementation_hash d7e2979414cf9877ec5c07629656fb3bad1aa8c876ae5571cbaca051f3465ee3. Independently verified the SHA-256 hashes of all three supplied source texts.
- Both arms change only NextItNetModel.__init__ and encode_sequence. Each adds the same zero initialized hidden-width parameter; training settings, masking, preceding-context construction, fallback, forward, and candidate scoring are unchanged.
- The candidate implements q*(c-u) and the control implements q*(0.85u+0.15c), with q=0.15*tanh(a). Both match the incumbent readout at initialization and implement their stated interventions.

限制：

- Host CPU admission, synthetic mechanism oracles, gradient and scoring checks, and independent initialization audits are pending. Source approval does not establish runtime correctness or causal isolation.
- The source bundle’s implementation_hash aggregation method was not supplied; the individual source SHA-256 hashes were verified.

本轮没有 role_rejected 事件。

## 实际调用与训练成本

| 调用目录 | 输入 token | 缓存输入 token | 输出 token | 用时秒 |
| --- | ---: | ---: | ---: | ---: |
| [analyst](../../records/research/rounds/round-5/analyst/request.json) | 415,398 | 357,888 | 5,984 | 92.3 |
| [experiment](../../records/research/rounds/round-5/experiment/request.json) | 202,793 | 176,128 | 1,826 | 46.5 |
| [researcher](../../records/research/rounds/round-5/researcher/request.json) | 585,696 | 504,320 | 5,477 | 133.3 |
| [reviewer](../../records/research/rounds/round-5/reviewer/request.json) | 215,976 | 163,968 | 2,909 | 60.4 |

缓存输入是输入 token 的子集；reasoning token 是输出的子集，不重复相加。这些计数不能直接换算 Pro 订阅额度。

| 训练任务 | 最佳 epoch | 完成 epoch | 分配 GPU 秒 |
| --- | ---: | ---: | ---: |
| [amazon-next-v9-round-5-control-1-CDs_and_Vinyl-42](../../records/study/jobs/amazon-next-v9-round-5-control-1-CDs_and_Vinyl-42/completed.json) | 55 | 75 | 473.9 |
| [amazon-next-v9-round-5-control-1-Electronics-42](../../records/study/jobs/amazon-next-v9-round-5-control-1-Electronics-42/completed.json) | 5 | 25 | 165.3 |
| [amazon-next-v9-round-5-control-1-Industrial_and_Scientific-42](../../records/study/jobs/amazon-next-v9-round-5-control-1-Industrial_and_Scientific-42/completed.json) | 75 | 95 | 589.1 |
| [amazon-next-v9-round-5-control-1-Movies_and_TV-42](../../records/study/jobs/amazon-next-v9-round-5-control-1-Movies_and_TV-42/completed.json) | 50 | 70 | 434.0 |
| [amazon-next-v9-round-5-model-CDs_and_Vinyl-42](../../records/study/jobs/amazon-next-v9-round-5-model-CDs_and_Vinyl-42/completed.json) | 95 | 115 | 739.9 |
| [amazon-next-v9-round-5-model-Electronics-42](../../records/study/jobs/amazon-next-v9-round-5-model-Electronics-42/completed.json) | 15 | 35 | 282.0 |
| [amazon-next-v9-round-5-model-Industrial_and_Scientific-42](../../records/study/jobs/amazon-next-v9-round-5-model-Industrial_and_Scientific-42/completed.json) | 95 | 115 | 760.6 |
| [amazon-next-v9-round-5-model-Movies_and_TV-42](../../records/study/jobs/amazon-next-v9-round-5-model-Movies_and_TV-42/completed.json) | 60 | 80 | 550.5 |

完整训练曲线在对应任务的 curve.jsonl；此表只计本轮新版本，复用基线不重复计费。

## 提案原文与源码差异

### 机制

Compare featurewise allocation toward or away from preceding context against the incumbent fixed blend and a matched-parameter common-mode calibration control. The contrast tests the provisional disagreement-allocation explanation; independent fits cannot alone prove it.

### 预期观测

Hypothesis: seed-42 unweighted four-category mean HR@10 exceeds the incumbent 0.59925 by more than 0.005 and exceeds the generated control. Report all four category effects and available history-length slices, including regressions. Slice effects are unknown.

### 证伪条件

Reject implementation admission if CPU or scoring contracts fail. The ranking prediction fails if candidate mean HR@10 is at most 0.60425, candidate mean is at most the generated control's mean, or the host eligibility gate is false. A passing development contrast remains provisional rather than causal confirmation.

### 计划正确性检查

- Confirm both arms equal the incumbent at initialization and retain the no-predecessor fallback.
- Check the fixed-state disagreement and common-mode CPU oracles before and after a parameter update.
- Check finite outputs and gradients, future-token invariance, padding gradient zero, batch independence, and unchanged inputs.
- Check forward and tail candidate-score identities after an update and save/load.
- Audit independent initialization and the frozen training and evaluation manifests.

- [候选相对基线的 diff](candidate.diff) · [候选完整源码](../../records/research/rounds/round-5/experiment/model/model.py)
- [对照相对基线的 diff](control.diff) · [对照完整源码](../../records/research/rounds/round-5/experiment/model/control-1.py)

## 原始证据导航

- [研究员完整提案](../../records/research/rounds/round-5/researcher/response.json)
- [研究员实际上下文](../../records/research/rounds/round-5/researcher/context.json)
- [实验员返回](../../records/research/rounds/round-5/experiment/response.json)
- [本轮全部归档](../../records/research/rounds/round-5)
- [拒绝事件摘录](rejections.json)
- [六项指标与机制比较](../../records/research/rounds/round-5/development-readout.json)
- [分析员完整返回与知识写入](../../records/research/rounds/round-5/analyst/response.json)
- [晋升记录](../../records/research/rounds/round-5/promotion.json)
