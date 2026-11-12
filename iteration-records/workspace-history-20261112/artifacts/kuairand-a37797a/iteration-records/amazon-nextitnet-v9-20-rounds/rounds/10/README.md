# 第 10 轮 当前状态逐特征校准

[返回总览](../../README.md) · [机器可读汇总](../../rounds.json)

状态：**未晋升**。当轮基线：`amazon-next-v9-round-8-control-1`。

## 假设与实验设计

当前状态的各维度需要单独校准。

| 版本 | 改动 |
| --- | --- |
| 候选 | 增加 0.1*tanh(g)*compressed_current，g 从零开始。 |
| 机制对照 | 同样参数校准 compressed_preceding_context。 |

基线和机制对照用途不同：前者判断是否值得保留，后者帮助区分解释。所有已训练版本使用 seed 42；并不承诺新增结构与基线逐参数初始化完全一致。

## 实验结果

| 品类 | 基线 HR@10 | 候选 HR@10 | 对照 HR@10 | 候选减基线 | 候选减对照 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Movies_and_TV | 0.67400 | 0.65100 | 0.65300 | -0.02300 | -0.00200 |
| Industrial_and_Scientific | 0.56400 | 0.55800 | 0.56700 | -0.00600 | -0.00900 |
| Electronics | 0.46400 | 0.46000 | 0.46100 | -0.00400 | -0.00100 |
| CDs_and_Vinyl | 0.67900 | 0.65700 | 0.67600 | -0.02200 | -0.01900 |
| **四品类平均** | **0.59525** | **0.58150** | **0.58925** | **-0.01375** | **-0.00775** |

Host 晋升：`无`；选择策略 `baseline-single-seed-v1`；confirmation_executed = `False`。

## 中文归因与审阅关注点

候选四品类均不如基线和对照，未支持当前状态校准。

分析员结论标签：`unknown`；建议：`discard`。这里的 keep 是保留开发集记录供 host 选择，不等于机制确认。

### 分析员原始理由

In the audited seed-42 development readout, mean HR@10 was 0.5815 for the candidate, 0.58925 for the generated control, and 0.59525 for the incumbent. The candidate trailed both in all four categories and was ineligible for confirmation. Host checks support completed execution, but do not establish the proposed mechanism.

### 尚未排除的解释

- The proposed fixed-state, nonzero-gain formula contrast, zero-gain incumbent equality, distinct behavior after an update, and post-update scoring identity are not supplied as verified diagnostic results.
- Matched parameter counts do not isolate feature calibration from differences in source tensors or optimization. Initialization and checkpoint selection may also affect this single-seed result.
- Exploratory history slices include positive and null results, but lack a control-by-history contrast; sparse long-history bins limit interpretation.
- No confirmation result is supplied.

## 审核和返回修改


### reviewer

原始判定：`approve`。[完整记录](../../records/research/rounds/round-10/reviewer/response.json)

阻塞项：



已检查：

- The supplied baseline.py, model.py, and control-1.py source texts match their stated SHA-256 hashes. The reviewed implementation_hash is 7a3c644be3a4950ff5547599a950b4a85b1290033926b74347ed9786c8766cdb.
- Both changes are confined to NextItNetModel.__init__ and encode_sequence; unrelated source and frozen settings match the baseline.
- Both arms add one zero-initialized hidden-width gain with the fixed 0.1 × tanh(g) transform. model.py applies it to compressed current state; control-1.py applies it to compressed preceding context only when a predecessor exists. Both retain the final valid-token mask.
- At zero gain, both formulas equal the baseline. The unchanged forward and score_candidates methods consume encode_sequence, preserving their scoring relationship.

限制：

- The implementation_hash derivation was not supplied, so only the individual source hashes were independently recomputed.
- Host CPU semantic checks are pending. This source approval does not establish runtime correctness, ranking benefit, or causal isolation.

本轮没有 role_rejected 事件。

## 实际调用与训练成本

| 调用目录 | 输入 token | 缓存输入 token | 输出 token | 用时秒 |
| --- | ---: | ---: | ---: | ---: |
| [analyst](../../records/research/rounds/round-10/analyst/request.json) | 344,142 | 288,128 | 5,573 | 68.5 |
| [experiment](../../records/research/rounds/round-10/experiment/request.json) | 203,136 | 177,408 | 1,822 | 37.5 |
| [researcher](../../records/research/rounds/round-10/researcher/request.json) | 673,804 | 598,016 | 7,289 | 103.2 |
| [reviewer](../../records/research/rounds/round-10/reviewer/request.json) | 209,130 | 180,608 | 2,566 | 64.7 |

缓存输入是输入 token 的子集；reasoning token 是输出的子集，不重复相加。这些计数不能直接换算 Pro 订阅额度。

| 训练任务 | 最佳 epoch | 完成 epoch | 分配 GPU 秒 |
| --- | ---: | ---: | ---: |
| [amazon-next-v9-round-10-control-1-CDs_and_Vinyl-42](../../records/study/jobs/amazon-next-v9-round-10-control-1-CDs_and_Vinyl-42/completed.json) | 75 | 95 | 567.8 |
| [amazon-next-v9-round-10-control-1-Electronics-42](../../records/study/jobs/amazon-next-v9-round-10-control-1-Electronics-42/completed.json) | 15 | 35 | 247.0 |
| [amazon-next-v9-round-10-control-1-Industrial_and_Scientific-42](../../records/study/jobs/amazon-next-v9-round-10-control-1-Industrial_and_Scientific-42/completed.json) | 80 | 100 | 651.8 |
| [amazon-next-v9-round-10-control-1-Movies_and_TV-42](../../records/study/jobs/amazon-next-v9-round-10-control-1-Movies_and_TV-42/completed.json) | 70 | 90 | 515.8 |
| [amazon-next-v9-round-10-model-CDs_and_Vinyl-42](../../records/study/jobs/amazon-next-v9-round-10-model-CDs_and_Vinyl-42/completed.json) | 35 | 55 | 403.6 |
| [amazon-next-v9-round-10-model-Electronics-42](../../records/study/jobs/amazon-next-v9-round-10-model-Electronics-42/completed.json) | 15 | 35 | 246.1 |
| [amazon-next-v9-round-10-model-Industrial_and_Scientific-42](../../records/study/jobs/amazon-next-v9-round-10-model-Industrial_and_Scientific-42/completed.json) | 55 | 75 | 532.3 |
| [amazon-next-v9-round-10-model-Movies_and_TV-42](../../records/study/jobs/amazon-next-v9-round-10-model-Movies_and_TV-42/completed.json) | 90 | 110 | 703.2 |

完整训练曲线在对应任务的 curve.jsonl；此表只计本轮新版本，复用基线不重复计费。

## 提案原文与源码差异

### 机制

Compare current encoded-feature calibration against the audited incumbent and an equal-parameter preceding-context calibration control. The fixed-state CPU contrast tests the distinct computations; the frozen ranking contrast tests their practical effect while leaving causal attribution provisional.

### 预期观测

Falsifiable prediction: seed-42 unweighted four-category mean HR@10 exceeds 0.60025, more than 0.005 above the observed 0.59525 incumbent mean, and exceeds the generated control. Report all four category effects and any available history-length slices with counts; no slice gain is established in advance.

### 证伪条件

Reject implementation admission if branch formulas, causality, masking, or scoring contracts fail. Reject development advancement if candidate mean HR@10 is at most 0.60025, is at most the generated control mean, or fails host eligibility. If development qualifies, apply the frozen confirmation procedure. A ranking gain alone does not prove feature calibration caused it.

### 计划正确性检查

- Confine edits to NextItNetModel.__init__ and encode_sequence in model.py and control-1.py; preserve imports, helpers, signatures, and frozen settings.
- With fixed states and identical nonzero gains, compare both branch formulas to explicit CPU calculations, including singleton, dense, empty, and right-aligned histories.
- Verify zero-gain equality to the incumbent, future-token invariance, finite values and gradients, padding behavior, batch independence, and distinct candidate/control behavior after a synthetic update.
- Verify forward and score_candidates identities after an update, plus independent initialization and frozen execution manifests.

- [候选相对基线的 diff](candidate.diff) · [候选完整源码](../../records/research/rounds/round-10/experiment/model/model.py)
- [对照相对基线的 diff](control.diff) · [对照完整源码](../../records/research/rounds/round-10/experiment/model/control-1.py)

## 原始证据导航

- [研究员完整提案](../../records/research/rounds/round-10/researcher/response.json)
- [研究员实际上下文](../../records/research/rounds/round-10/researcher/context.json)
- [实验员返回](../../records/research/rounds/round-10/experiment/response.json)
- [本轮全部归档](../../records/research/rounds/round-10)
- [拒绝事件摘录](rejections.json)
- [六项指标与机制比较](../../records/research/rounds/round-10/development-readout.json)
- [分析员完整返回与知识写入](../../records/research/rounds/round-10/analyst/response.json)
- [晋升记录](../../records/research/rounds/round-10/promotion.json)
