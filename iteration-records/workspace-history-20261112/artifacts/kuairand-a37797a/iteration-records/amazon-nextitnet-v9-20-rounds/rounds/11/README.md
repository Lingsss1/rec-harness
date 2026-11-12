# 第 11 轮 浅层状态跳连

[返回总览](../../README.md) · [机器可读汇总](../../rounds.json)

状态：**未晋升**。当轮基线：`amazon-next-v9-round-8-control-1`。

## 假设与实验设计

浅层特征可能在后续层中丢失。

| 版本 | 改动 |
| --- | --- |
| 候选 | 第 2 个 block 的状态经归一化和零初始化投影加入读出。 |
| 机制对照 | 第 7 个 block 的状态使用同样投影。 |

基线和机制对照用途不同：前者判断是否值得保留，后者帮助区分解释。所有已训练版本使用 seed 42；并不承诺新增结构与基线逐参数初始化完全一致。

## 实验结果

| 品类 | 基线 HR@10 | 候选 HR@10 | 对照 HR@10 | 候选减基线 | 候选减对照 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Movies_and_TV | 0.67400 | 0.66300 | 0.66100 | -0.01100 | +0.00200 |
| Industrial_and_Scientific | 0.56400 | 0.57500 | 0.58100 | +0.01100 | -0.00600 |
| Electronics | 0.46400 | 0.45100 | 0.45900 | -0.01300 | -0.00800 |
| CDs_and_Vinyl | 0.67900 | 0.67400 | 0.66800 | -0.00500 | +0.00600 |
| **四品类平均** | **0.59525** | **0.59075** | **0.59225** | **-0.00450** | **-0.00150** |

Host 晋升：`无`；选择策略 `baseline-single-seed-v1`；confirmation_executed = `False`。

## 中文归因与审阅关注点

浅层通路未显示优势；归一化并没有让不同深度的状态分布完全可比。

分析员结论标签：`provisional`；建议：`discard`。这里的 keep 是保留开发集记录供 host 选择，不等于机制确认。

### 分析员原始理由

Discard from advancement: audited seed-42 development mean HR@10 was 0.59075 for the candidate, 0.59225 for the matched seventh-block control, and 0.59525 for the incumbent. Neither new arm exceeded the incumbent, and host eligibility for confirmation is false. The candidate-control difference was -0.0015 overall, with positive and negative category contrasts. This selection result does not establish whether shallow features were preserved or useful.

### 尚未排除的解释

- The final encoder may already retain useful shallow features; the added paths may affect capacity or optimization; L2 normalization does not equalize the tapped state distributions; and single-seed initialization or checkpoint selection may contribute to the mixed effects.
- Host admission and synthetic pre- and post-training probes do not report the proposed fixed-state, nonzero-weight branch intervention, zero-initialization equivalence to the incumbent, or forward and scoring identities after a synthetic update. All-input semantics and causal isolation remain unproved.
- History-length results are exploratory dependent slices, include negative and null effects, and lack a matched control-by-history contrast. Sparse long-history bins cannot support a history-specific mechanism claim. No confirmation or final-test evidence was supplied.

## 审核和返回修改


### reviewer

原始判定：`approve`。[完整记录](../../records/research/rounds/round-11/reviewer/response.json)

阻塞项：



已检查：

- The supplied baseline, model.py, and control-1.py source texts match their stated SHA-256 hashes.
- Changes from the baseline are confined to NextItNetModel.__init__ and encode_sequence. Imports, signatures, training helpers, forward, and score_candidates are unchanged.
- Both arms add the same zero-initialized, bias-free projection and masked 0.1 × tanh residual after the incumbent blend. model.py taps the masked second-block output; control-1.py taps the masked seventh-block output. Tap depth is their only source difference.
- Zero projection weights leave the incumbent readout formula unchanged. The residual is inside encode_sequence, so forward and score_candidates use the same representation.

限制：

- Host CPU behavioral and protocol checks are pending. Source review does not establish runtime correctness, ranking benefit, or causal isolation.

本轮没有 role_rejected 事件。

## 实际调用与训练成本

| 调用目录 | 输入 token | 缓存输入 token | 输出 token | 用时秒 |
| --- | ---: | ---: | ---: | ---: |
| [analyst](../../records/research/rounds/round-11/analyst/request.json) | 331,569 | 277,760 | 4,511 | 103.8 |
| [experiment](../../records/research/rounds/round-11/experiment/request.json) | 223,843 | 195,456 | 2,434 | 64.8 |
| [researcher](../../records/research/rounds/round-11/researcher/request.json) | 527,424 | 453,760 | 5,338 | 74.2 |
| [reviewer](../../records/research/rounds/round-11/reviewer/request.json) | 200,967 | 174,336 | 2,409 | 43.4 |

缓存输入是输入 token 的子集；reasoning token 是输出的子集，不重复相加。这些计数不能直接换算 Pro 订阅额度。

| 训练任务 | 最佳 epoch | 完成 epoch | 分配 GPU 秒 |
| --- | ---: | ---: | ---: |
| [amazon-next-v9-round-11-control-1-CDs_and_Vinyl-42](../../records/study/jobs/amazon-next-v9-round-11-control-1-CDs_and_Vinyl-42/completed.json) | 60 | 80 | 555.2 |
| [amazon-next-v9-round-11-control-1-Electronics-42](../../records/study/jobs/amazon-next-v9-round-11-control-1-Electronics-42/completed.json) | 5 | 25 | 154.7 |
| [amazon-next-v9-round-11-control-1-Industrial_and_Scientific-42](../../records/study/jobs/amazon-next-v9-round-11-control-1-Industrial_and_Scientific-42/completed.json) | 75 | 95 | 631.2 |
| [amazon-next-v9-round-11-control-1-Movies_and_TV-42](../../records/study/jobs/amazon-next-v9-round-11-control-1-Movies_and_TV-42/completed.json) | 110 | 130 | 850.1 |
| [amazon-next-v9-round-11-model-CDs_and_Vinyl-42](../../records/study/jobs/amazon-next-v9-round-11-model-CDs_and_Vinyl-42/completed.json) | 70 | 90 | 604.6 |
| [amazon-next-v9-round-11-model-Electronics-42](../../records/study/jobs/amazon-next-v9-round-11-model-Electronics-42/completed.json) | 5 | 25 | 192.8 |
| [amazon-next-v9-round-11-model-Industrial_and_Scientific-42](../../records/study/jobs/amazon-next-v9-round-11-model-Industrial_and_Scientific-42/completed.json) | 75 | 95 | 609.9 |
| [amazon-next-v9-round-11-model-Movies_and_TV-42](../../records/study/jobs/amazon-next-v9-round-11-model-Movies_and_TV-42/completed.json) | 90 | 110 | 616.2 |

完整训练曲线在对应任务的 curve.jsonl；此表只计本轮新版本，复用基线不重复计费。

## 提案原文与源码差异

### 机制

The candidate-control contrast tests whether access to a shallow convolutional state helps more than equal-capacity access to a late state. CPU intervention checks test the intended computation; frozen ranking results test predictive value while leaving causal attribution provisional.

### 预期观测

Falsifiable prediction: the candidate's seed-42 unweighted four-category mean HR@10 exceeds 0.60025, strictly more than 0.005 above the observed incumbent mean of 0.59525, and exceeds the generated seventh-block control. Report category and history-length effects with user counts; no slice gain is established in advance.

### 证伪条件

Reject implementation admission if branch formulas, causality, masking, or scoring contracts fail. Reject development advancement if candidate mean HR@10 is at most 0.60025, does not exceed the generated control, or fails host eligibility. If it qualifies, apply the frozen confirmation procedure. A ranking gain alone does not prove the proposed mechanism.

### 计划正确性检查

- Verify edits are confined to the declared methods in model.py and control-1.py; preserve signatures, imports, training helpers, and unrelated code.
- On fixed CPU states with identical nonzero projections, compare each branch with its explicit formula and verify that changing only its tapped state changes only that branch.
- Check zero-initialization equivalence to the incumbent, then distinct outputs for suitable nonzero test weights.
- Check empty, singleton, dense, and right-aligned histories; future-token invariance; finite outputs and gradients; zero padding gradients; and batch independence before and after a synthetic update.
- Verify forward and score_candidates identities after the update and audit independent initialization and frozen execution manifests.

- [候选相对基线的 diff](candidate.diff) · [候选完整源码](../../records/research/rounds/round-11/experiment/model/model.py)
- [对照相对基线的 diff](control.diff) · [对照完整源码](../../records/research/rounds/round-11/experiment/model/control-1.py)

## 原始证据导航

- [研究员完整提案](../../records/research/rounds/round-11/researcher/response.json)
- [研究员实际上下文](../../records/research/rounds/round-11/researcher/context.json)
- [实验员返回](../../records/research/rounds/round-11/experiment/response.json)
- [本轮全部归档](../../records/research/rounds/round-11)
- [拒绝事件摘录](rejections.json)
- [六项指标与机制比较](../../records/research/rounds/round-11/development-readout.json)
- [分析员完整返回与知识写入](../../records/research/rounds/round-11/analyst/response.json)
- [晋升记录](../../records/research/rounds/round-11/promotion.json)
