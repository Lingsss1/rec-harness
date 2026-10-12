# 第 09 轮 直接物品身份投影

[返回总览](../../README.md) · [机器可读汇总](../../rounds.json)

状态：**未晋升**。当轮基线：`amazon-next-v9-round-8-control-1`。

## 假设与实验设计

直接物品通路可补足编码过程中丢失的身份信息。

| 版本 | 改动 |
| --- | --- |
| 候选 | 对当前物品 embedding 归一化、零初始化投影后加入残差。 |
| 机制对照 | 相同投影作用于最终编码状态。 |

基线和机制对照用途不同：前者判断是否值得保留，后者帮助区分解释。所有已训练版本使用 seed 42；并不承诺新增结构与基线逐参数初始化完全一致。

## 实验结果

| 品类 | 基线 HR@10 | 候选 HR@10 | 对照 HR@10 | 候选减基线 | 候选减对照 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Movies_and_TV | 0.67400 | 0.65900 | 0.65100 | -0.01500 | +0.00800 |
| Industrial_and_Scientific | 0.56400 | 0.57400 | 0.58600 | +0.01000 | -0.01200 |
| Electronics | 0.46400 | 0.45200 | 0.45700 | -0.01200 | -0.00500 |
| CDs_and_Vinyl | 0.67900 | 0.67900 | 0.67800 | +0.00000 | +0.00100 |
| **四品类平均** | **0.59525** | **0.59100** | **0.59300** | **-0.00425** | **-0.00200** |

Host 晋升：`无`；选择策略 `baseline-single-seed-v1`；confirmation_executed = `False`。

## 中文归因与审阅关注点

两者均低于基线，不支持新增直接物品通路；共享 embedding 梯度和输入分布也发生变化。

分析员结论标签：`provisional`；建议：`discard`。这里的 keep 是保留开发集记录供 host 选择，不等于机制确认。

### 分析员原始理由

In audited seed-42 public development, mean HR@10 was 0.591 for the candidate, 0.59525 for the incumbent, and 0.593 for the matched control. The candidate missed its preregistered 0.60025 threshold, and host eligibility for confirmation is false. Category effects were mixed, including a null candidate–incumbent effect for CDs_and_Vinyl. Admission and training checks support the recorded development comparison within their stated scope, but do not establish the proposed mechanism.

### 尚未排除的解释

- The fixed-encoder-state, nonzero-projection branch-formula contrast was not archived, so causal isolation of current-item identity remains unverified.
- The supplied checks do not individually demonstrate zero-initialization equivalence to the incumbent, distinct branch outputs after an update, or exact post-update forward and scoring identities.
- The mixed results remain compatible with existing identity retention, added capacity, unequal input distributions, shared-embedding gradient changes, optimization, checkpoint selection, or seed variation.
- History-length slices are exploratory; sparse, zero-user, and null slices cannot establish a long-range mechanism. No confirmation result or external mechanism source was supplied.
- Next discriminating experiment: run the fixed-state branch diagnostic, then preregister independent-seed comparisons under the frozen protocol with input-distribution and gradient measurements and an embedding-gradient-controlled ablation.

## 审核和返回修改


### reviewer

原始判定：`approve`。[完整记录](../../records/research/rounds/round-9/reviewer/response.json)

阻塞项：



已检查：

- Independently recomputed the SHA-256 hashes of all three supplied source texts; each matches its declared hash.
- Both changes are confined to NextItNetModel.__init__ and encode_sequence. Imports, signatures, training helpers, forward, and score_candidates are unchanged.
- Both arms add a bias-free hidden-to-hidden projection with zero-initialized weights, the same normalization and scaling, a 0.1 × tanh residual, and valid-token masking. At initialization, the residual is zero.
- The candidate projects the current item embedding; control-1.py projects the current post-encoder state. Their source texts otherwise match. The residual is inside encode_sequence, which both forward and score_candidates call.

限制：

- No /workspace/model directory or Python interpreter was supplied, so this review used the source texts in context.json and did not run CPU contract checks. Those checks remain pending host admission.
- The aggregate implementation_hash is supplied without a derivation rule; the three individual source hashes were independently verified. Approval is source review, not proof of runtime correctness, ranking benefit, or causal isolation.

本轮没有 role_rejected 事件。

## 实际调用与训练成本

| 调用目录 | 输入 token | 缓存输入 token | 输出 token | 用时秒 |
| --- | ---: | ---: | ---: | ---: |
| [analyst](../../records/research/rounds/round-9/analyst/request.json) | 354,687 | 301,056 | 5,731 | 124.0 |
| [experiment](../../records/research/rounds/round-9/experiment/request.json) | 221,743 | 192,640 | 1,850 | 56.3 |
| [researcher](../../records/research/rounds/round-9/researcher/request.json) | 604,157 | 541,184 | 5,109 | 89.4 |
| [reviewer](../../records/research/rounds/round-9/reviewer/request.json) | 243,980 | 212,224 | 4,387 | 99.3 |

缓存输入是输入 token 的子集；reasoning token 是输出的子集，不重复相加。这些计数不能直接换算 Pro 订阅额度。

| 训练任务 | 最佳 epoch | 完成 epoch | 分配 GPU 秒 |
| --- | ---: | ---: | ---: |
| [amazon-next-v9-round-9-control-1-CDs_and_Vinyl-42](../../records/study/jobs/amazon-next-v9-round-9-control-1-CDs_and_Vinyl-42/completed.json) | 70 | 90 | 509.5 |
| [amazon-next-v9-round-9-control-1-Electronics-42](../../records/study/jobs/amazon-next-v9-round-9-control-1-Electronics-42/completed.json) | 15 | 35 | 252.9 |
| [amazon-next-v9-round-9-control-1-Industrial_and_Scientific-42](../../records/study/jobs/amazon-next-v9-round-9-control-1-Industrial_and_Scientific-42/completed.json) | 75 | 95 | 635.3 |
| [amazon-next-v9-round-9-control-1-Movies_and_TV-42](../../records/study/jobs/amazon-next-v9-round-9-control-1-Movies_and_TV-42/completed.json) | 65 | 85 | 532.5 |
| [amazon-next-v9-round-9-model-CDs_and_Vinyl-42](../../records/study/jobs/amazon-next-v9-round-9-model-CDs_and_Vinyl-42/completed.json) | 90 | 110 | 759.6 |
| [amazon-next-v9-round-9-model-Electronics-42](../../records/study/jobs/amazon-next-v9-round-9-model-Electronics-42/completed.json) | 5 | 25 | 242.6 |
| [amazon-next-v9-round-9-model-Industrial_and_Scientific-42](../../records/study/jobs/amazon-next-v9-round-9-model-Industrial_and_Scientific-42/completed.json) | 105 | 125 | 856.8 |
| [amazon-next-v9-round-9-model-Movies_and_TV-42](../../records/study/jobs/amazon-next-v9-round-9-model-Movies_and_TV-42/completed.json) | 90 | 110 | 763.0 |

完整训练曲线在对应任务的 curve.jsonl；此表只计本轮新版本，复用基线不重复计费。

## 提案原文与源码差异

### 机制

Compare a direct current-item identity residual against both the audited incumbent and an equal-capacity residual from the current encoder state. The CPU contrast tests whether the branches compute the intended distinct signals; the frozen ranking comparison tests whether the direct path helps prediction.

### 预期观测

Falsifiable prediction: seed-42 unweighted four-category mean HR@10 exceeds 0.60025, more than 0.005 above the observed 0.59525 incumbent mean, and exceeds the generated equal-capacity control. Report each category effect and history-length slice with user counts. No slice gain is established in advance.

### 证伪条件

Reject implementation admission if branch formulas, causality, masking, or scoring contracts fail. Reject development advancement if candidate mean HR@10 is at most 0.60025, does not exceed the generated control, or fails host eligibility. If development qualifies, use the frozen confirmation procedure. An aggregate gain alone does not prove the proposed mechanism.

### 计划正确性检查

- Verify changes are confined to the declared methods in model.py and control-1.py; preserve signatures, imports, training helpers, and unrelated code.
- On fixed CPU tensors with the same assigned nonzero projection, compare both branch outputs to explicit formulas and verify the candidate responds to a changed raw item embedding when the encoder state is fixed.
- Check equality to the incumbent at zero initialization, then distinct candidate and control outputs after assigning the nonzero projection.
- Check empty, singleton, dense, and right-aligned histories; future-token perturbation invariance; finite outputs and gradients; zero padding gradients; and batch independence before and after a synthetic update.
- Verify forward and score_candidates identities after the update, independent category/seed initialization, and frozen execution manifests.

- [候选相对基线的 diff](candidate.diff) · [候选完整源码](../../records/research/rounds/round-9/experiment/model/model.py)
- [对照相对基线的 diff](control.diff) · [对照完整源码](../../records/research/rounds/round-9/experiment/model/control-1.py)

## 原始证据导航

- [研究员完整提案](../../records/research/rounds/round-9/researcher/response.json)
- [研究员实际上下文](../../records/research/rounds/round-9/researcher/context.json)
- [实验员返回](../../records/research/rounds/round-9/experiment/response.json)
- [本轮全部归档](../../records/research/rounds/round-9)
- [拒绝事件摘录](rejections.json)
- [六项指标与机制比较](../../records/research/rounds/round-9/development-readout.json)
- [分析员完整返回与知识写入](../../records/research/rounds/round-9/analyst/response.json)
- [晋升记录](../../records/research/rounds/round-9/promotion.json)
