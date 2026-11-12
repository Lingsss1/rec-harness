# 第 14 轮 卷积 dilation 顺序

[返回总览](../../README.md) · [机器可读汇总](../../rounds.json)

状态：**晋升候选**。当轮基线：`amazon-next-v9-round-8-control-1`。

## 假设与实验设计

先扩大再缩小感受野，末端局部加工可能改善排名。

| 版本 | 改动 |
| --- | --- |
| 候选 | 顺序 [1,2,4,8,8,4,2,1]。 |
| 机制对照 | 顺序 [8,4,2,1,1,2,4,8]，保持 dilation 多重集相同。 |

基线和机制对照用途不同：前者判断是否值得保留，后者帮助区分解释。所有已训练版本使用 seed 42；并不承诺新增结构与基线逐参数初始化完全一致。

## 实验结果

| 品类 | 基线 HR@10 | 候选 HR@10 | 对照 HR@10 | 候选减基线 | 候选减对照 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Movies_and_TV | 0.67400 | 0.65400 | 0.67300 | -0.02000 | -0.01900 |
| Industrial_and_Scientific | 0.56400 | 0.57900 | 0.57300 | +0.01500 | +0.00600 |
| Electronics | 0.46400 | 0.46700 | 0.46500 | +0.00300 | +0.00200 |
| CDs_and_Vinyl | 0.67900 | 0.68700 | 0.66800 | +0.00800 | +0.01900 |
| **四品类平均** | **0.59525** | **0.59675** | **0.59475** | **+0.00150** | **+0.00200** |

Host 晋升：`amazon-next-v9-round-14-model`；选择策略 `baseline-single-seed-v1`；confirmation_executed = `False`。

## 中文归因与审阅关注点

候选晋升，但多个层的 dilation 同时变化；无法把小幅收益单独归因于末端局部精炼。Movies 品类反而下降。

分析员结论标签：`provisional`；建议：`keep`。这里的 keep 是保留开发集记录供 host 选择，不等于机制确认。

### 分析员原始理由

The audited seed-42 public-development mean HR@10 is 0.59675 for the candidate, 0.59525 for the incumbent, and 0.59475 for the generated control. The host marks the candidate eligible under its baseline-only rule, so retain it for host selection. The candidate's +0.00150 gain over the incumbent misses the proposal's >0.005 prediction and its >0.60025 mean target. Movies_and_TV fell by 0.020 versus the incumbent and 0.019 versus the control; the control's -0.00050 aggregate contrast is near null. These outcomes do not establish late local refinement as the cause or show confirmed generalization.

### 尚未排除的解释

- The candidate and control change multiple dilation positions. Optimization and checkpoint selection could explain the small aggregate difference instead of final-block local refinement.
- The single seed and differing category effects leave stability unresolved. Exploratory history slices include positive, negative, and null effects; sparse long-history bins cannot establish a history-dependent mechanism.
- Host admission and synthetic pre- and post-training checks cover sampled behavior, but do not establish all-input behavior or isolate the proposed mechanism. The author's own CPU checks were unexecuted, and the proposal's full diagnostic set is not separately demonstrated.
- No confirmation or final-test result is supplied.

## 审核和返回修改


### reviewer

原始判定：`approve`。[完整记录](../../records/research/rounds/round-14/reviewer/response.json)

阻塞项：



已检查：

- For implementation_hash 86c7621e38dd86b65d834c5deddf4c4700fe979d9173933491176de36e6f4180, independently computed SHA-256 hashes match the declared baseline, candidate, and control source hashes.
- Each source differs from the baseline only at model.py:218 or control-1.py:218, within NextItNetModel.__init__. The candidate and control use the exact proposed eight-block schedules; their dilation multisets and block counts match.
- The unchanged causal convolutions, encode_sequence, forward, and score_candidates paths preserve the source-level masking and scoring relationships. The control implements its stated schedule intervention.

限制：

- /workspace/model was not supplied; review used the three source texts in /workspace/context.json.
- Host CPU tensor, gradient, and scoring checks are pending. Source approval does not establish runtime correctness, ranking benefit, or causal isolation.

本轮没有 role_rejected 事件。

## 实际调用与训练成本

| 调用目录 | 输入 token | 缓存输入 token | 输出 token | 用时秒 |
| --- | ---: | ---: | ---: | ---: |
| [analyst](../../records/research/rounds/round-14/analyst/request.json) | 319,249 | 263,552 | 5,284 | 67.0 |
| [experiment](../../records/research/rounds/round-14/experiment/request.json) | 294,326 | 265,472 | 2,125 | 61.4 |
| [researcher](../../records/research/rounds/round-14/researcher/request.json) | 592,686 | 519,680 | 6,430 | 142.6 |
| [reviewer](../../records/research/rounds/round-14/reviewer/request.json) | 313,822 | 279,296 | 3,246 | 85.6 |

缓存输入是输入 token 的子集；reasoning token 是输出的子集，不重复相加。这些计数不能直接换算 Pro 订阅额度。

| 训练任务 | 最佳 epoch | 完成 epoch | 分配 GPU 秒 |
| --- | ---: | ---: | ---: |
| [amazon-next-v9-round-14-control-1-CDs_and_Vinyl-42](../../records/study/jobs/amazon-next-v9-round-14-control-1-CDs_and_Vinyl-42/completed.json) | 55 | 75 | 471.7 |
| [amazon-next-v9-round-14-control-1-Electronics-42](../../records/study/jobs/amazon-next-v9-round-14-control-1-Electronics-42/completed.json) | 55 | 75 | 431.1 |
| [amazon-next-v9-round-14-control-1-Industrial_and_Scientific-42](../../records/study/jobs/amazon-next-v9-round-14-control-1-Industrial_and_Scientific-42/completed.json) | 75 | 95 | 538.4 |
| [amazon-next-v9-round-14-control-1-Movies_and_TV-42](../../records/study/jobs/amazon-next-v9-round-14-control-1-Movies_and_TV-42/completed.json) | 90 | 110 | 705.8 |
| [amazon-next-v9-round-14-model-CDs_and_Vinyl-42](../../records/study/jobs/amazon-next-v9-round-14-model-CDs_and_Vinyl-42/completed.json) | 95 | 115 | 782.9 |
| [amazon-next-v9-round-14-model-Electronics-42](../../records/study/jobs/amazon-next-v9-round-14-model-Electronics-42/completed.json) | 40 | 60 | 431.9 |
| [amazon-next-v9-round-14-model-Industrial_and_Scientific-42](../../records/study/jobs/amazon-next-v9-round-14-model-Industrial_and_Scientific-42/completed.json) | 75 | 95 | 609.8 |
| [amazon-next-v9-round-14-model-Movies_and_TV-42](../../records/study/jobs/amazon-next-v9-round-14-model-Movies_and_TV-42/completed.json) | 60 | 80 | 553.1 |

完整训练曲线在对应任务的 curve.jsonl；此表只计本轮新版本，复用基线不重复计费。

## 提案原文与源码差异

### 机制

Compare late local refinement with a distinct, equal-capacity dilation ordering and the audited incumbent. Structural CPU checks verify the intended orders and causal computation; frozen ranking comparisons test usefulness without establishing causal attribution.

### 预期观测

Provisional prediction: seed-42 unweighted four-category mean HR@10 exceeds 0.60025, strictly more than 0.005 above the observed 0.59525 incumbent mean, and exceeds the generated control. Report every category effect and available history-length slice with counts; no slice benefit is established.

### 证伪条件

Reject implementation admission if the schedules, causality, masking, or scoring contracts fail. Reject development advancement if candidate mean HR@10 is at most 0.60025, does not exceed the generated control, or fails host eligibility. Apply the frozen confirmation procedure only if development qualifies. An aggregate gain does not prove the proposed mechanism.

### 计划正确性检查

- Verify changes in model.py and control-1.py are confined to NextItNetModel.__init__; preserve signatures, imports, helpers, and unrelated code.
- Verify default schedules are exactly [1,2,4,8,8,4,2,1] and [8,4,2,1,1,2,4,8], with identical dilation multisets and parameter counts.
- Check empty, singleton, dense, and right-aligned histories; future-token invariance; finite outputs and gradients; zero padding gradients; and batch independence before and after a synthetic update.
- Verify forward and score_candidates identities after the update, fresh category/seed initialization, and frozen execution manifests.

- [候选相对基线的 diff](candidate.diff) · [候选完整源码](../../records/research/rounds/round-14/experiment/model/model.py)
- [对照相对基线的 diff](control.diff) · [对照完整源码](../../records/research/rounds/round-14/experiment/model/control-1.py)

## 原始证据导航

- [研究员完整提案](../../records/research/rounds/round-14/researcher/response.json)
- [研究员实际上下文](../../records/research/rounds/round-14/researcher/context.json)
- [实验员返回](../../records/research/rounds/round-14/experiment/response.json)
- [本轮全部归档](../../records/research/rounds/round-14)
- [拒绝事件摘录](rejections.json)
- [六项指标与机制比较](../../records/research/rounds/round-14/development-readout.json)
- [分析员完整返回与知识写入](../../records/research/rounds/round-14/analyst/response.json)
- [晋升记录](../../records/research/rounds/round-14/promotion.json)
