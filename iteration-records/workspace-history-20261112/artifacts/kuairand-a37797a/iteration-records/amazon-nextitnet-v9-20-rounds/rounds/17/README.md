# 第 17 轮 有界注入与归一化位置

[返回总览](../../README.md) · [机器可读汇总](../../rounds.json)

状态：**未晋升**。当轮基线：`amazon-next-v9-round-15-model`。

## 假设与实验设计

把有界历史信号放在 LayerNorm 前可能比后面更有效。

| 版本 | 改动 |
| --- | --- |
| 候选 | 0.1*tanh(前缀均值) 在 LayerNorm 前加入。 |
| 机制对照 | 同一保存的信号在 LayerNorm 后加入。 |

基线和机制对照用途不同：前者判断是否值得保留，后者帮助区分解释。所有已训练版本使用 seed 42；并不承诺新增结构与基线逐参数初始化完全一致。

## 实验结果

| 品类 | 基线 HR@10 | 候选 HR@10 | 对照 HR@10 | 候选减基线 | 候选减对照 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Movies_and_TV | 0.66500 | 0.64700 | 0.64600 | -0.01800 | +0.00100 |
| Industrial_and_Scientific | 0.57200 | 0.57700 | 0.55200 | +0.00500 | +0.02500 |
| Electronics | 0.47700 | 0.48200 | 0.47900 | +0.00500 | +0.00300 |
| CDs_and_Vinyl | 0.68500 | 0.67800 | 0.68600 | -0.00700 | -0.00800 |
| **四品类平均** | **0.59975** | **0.59600** | **0.59075** | **-0.00375** | **+0.00525** |

Host 晋升：`无`；选择策略 `baseline-single-seed-v1`；confirmation_executed = `False`。

## 中文归因与审阅关注点

归一化前版本优于对照但不如基线；位置差异可能改变尺度而非证明历史信息有用。

分析员结论标签：`provisional`；建议：`discard`。这里的 keep 是保留开发集记录供 host 选择，不等于机制确认。

### 分析员原始理由

The audited seed-42 development mean HR@10 is 0.596 for the candidate, 0.59975 for the incumbent, and 0.59075 for the generated control. The candidate exceeds the control by 0.00525 but trails the incumbent by 0.00375, and the supplied readout marks it ineligible for confirmation. Category effects are mixed. The control contrast does not establish the proposed mechanism; no confirmation result is supplied.

### 尚未排除的解释

- LayerNorm placement may change signal scale or optimization without a history-specific benefit.
- Single-seed variation and validation checkpoint selection may explain the category-dependent contrast; CDs_and_Vinyl favors the control.
- The incumbent readout may already capture useful preceding-state information. History slices are exploratory and lack a control-by-history comparison.
- The supplied audited probes do not document every proposed fixed-state formula, history-condition, and post-update forward/scoring check. No independent replication or confirmation result is supplied.

## 审核和返回修改


### reviewer

原始判定：`approve`。[完整记录](../../records/research/rounds/round-17/reviewer/response.json)

阻塞项：



已检查：

- Reviewed implementation_hash 575ff7ca2c981bbfb4b77f3e1ec3a68c3a64fba2eea2ca490dc819b8b8d27f03. Independently recalculated all three supplied source hashes; each matches its declared SHA-256.
- Both variants change only NextItNetModel.encode_sequence. model.py adds the fixed 0.1 tanh signal from valid, strictly preceding final-block states before LayerNorm; control-1.py adds the same signal after LayerNorm.
- The preceding mean is zero without a valid predecessor. The existing context readout, forward path, candidate scoring, architecture, and training helpers are unchanged.

限制：

- Review used the source texts in context.json because /workspace/model was not supplied. Host CPU checks, including the proposal-specific fixed-state and gradient checks, are pending. Source approval does not establish runtime correctness, ranking benefit, or causal isolation.

本轮没有 role_rejected 事件。

## 实际调用与训练成本

| 调用目录 | 输入 token | 缓存输入 token | 输出 token | 用时秒 |
| --- | ---: | ---: | ---: | ---: |
| [analyst](../../records/research/rounds/round-17/analyst/request.json) | 324,361 | 246,656 | 6,106 | 77.7 |
| [experiment](../../records/research/rounds/round-17/experiment/request.json) | 287,387 | 256,896 | 2,226 | 49.9 |
| [researcher](../../records/research/rounds/round-17/researcher/request.json) | 520,549 | 447,744 | 5,556 | 89.3 |
| [reviewer](../../records/research/rounds/round-17/reviewer/request.json) | 217,201 | 186,112 | 2,627 | 63.6 |

缓存输入是输入 token 的子集；reasoning token 是输出的子集，不重复相加。这些计数不能直接换算 Pro 订阅额度。

| 训练任务 | 最佳 epoch | 完成 epoch | 分配 GPU 秒 |
| --- | ---: | ---: | ---: |
| [amazon-next-v9-round-17-control-1-CDs_and_Vinyl-42](../../records/study/jobs/amazon-next-v9-round-17-control-1-CDs_and_Vinyl-42/completed.json) | 90 | 110 | 636.2 |
| [amazon-next-v9-round-17-control-1-Electronics-42](../../records/study/jobs/amazon-next-v9-round-17-control-1-Electronics-42/completed.json) | 40 | 60 | 401.5 |
| [amazon-next-v9-round-17-control-1-Industrial_and_Scientific-42](../../records/study/jobs/amazon-next-v9-round-17-control-1-Industrial_and_Scientific-42/completed.json) | 35 | 55 | 353.4 |
| [amazon-next-v9-round-17-control-1-Movies_and_TV-42](../../records/study/jobs/amazon-next-v9-round-17-control-1-Movies_and_TV-42/completed.json) | 55 | 75 | 497.2 |
| [amazon-next-v9-round-17-model-CDs_and_Vinyl-42](../../records/study/jobs/amazon-next-v9-round-17-model-CDs_and_Vinyl-42/completed.json) | 85 | 105 | 724.6 |
| [amazon-next-v9-round-17-model-Electronics-42](../../records/study/jobs/amazon-next-v9-round-17-model-Electronics-42/completed.json) | 40 | 60 | 454.3 |
| [amazon-next-v9-round-17-model-Industrial_and_Scientific-42](../../records/study/jobs/amazon-next-v9-round-17-model-Industrial_and_Scientific-42/completed.json) | 90 | 110 | 744.4 |
| [amazon-next-v9-round-17-model-Movies_and_TV-42](../../records/study/jobs/amazon-next-v9-round-17-model-Movies_and_TV-42/completed.json) | 50 | 70 | 512.5 |

完整训练曲线在对应任务的 curve.jsonl；此表只计本轮新版本，复用基线不重复计费。

## 提案原文与源码差异

### 机制

Compare pre-normalization integration of preceding final-block states against both the audited incumbent and an equal-operation post-normalization control. Fixed-state CPU checks distinguish the computations; frozen ranking comparisons test usefulness without proving causal attribution.

### 预期观测

Provisional prediction: seed-42 unweighted four-category development mean HR@10 exceeds 0.60475, strictly more than 0.005 above the observed 0.59975 incumbent mean, and exceeds the generated control. Report each category effect and available history-length slices with counts; no slice benefit is established.

### 证伪条件

Reject implementation admission if the formulas, causality, masking, or scoring contracts fail. Reject development advancement if candidate mean HR@10 is at most 0.60475, does not exceed the generated control, or fails host eligibility. Use the frozen confirmation procedure only if development qualifies. An aggregate gain does not establish the proposed mechanism.

### 计划正确性检查

- Verify edits in both files are confined to NextItNetModel.encode_sequence and preserve unrelated code and signatures.
- With fixed states, verify the candidate and control match their exact pre-norm and post-norm formulas using the same preceding mean and coefficient.
- Check empty, singleton, dense, and right-aligned histories; future-token invariance; batch independence; finite outputs and gradients; and zero padding gradients before and after a synthetic update.
- Verify forward and score_candidates identities after the update, plus fresh category/seed initialization and frozen execution manifests.

- [候选相对基线的 diff](candidate.diff) · [候选完整源码](../../records/research/rounds/round-17/experiment/model/model.py)
- [对照相对基线的 diff](control.diff) · [对照完整源码](../../records/research/rounds/round-17/experiment/model/control-1.py)

## 原始证据导航

- [研究员完整提案](../../records/research/rounds/round-17/researcher/response.json)
- [研究员实际上下文](../../records/research/rounds/round-17/researcher/context.json)
- [实验员返回](../../records/research/rounds/round-17/experiment/response.json)
- [本轮全部归档](../../records/research/rounds/round-17)
- [拒绝事件摘录](rejections.json)
- [六项指标与机制比较](../../records/research/rounds/round-17/development-readout.json)
- [分析员完整返回与知识写入](../../records/research/rounds/round-17/analyst/response.json)
- [晋升记录](../../records/research/rounds/round-17/promotion.json)
