# 第 19 轮 中间层直接混入读出

[返回总览](../../README.md) · [机器可读汇总](../../rounds.json)

状态：**未晋升**。当轮基线：`amazon-next-v9-round-15-model`。

## 假设与实验设计

显式保留中间层状态可能改善特征保留。

| 版本 | 改动 |
| --- | --- |
| 候选 | 归一化后的最终状态混入 10% 的第 4 层状态。 |
| 机制对照 | 混入 10% 的第 6 层状态。 |

基线和机制对照用途不同：前者判断是否值得保留，后者帮助区分解释。所有已训练版本使用 seed 42；并不承诺新增结构与基线逐参数初始化完全一致。

## 实验结果

| 品类 | 基线 HR@10 | 候选 HR@10 | 对照 HR@10 | 候选减基线 | 候选减对照 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Movies_and_TV | 0.66500 | 0.64900 | 0.66700 | -0.01600 | -0.01800 |
| Industrial_and_Scientific | 0.57200 | 0.57100 | 0.57800 | -0.00100 | -0.00700 |
| Electronics | 0.47700 | 0.45400 | 0.45400 | -0.02300 | +0.00000 |
| CDs_and_Vinyl | 0.68500 | 0.68900 | 0.68600 | +0.00400 | +0.00300 |
| **四品类平均** | **0.59975** | **0.59075** | **0.59625** | **-0.00900** | **-0.00550** |

Host 晋升：`无`；选择策略 `baseline-single-seed-v1`；confirmation_executed = `False`。

## 中文归因与审阅关注点

两者均低于基线，第 4 层也不如第 6 层；不能支持中层信息补充。

分析员结论标签：`unknown`；建议：`discard`。这里的 keep 是保留开发集记录供 host 选择，不等于机制确认。

### 分析员原始理由

The audited seed-42 public development mean HR@10 was 0.59075 for the block-four candidate, 0.59625 for the block-six control, and 0.59975 for the incumbent. The candidate trailed the incumbent by 0.009 and the control by 0.0055; eligible_for_confirmation is false. The category contrasts include a tie with the control on Electronics and a small candidate advantage on CDs_and_Vinyl. Host admission and frozen-protocol execution were verified, but these outcomes do not establish whether an intermediate state preserves useful features. Performance selection belongs to the host; no confirmation result was supplied.

### 尚未排除的解释

- The fixed blend may affect optimization or regularization independently of retained depth; block-four and block-six activations and gradients may also differ.
- The single seed and public-validation checkpoint selection leave run variability unresolved. Category and history-length slices are dependent parts of the same development evaluation; sparse long-history bins cannot establish a history-specific effect.
- The reported fixed-block-output perturbation, exact depth-sensitivity, and post-update forward and score_candidates identity checks are not established by the supplied host evidence. Synthetic contracts and executed tensor paths do not prove all-input semantics or causal isolation.
- A discriminating follow-up would hold block outputs fixed, perturb masked blocks four and six separately, and check each arm's readout response and scoring identity. A separately preregistered matched-depth experiment across independent seeds could then distinguish depth effects from activation, gradient, and checkpoint variation.

## 审核和返回修改


### reviewer

原始判定：`approve`。[完整记录](../../records/research/rounds/round-19/reviewer/response.json)

阻塞项：



已检查：

- For implementation_hash 844ba54b7705c390f4c4c9e5da526bb1b70c58ac828fd36a7ab2910a1940f063, independently computed SHA-256 hashes match the supplied baseline.py, model.py, and control-1.py source hashes.
- Both edits are confined to NextItNetModel.encode_sequence. Signatures, other methods, and unrelated source are unchanged; the two arms differ only in the retained-index expression.
- At eight blocks, model.py retains masked block four and control-1.py retains masked block six. Both apply 0.9 times the normalized final state plus 0.1 times the retained state normalized by the existing LayerNorm, with no added parameters.
- The indices coincide at one and two blocks and are 1 and 2 respectively at four blocks, matching the frozen proposal. Both arms keep the existing context, compression, output mask, forward path, and score_candidates identity.

限制：

- The model directory was not supplied; this review used the source texts embedded in context.json.
- Host CPU contracts, including fixed-output perturbation tests, and frozen-protocol execution remain pending. Source approval does not establish runtime correctness, ranking benefit, or causal isolation.

本轮没有 role_rejected 事件。

## 实际调用与训练成本

| 调用目录 | 输入 token | 缓存输入 token | 输出 token | 用时秒 |
| --- | ---: | ---: | ---: | ---: |
| [analyst](../../records/research/rounds/round-19/analyst/request.json) | 273,978 | 227,328 | 3,607 | 64.8 |
| [experiment](../../records/research/rounds/round-19/experiment/request.json) | 253,246 | 221,184 | 2,001 | 49.2 |
| [researcher](../../records/research/rounds/round-19/researcher/request.json) | 559,169 | 494,464 | 4,580 | 80.7 |
| [reviewer](../../records/research/rounds/round-19/reviewer/request.json) | 227,840 | 198,144 | 3,523 | 58.0 |

缓存输入是输入 token 的子集；reasoning token 是输出的子集，不重复相加。这些计数不能直接换算 Pro 订阅额度。

| 训练任务 | 最佳 epoch | 完成 epoch | 分配 GPU 秒 |
| --- | ---: | ---: | ---: |
| [amazon-next-v9-round-19-control-1-CDs_and_Vinyl-42](../../records/study/jobs/amazon-next-v9-round-19-control-1-CDs_and_Vinyl-42/completed.json) | 95 | 115 | 770.7 |
| [amazon-next-v9-round-19-control-1-Electronics-42](../../records/study/jobs/amazon-next-v9-round-19-control-1-Electronics-42/completed.json) | 5 | 25 | 144.5 |
| [amazon-next-v9-round-19-control-1-Industrial_and_Scientific-42](../../records/study/jobs/amazon-next-v9-round-19-control-1-Industrial_and_Scientific-42/completed.json) | 75 | 95 | 636.5 |
| [amazon-next-v9-round-19-control-1-Movies_and_TV-42](../../records/study/jobs/amazon-next-v9-round-19-control-1-Movies_and_TV-42/completed.json) | 110 | 130 | 773.4 |
| [amazon-next-v9-round-19-model-CDs_and_Vinyl-42](../../records/study/jobs/amazon-next-v9-round-19-model-CDs_and_Vinyl-42/completed.json) | 95 | 115 | 758.2 |
| [amazon-next-v9-round-19-model-Electronics-42](../../records/study/jobs/amazon-next-v9-round-19-model-Electronics-42/completed.json) | 10 | 30 | 223.2 |
| [amazon-next-v9-round-19-model-Industrial_and_Scientific-42](../../records/study/jobs/amazon-next-v9-round-19-model-Industrial_and_Scientific-42/completed.json) | 75 | 95 | 568.7 |
| [amazon-next-v9-round-19-model-Movies_and_TV-42](../../records/study/jobs/amazon-next-v9-round-19-model-Movies_and_TV-42/completed.json) | 65 | 85 | 462.9 |

完整训练曲线在对应任务的 curve.jsonl；此表只计本轮新版本，复用基线不重复计费。

## 提案原文与源码差异

### 机制

Compare a block-four readout with a matched block-six readout and the audited incumbent. CPU checks test depth-specific computation; frozen ranking comparisons test usefulness without establishing causal attribution.

### 预期观测

Provisional prediction: candidate seed-42 unweighted four-category mean HR@10 exceeds 0.60475, strictly more than 0.005 above the observed incumbent mean of 0.59975, and exceeds the generated control. Report all four category effects and any available history-length slices with counts. No slice improvement is established in advance.

### 证伪条件

Reject implementation admission if retained-depth selection, causality, masking, finiteness, or scoring contracts fail. Reject development advancement if candidate mean HR@10 is at most 0.60475, does not exceed the generated control, or fails host eligibility. Apply the frozen confirmation procedure only if development qualifies. An aggregate gain does not prove intermediate-feature preservation.

### 计划正确性检查

- Verify edits in model.py and control-1.py are confined to NextItNetModel.encode_sequence; preserve signatures, imports, helpers, and unrelated code.
- At eight blocks, verify candidate and control retain masked blocks four and six respectively, with identical coefficients, parameter counts, and final-state normalization.
- At one and two blocks, verify retained indices coincide; at four blocks, verify candidate index 1 and control index 2.
- With fixed block outputs, verify both formulas and each arm's sensitivity to its specified retained depth.
- Check empty, singleton, dense, and right-aligned histories; future-token invariance; finite outputs and gradients; zero padding gradients; and batch independence before and after a synthetic update.
- Verify forward and score_candidates identities after an update, independent initialization per category and seed, and frozen execution manifests.

- [候选相对基线的 diff](candidate.diff) · [候选完整源码](../../records/research/rounds/round-19/experiment/model/model.py)
- [对照相对基线的 diff](control.diff) · [对照完整源码](../../records/research/rounds/round-19/experiment/model/control-1.py)

## 原始证据导航

- [研究员完整提案](../../records/research/rounds/round-19/researcher/response.json)
- [研究员实际上下文](../../records/research/rounds/round-19/researcher/context.json)
- [实验员返回](../../records/research/rounds/round-19/experiment/response.json)
- [本轮全部归档](../../records/research/rounds/round-19)
- [拒绝事件摘录](rejections.json)
- [六项指标与机制比较](../../records/research/rounds/round-19/development-readout.json)
- [分析员完整返回与知识写入](../../records/research/rounds/round-19/analyst/response.json)
- [晋升记录](../../records/research/rounds/round-19/promotion.json)
