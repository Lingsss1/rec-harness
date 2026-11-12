# 第 08 轮 历史均值的注入深度

[返回总览](../../README.md) · [机器可读汇总](../../rounds.json)

状态：**晋升对照**。当轮基线：`amazon-next-v8-round-1-control-1`。

## 假设与实验设计

中层注入历史摘要后，剩余层的加工可提高效果。

| 版本 | 改动 |
| --- | --- |
| 候选 | 第 4 个 block 后注入 0.1 倍严格前缀均值。 |
| 机制对照 | 第 8 个 block 后注入同样公式的均值。 |

基线和机制对照用途不同：前者判断是否值得保留，后者帮助区分解释。所有已训练版本使用 seed 42；并不承诺新增结构与基线逐参数初始化完全一致。

## 实验结果

| 品类 | 基线 HR@10 | 候选 HR@10 | 对照 HR@10 | 候选减基线 | 候选减对照 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Movies_and_TV | 0.64800 | 0.65800 | 0.67400 | +0.01000 | -0.01600 |
| Industrial_and_Scientific | 0.56100 | 0.55100 | 0.56400 | -0.01000 | -0.01300 |
| Electronics | 0.48600 | 0.45800 | 0.46400 | -0.02800 | -0.00600 |
| CDs_and_Vinyl | 0.68400 | 0.67600 | 0.67900 | -0.00800 | -0.00300 |
| **四品类平均** | **0.59475** | **0.58575** | **0.59525** | **-0.00900** | **-0.00950** |

Host 晋升：`amazon-next-v9-round-8-control-1`；选择策略 `baseline-single-seed-v1`；confirmation_executed = `False`。

## 中文归因与审阅关注点

晋升的是末端对照，中层候选四品类均不如对照。不同深度的输入状态不同，不能将差异只归因于后续加工。

分析员结论标签：`provisional`；建议：`keep`。这里的 keep 是保留开发集记录供 host 选择，不等于机制确认。

### 分析员原始理由

Keep the audited development record because the host reports eligible_for_confirmation=true; its eligible_models list contains only the terminal control. This does not endorse the midpoint candidate. Seed-42 mean HR@10 was 0.58575 for the candidate, 0.59475 for the incumbent and 0.59525 for the terminal control. The candidate missed its ranking prediction and trailed the control in all four categories. The control's 0.0005 gain over the incumbent is near null. Performance selection belongs to the host; these development outcomes do not establish a causal mechanism or confirmation result.

### 尚未排除的解释

- Midpoint and terminal insertion use states from different encoder depths; their ranking contrast does not isolate downstream processing.
- The midpoint residual may change state scale, normalization or optimization, while the incumbent blocks may already encode sufficient history.
- One seed and public-validation checkpoint selection leave category variation and the near-null control gain uncertain.
- The supplied synthetic checks do not document the proposed explicit prefix-mean oracles, identity versus nonidentity downstream-block test, or post-update scoring identity. No /workspace/model directory was supplied for direct source inspection.
- Exploratory history slices include sparse or empty long-history bins and lack control-by-history contrasts. Repeated-seed estimates and confirmation outcomes are absent.
- The host reports control eligibility under a baseline-only development selection rule, while the supplied protocol lists a 0.005 minimum effect. This analyst does not resolve host performance selection.

## 审核和返回修改


### reviewer

原始判定：`approve`。[完整记录](../../records/research/rounds/round-8/reviewer/response.json)

阻塞项：



已检查：

- For implementation_hash 76b809b98135c8c518d2454364e075e8a34c2b802b9fb902bb2afd0229329f1f, independently recomputed SHA-256 hashes match the supplied baseline, candidate, and control source texts.
- Compared with the baseline, each arm changes only NextItNetModel.encode_sequence. The candidate inserts after block four and control-1.py after block eight under the frozen eight-block configuration; their only difference is that placement condition.
- Both insertions use a masked, shifted prefix sum and count, exclude the current and future positions, apply the fixed 0.1 scale once, and mask the result. Final normalization, readout, forward scoring, and candidate scoring are unchanged.

限制：

- No /workspace/model directory was supplied, so the review covers the source snapshots embedded in context.json; their correspondence to on-disk files could not be checked.
- The composite implementation_hash construction was not supplied. Individual source hashes were verified, but the composite hash could only be referenced as provided.
- CPU mechanism and scoring checks, host admission, and training are pending. Source approval does not establish runtime correctness, ranking benefit, or causal isolation. Unchanged helper defaults were not treated as evidence of a host protocol violation.

本轮没有 role_rejected 事件。

## 实际调用与训练成本

| 调用目录 | 输入 token | 缓存输入 token | 输出 token | 用时秒 |
| --- | ---: | ---: | ---: | ---: |
| [analyst](../../records/research/rounds/round-8/analyst/request.json) | 316,997 | 261,248 | 5,691 | 76.4 |
| [experiment](../../records/research/rounds/round-8/experiment/request.json) | 228,717 | 174,592 | 2,483 | 64.9 |
| [researcher](../../records/research/rounds/round-8/researcher/request.json) | 392,151 | 319,488 | 4,724 | 70.7 |
| [reviewer](../../records/research/rounds/round-8/reviewer/request.json) | 205,577 | 177,536 | 3,363 | 72.4 |

缓存输入是输入 token 的子集；reasoning token 是输出的子集，不重复相加。这些计数不能直接换算 Pro 订阅额度。

| 训练任务 | 最佳 epoch | 完成 epoch | 分配 GPU 秒 |
| --- | ---: | ---: | ---: |
| [amazon-next-v9-round-8-control-1-CDs_and_Vinyl-42](../../records/study/jobs/amazon-next-v9-round-8-control-1-CDs_and_Vinyl-42/completed.json) | 85 | 105 | 657.3 |
| [amazon-next-v9-round-8-control-1-Electronics-42](../../records/study/jobs/amazon-next-v9-round-8-control-1-Electronics-42/completed.json) | 15 | 35 | 227.9 |
| [amazon-next-v9-round-8-control-1-Industrial_and_Scientific-42](../../records/study/jobs/amazon-next-v9-round-8-control-1-Industrial_and_Scientific-42/completed.json) | 45 | 65 | 415.3 |
| [amazon-next-v9-round-8-control-1-Movies_and_TV-42](../../records/study/jobs/amazon-next-v9-round-8-control-1-Movies_and_TV-42/completed.json) | 115 | 135 | 826.4 |
| [amazon-next-v9-round-8-model-CDs_and_Vinyl-42](../../records/study/jobs/amazon-next-v9-round-8-model-CDs_and_Vinyl-42/completed.json) | 85 | 105 | 666.1 |
| [amazon-next-v9-round-8-model-Electronics-42](../../records/study/jobs/amazon-next-v9-round-8-model-Electronics-42/completed.json) | 15 | 35 | 311.5 |
| [amazon-next-v9-round-8-model-Industrial_and_Scientific-42](../../records/study/jobs/amazon-next-v9-round-8-model-Industrial_and_Scientific-42/completed.json) | 20 | 40 | 284.8 |
| [amazon-next-v9-round-8-model-Movies_and_TV-42](../../records/study/jobs/amazon-next-v9-round-8-model-Movies_and_TV-42/completed.json) | 65 | 85 | 613.5 |

完整训练曲线在对应任务的 curve.jsonl；此表只计本轮新版本，复用基线不重复计费。

## 提案原文与源码差异

### 机制

The incumbent has no added intermediate prefix residual. Compare it with midpoint injection and matched terminal injection. This is distinct from the failed local-difference projection recorded in amazon-research-a907a013823c2c31-8f30ce414f70fcc2-method and -proposal. The development and mechanism observation amazon-research-a907a013823c2c31-8f30ce414f70fcc2-mechanism-88bd6f35d2169b64 document that earlier failure without refuting all historical processing mechanisms.

### 预期观测

Provisional prediction: seed-42 unweighted four-category mean HR@10 exceeds 0.59975 and exceeds the generated terminal-injection control. Report all category effects and available retained-history slices with user counts. No category benefit or long-history advantage is established in advance; sparse slices remain descriptive.

### 证伪条件

Reject admission if insertion formulas, locations, causal masking or scoring contracts fail. Reject the ranking prediction if candidate mean HR@10 is at most 0.59975 or does not exceed the generated control. Follow host eligibility and any required frozen confirmation procedure. The prior graph's minimum_effect=0 selection metadata does not replace the current protocol's 0.005 requirement. Passing development remains provisional evidence.

### 计划正确性检查

- Verify changes are confined to encode_sequence in model.py and control-1.py; preserve signatures and unrelated code.
- Compare insertion outputs with explicit CPU prefix-mean oracles for empty, singleton, dense, right-aligned and internally padded histories.
- Instrument insertion locations and verify exactly one insertion per arm, after block four versus block eight under frozen configuration.
- Verify equality with identity downstream blocks and demonstrate placement-dependent outputs with fixed nonidentity blocks; these are computational checks only.
- Check causality, finite outputs and gradients, zero padding gradients, input immutability, batch independence and permutation equivariance before and after a synthetic update.
- Verify forward scoring identities and score_candidates = item_emb(candidates) @ encode_sequence(seq)[0,-1] after an update.
- Audit source identities, fresh initialization and frozen execution manifests.

- [候选相对基线的 diff](candidate.diff) · [候选完整源码](../../records/research/rounds/round-8/experiment/model/model.py)
- [对照相对基线的 diff](control.diff) · [对照完整源码](../../records/research/rounds/round-8/experiment/model/control-1.py)

## 原始证据导航

- [研究员完整提案](../../records/research/rounds/round-8/researcher/response.json)
- [研究员实际上下文](../../records/research/rounds/round-8/researcher/context.json)
- [实验员返回](../../records/research/rounds/round-8/experiment/response.json)
- [本轮全部归档](../../records/research/rounds/round-8)
- [拒绝事件摘录](rejections.json)
- [六项指标与机制比较](../../records/research/rounds/round-8/development-readout.json)
- [分析员完整返回与知识写入](../../records/research/rounds/round-8/analyst/response.json)
- [晋升记录](../../records/research/rounds/round-8/promotion.json)
