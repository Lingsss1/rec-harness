# 第 03 轮 历史长度决定混合权重

[返回总览](../../README.md) · [机器可读汇总](../../rounds.json)

状态：**未晋升**。当轮基线：`amazon-next-v8-round-1-control-1`。

## 假设与实验设计

历史越长，上下文估计越可靠，应获得更高权重。

| 版本 | 改动 |
| --- | --- |
| 候选 | 历史权重 0.15*n/(n+4)。 |
| 机制对照 | 历史权重 0.15*4/(n+4)，随长度下降。 |

基线和机制对照用途不同：前者判断是否值得保留，后者帮助区分解释。所有已训练版本使用 seed 42；并不承诺新增结构与基线逐参数初始化完全一致。

## 实验结果

| 品类 | 基线 HR@10 | 候选 HR@10 | 对照 HR@10 | 候选减基线 | 候选减对照 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Movies_and_TV | 0.64800 | 0.64600 | 0.63000 | -0.00200 | +0.01600 |
| Industrial_and_Scientific | 0.56100 | 0.55000 | 0.56900 | -0.01100 | -0.01900 |
| Electronics | 0.48600 | 0.45600 | 0.45200 | -0.03000 | +0.00400 |
| CDs_and_Vinyl | 0.68400 | 0.66600 | 0.64900 | -0.01800 | +0.01700 |
| **四品类平均** | **0.59475** | **0.57950** | **0.57500** | **-0.01525** | **+0.00450** |

Host 晋升：`无`；选择策略 `baseline-single-seed-v1`；confirmation_executed = `False`。

## 中文归因与审阅关注点

候选四品类均低于基线，预期短历史收益未出现；方向与平均权重变化没有完全隔离。

分析员结论标签：`provisional`；建议：`discard`。这里的 keep 是保留开发集记录供 host 选择，不等于机制确认。

### 分析员原始理由

Audited seed-42 development mean HR@10 was 0.5795 for the candidate, 0.575 for the generated control, and 0.59475 for the incumbent. Both new arms fell below the incumbent and were ineligible for confirmation. The candidate exceeded the control by 0.0045, but its result was below the proposal’s 0.59975 threshold and below the incumbent in every category. The predicted gain for users with 1–5 training interactions was negative in all four categories. The development comparison does not establish a causal mechanism; no confirmation result was supplied.

### 尚未排除的解释

- The candidate–control contrast was mixed by category, including a 0.019 HR@10 loss for the candidate in Industrial_and_Scientific. Count direction, overall context attenuation, and category effects remain competing explanations.
- Independent training trajectories and public-validation checkpoint selection may explain the small candidate–control mean difference.
- Several history slices had null effects; sparse long-history bins and the absence of generated-control history-bin results prevent a discriminating history interaction test.
- The author could not run CPU checks locally. Host admission and pre/post training synthetic contracts report success, but those finite checks do not prove the proposed mechanism or all-input behavior. Direct numerical equivalence of forward and candidate-scoring states before and after training was not supplied.
- No independent-seed replication, held-out confirmation, or average-context-matched fixed-weight control was supplied.

## 审核和返回修改


### reviewer

原始判定：`approve`。[完整记录](../../records/research/rounds/round-3/reviewer/response.json)

阻塞项：



已检查：

- Reviewed implementation_hash 4c2c05cf34d6564e271397aba9d946312ec94cc4f7160a54351c57b478ec9457. Recomputed SHA-256 for all three supplied source texts; each matches its declared source hash.
- Compared complete source texts. model.py and control-1.py differ from baseline.py only in the blend inside NextItNetModel.encode_sequence, as permitted by the insertion contract.
- The candidate uses 0.15*n/(n+4); the control uses 0.15*4/(n+4). Their weights have the specified opposite trends and both equal 0.075 at n=4.
- Both arms retain the strictly preceding count and mean, compression, zero-predecessor fallback, validity mask, and causal encoder. forward and score_candidates use encode_sequence; candidate scoring reads the right-aligned tail.

限制：

- The model/ directory was not supplied; review used the source texts in context.json.
- Host CPU checks and training results are pending. Source review does not establish runtime correctness, ranking benefit, or causal isolation.
- No canonical construction for the aggregate implementation_hash was supplied, so only the three individual source hashes were independently recomputed.

本轮没有 role_rejected 事件。

## 实际调用与训练成本

| 调用目录 | 输入 token | 缓存输入 token | 输出 token | 用时秒 |
| --- | ---: | ---: | ---: | ---: |
| [analyst](../../records/research/rounds/round-3/analyst/request.json) | 380,835 | 326,400 | 6,104 | 85.5 |
| [experiment](../../records/research/rounds/round-3/experiment/request.json) | 231,929 | 205,824 | 1,617 | 48.4 |
| [researcher](../../records/research/rounds/round-3/researcher/request.json) | 557,673 | 496,384 | 4,990 | 101.7 |
| [reviewer](../../records/research/rounds/round-3/reviewer/request.json) | 303,471 | 272,128 | 4,761 | 82.9 |

缓存输入是输入 token 的子集；reasoning token 是输出的子集，不重复相加。这些计数不能直接换算 Pro 订阅额度。

| 训练任务 | 最佳 epoch | 完成 epoch | 分配 GPU 秒 |
| --- | ---: | ---: | ---: |
| [amazon-next-v9-round-3-control-1-CDs_and_Vinyl-42](../../records/study/jobs/amazon-next-v9-round-3-control-1-CDs_and_Vinyl-42/completed.json) | 45 | 65 | 387.9 |
| [amazon-next-v9-round-3-control-1-Electronics-42](../../records/study/jobs/amazon-next-v9-round-3-control-1-Electronics-42/completed.json) | 35 | 55 | 356.1 |
| [amazon-next-v9-round-3-control-1-Industrial_and_Scientific-42](../../records/study/jobs/amazon-next-v9-round-3-control-1-Industrial_and_Scientific-42/completed.json) | 70 | 90 | 588.9 |
| [amazon-next-v9-round-3-control-1-Movies_and_TV-42](../../records/study/jobs/amazon-next-v9-round-3-control-1-Movies_and_TV-42/completed.json) | 40 | 60 | 350.9 |
| [amazon-next-v9-round-3-model-CDs_and_Vinyl-42](../../records/study/jobs/amazon-next-v9-round-3-model-CDs_and_Vinyl-42/completed.json) | 45 | 65 | 460.4 |
| [amazon-next-v9-round-3-model-Electronics-42](../../records/study/jobs/amazon-next-v9-round-3-model-Electronics-42/completed.json) | 35 | 55 | 376.6 |
| [amazon-next-v9-round-3-model-Industrial_and_Scientific-42](../../records/study/jobs/amazon-next-v9-round-3-model-Industrial_and_Scientific-42/completed.json) | 35 | 55 | 338.0 |
| [amazon-next-v9-round-3-model-Movies_and_TV-42](../../records/study/jobs/amazon-next-v9-round-3-model-Movies_and_TV-42/completed.json) | 65 | 85 | 583.5 |

完整训练曲线在对应任务的 curve.jsonl；此表只计本轮新版本，复用基线不重复计费。

## 提案原文与源码差异

### 机制

Compare the incumbent fixed 0.15 context weight with opposed, parameter-free predecessor-count weights. The candidate gives more context weight as evidence accumulates; the generated control gives less. Performance and slice contrasts test this provisional explanation.

### 预期观测

Provisional prediction: candidate mean HR@10 exceeds 0.59975 and the generated control. The clearest predicted gain is among users with 1–5 training interactions; report all category and history-length slices, including regressions. These effects have not been measured.

### 证伪条件

Reject the candidate if a CPU contract fails, its seed-42 four-category mean HR@10 is at most 0.59975, or it does not exceed the generated control's mean. If both changed arms improve similarly, the count-direction mechanism remains unsupported even if the candidate clears the incumbent.

### 计划正确性检查

- CPU synthetic count sweep verifies the two opposed formulas and equality at n=4.
- Verify finite empty-history output, zero padding output and gradient, causal input gradients, and right-aligned tail readout.
- Verify forward and candidate scoring use the same encoded states before and after training.
- Check the host receipts for seed 42, all four categories, independent initialization, and the frozen protocol hash.

- [候选相对基线的 diff](candidate.diff) · [候选完整源码](../../records/research/rounds/round-3/experiment/model/model.py)
- [对照相对基线的 diff](control.diff) · [对照完整源码](../../records/research/rounds/round-3/experiment/model/control-1.py)

## 原始证据导航

- [研究员完整提案](../../records/research/rounds/round-3/researcher/response.json)
- [研究员实际上下文](../../records/research/rounds/round-3/researcher/context.json)
- [实验员返回](../../records/research/rounds/round-3/experiment/response.json)
- [本轮全部归档](../../records/research/rounds/round-3)
- [拒绝事件摘录](rejections.json)
- [六项指标与机制比较](../../records/research/rounds/round-3/development-readout.json)
- [分析员完整返回与知识写入](../../records/research/rounds/round-3/analyst/response.json)
- [晋升记录](../../records/research/rounds/round-3/promotion.json)
