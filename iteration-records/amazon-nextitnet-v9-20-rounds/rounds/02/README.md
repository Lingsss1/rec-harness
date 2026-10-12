# 第 02 轮 逐特征压缩强度

[返回总览](../../README.md) · [机器可读汇总](../../rounds.json)

状态：**未晋升**。当轮基线：`amazon-next-v8-round-1-control-1`。

## 假设与实验设计

固定压缩可能无法适应各特征的动态范围。

| 版本 | 改动 |
| --- | --- |
| 候选 | 每个特征学习 s，使用 s*tanh(u/s)，初始 s=4。 |
| 机制对照 | 参数向量取均值，得到共享压缩强度。 |

基线和机制对照用途不同：前者判断是否值得保留，后者帮助区分解释。所有已训练版本使用 seed 42；并不承诺新增结构与基线逐参数初始化完全一致。

## 实验结果

| 品类 | 基线 HR@10 | 候选 HR@10 | 对照 HR@10 | 候选减基线 | 候选减对照 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Movies_and_TV | 0.64800 | 0.63700 | 0.64400 | -0.01100 | -0.00700 |
| Industrial_and_Scientific | 0.56100 | 0.57100 | 0.56800 | +0.01000 | +0.00300 |
| Electronics | 0.48600 | 0.45300 | 0.45100 | -0.03300 | +0.00200 |
| CDs_and_Vinyl | 0.68400 | 0.67400 | 0.66000 | -0.01000 | +0.01400 |
| **四品类平均** | **0.59475** | **0.58375** | **0.58075** | **-0.01100** | **+0.00300** |

Host 晋升：`无`；选择策略 `baseline-single-seed-v1`；confirmation_executed = `False`。

## 中文归因与审阅关注点

候选略优于共享尺度，但两者均退步；没有支持替换原压缩函数的证据。

分析员结论标签：`unknown`；建议：`discard`。这里的 keep 是保留开发集记录供 host 选择，不等于机制确认。

### 分析员原始理由

In the audited seed-42 development readout, mean HR@10 was 0.58375 for the candidate, 0.59475 for the incumbent, and 0.58075 for the scalar control. The candidate missed the preregistered 0.60425 threshold, and eligible_for_confirmation is false. Its 0.003 advantage over the scalar control was small and mixed across categories. Performance selection belongs to the host; these results do not establish the proposed mechanism.

### 尚未排除的解释

- The scalar control also trailed the incumbent by 0.014, so neither learned-scale arm showed an aggregate benefit over the incumbent.
- Different gradients, independent training trajectories, checkpoint selection, and category variation could explain the small candidate-control difference.
- Host synthetic checks do not document the proposal-specific fixed-state unequal-coordinate contrast or every proposed score and updated-parameter diagnostic.
- No independent multi-seed replication or confirmation result was supplied.

## 审核和返回修改


### reviewer

原始判定：`approve`。[完整记录](../../records/research/rounds/round-2/reviewer/response.json)

阻塞项：



已检查：

- Reviewed the supplied baseline, candidate, and control source texts for implementation_hash 2798733c9609c4266ca86f94449d551aaec962132a510abfcbcf3d3a77afa03d. Both changes are confined to the declared constructor and encoder methods; the training helpers and scoring methods are unchanged.
- Both arms initialize a hidden-size parameter vector to zero, producing scale 4. The candidate applies a separate bounded scale per feature at both compression sites. The control reduces its vector to a mean before producing one shared scale.
- The preceding-context calculation, 0.85/0.15 blend, predecessor-free fallback, and padding mask are preserved. Forward and candidate scoring still consume encode_sequence, with candidate scores computed from the tail representation and item embeddings.

限制：

- The source texts were supplied in context.json; no /workspace/model directory was present. The declared source hashes were not independently recomputed.
- Host CPU admission and proposal-specific numerical checks remain pending. This source approval does not establish runtime correctness, ranking gains, or causal isolation.

本轮没有 role_rejected 事件。

## 实际调用与训练成本

| 调用目录 | 输入 token | 缓存输入 token | 输出 token | 用时秒 |
| --- | ---: | ---: | ---: | ---: |
| [analyst](../../records/research/rounds/round-2/analyst/request.json) | 471,779 | 390,784 | 4,903 | 93.8 |
| [experiment](../../records/research/rounds/round-2/experiment/request.json) | 243,728 | 216,832 | 1,890 | 70.6 |
| [researcher](../../records/research/rounds/round-2/researcher/request.json) | 524,139 | 457,088 | 4,490 | 104.4 |
| [reviewer](../../records/research/rounds/round-2/reviewer/request.json) | 304,524 | 271,616 | 6,139 | 142.9 |

缓存输入是输入 token 的子集；reasoning token 是输出的子集，不重复相加。这些计数不能直接换算 Pro 订阅额度。

| 训练任务 | 最佳 epoch | 完成 epoch | 分配 GPU 秒 |
| --- | ---: | ---: | ---: |
| [amazon-next-v9-round-2-control-1-CDs_and_Vinyl-42](../../records/study/jobs/amazon-next-v9-round-2-control-1-CDs_and_Vinyl-42/completed.json) | 60 | 80 | 472.5 |
| [amazon-next-v9-round-2-control-1-Electronics-42](../../records/study/jobs/amazon-next-v9-round-2-control-1-Electronics-42/completed.json) | 20 | 40 | 228.0 |
| [amazon-next-v9-round-2-control-1-Industrial_and_Scientific-42](../../records/study/jobs/amazon-next-v9-round-2-control-1-Industrial_and_Scientific-42/completed.json) | 45 | 65 | 474.1 |
| [amazon-next-v9-round-2-control-1-Movies_and_TV-42](../../records/study/jobs/amazon-next-v9-round-2-control-1-Movies_and_TV-42/completed.json) | 50 | 70 | 475.0 |
| [amazon-next-v9-round-2-model-CDs_and_Vinyl-42](../../records/study/jobs/amazon-next-v9-round-2-model-CDs_and_Vinyl-42/completed.json) | 85 | 105 | 739.3 |
| [amazon-next-v9-round-2-model-Electronics-42](../../records/study/jobs/amazon-next-v9-round-2-model-Electronics-42/completed.json) | 5 | 25 | 212.1 |
| [amazon-next-v9-round-2-model-Industrial_and_Scientific-42](../../records/study/jobs/amazon-next-v9-round-2-model-Industrial_and_Scientific-42/completed.json) | 45 | 65 | 443.9 |
| [amazon-next-v9-round-2-model-Movies_and_TV-42](../../records/study/jobs/amazon-next-v9-round-2-model-Movies_and_TV-42/completed.json) | 60 | 80 | 563.8 |

完整训练曲线在对应任务的 curve.jsonl；此表只计本轮新版本，复用基线不重复计费。

## 提案原文与源码差异

### 机制

The incumbent comparison tests practical value of learnable featurewise compression. The generated scalar-scale control tests whether feature specificity adds value beyond learning a shared compression strength. Fixed-state checks establish the intended algebraic contrast; independently trained ranking comparisons remain provisional.

### 预期观测

Provisional prediction: candidate four-category mean HR@10 exceeds the incumbent mean 0.59925 by more than the frozen 0.005 minimum and exceeds the scalar-scale control. Report all four category effects; no category-specific gain is established in the supplied evidence.

### 证伪条件

Reject the ranking prediction if CPU contracts fail, candidate seed-42 four-category mean HR@10 is at most 0.60425, or candidate mean does not exceed the generated control. A pass remains development evidence, not proof that feature-specific compression caused the gain.

### 计划正确性检查

- At initialization, compare both arms numerically with the incumbent on dense, right-aligned, singleton, and empty histories.
- At fixed states, perturb two scale coordinates unequally and verify feature-specific candidate behavior versus shared-scale control behavior.
- Check causal input gradients, zero padding gradients and outputs, finite outputs and gradients after parameter updates, and tail readout.
- Check forward score identities and the exact score_candidates dot-product identity after learning.

- [候选相对基线的 diff](candidate.diff) · [候选完整源码](../../records/research/rounds/round-2/experiment/model/model.py)
- [对照相对基线的 diff](control.diff) · [对照完整源码](../../records/research/rounds/round-2/experiment/model/control-1.py)

## 原始证据导航

- [研究员完整提案](../../records/research/rounds/round-2/researcher/response.json)
- [研究员实际上下文](../../records/research/rounds/round-2/researcher/context.json)
- [实验员返回](../../records/research/rounds/round-2/experiment/response.json)
- [本轮全部归档](../../records/research/rounds/round-2)
- [拒绝事件摘录](rejections.json)
- [六项指标与机制比较](../../records/research/rounds/round-2/development-readout.json)
- [分析员完整返回与知识写入](../../records/research/rounds/round-2/analyst/response.json)
- [晋升记录](../../records/research/rounds/round-2/promotion.json)
