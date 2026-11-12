# 第 04 轮 残差逐通道缩放

[返回总览](../../README.md) · [机器可读汇总](../../rounds.json)

状态：**未晋升**。当轮基线：`amazon-next-v8-round-1-control-1`。

## 假设与实验设计

不同残差通道需要不同更新强度。

| 版本 | 改动 |
| --- | --- |
| 候选 | 每个 block 的每个通道学习缩放，初始为 1。 |
| 机制对照 | 每个 block 学习一个共享标量，初始为 1。 |

基线和机制对照用途不同：前者判断是否值得保留，后者帮助区分解释。所有已训练版本使用 seed 42；并不承诺新增结构与基线逐参数初始化完全一致。

## 实验结果

| 品类 | 基线 HR@10 | 候选 HR@10 | 对照 HR@10 | 候选减基线 | 候选减对照 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Movies_and_TV | 0.64800 | 0.65000 | 0.64700 | +0.00200 | +0.00300 |
| Industrial_and_Scientific | 0.56100 | 0.57100 | 0.57300 | +0.01000 | -0.00200 |
| Electronics | 0.48600 | 0.46800 | 0.46800 | -0.01800 | +0.00000 |
| CDs_and_Vinyl | 0.68400 | 0.66900 | 0.68200 | -0.01500 | -0.01300 |
| **四品类平均** | **0.59475** | **0.58950** | **0.59250** | **-0.00525** | **-0.00300** |

Host 晋升：`无`；选择策略 `baseline-single-seed-v1`；confirmation_executed = `False`。

## 中文归因与审阅关注点

候选不如基线和标量对照，未支持通道特异性带来额外价值。

分析员结论标签：`provisional`；建议：`discard`。这里的 keep 是保留开发集记录供 host 选择，不等于机制确认。

### 分析员原始理由

The audited seed-42 development mean HR@10 is 0.5895 for the channel-scale candidate, 0.59475 for the incumbent, and 0.5925 for the scalar-scale control. The candidate trails both, and the host readout marks it ineligible for confirmation. Candidate effects versus the incumbent are +0.002 for Movies_and_TV, +0.010 for Industrial_and_Scientific, -0.018 for Electronics, and -0.015 for CDs_and_Vinyl. Versus the scalar control they are +0.003, -0.002, 0, and -0.013. These single-seed development outcomes do not establish a causal mechanism; no confirmation result is supplied. Performance selection belongs to the host.

### 尚未排除的解释

- Seed or checkpoint variation, category differences, and extra trainable capacity remain competing explanations for the candidate-control contrast.
- Host admission and pre/post training probes support the audited execution within their stated scope. The supplied summaries do not explicitly establish the proposed fixed-weight three-arm equality, equal channel-to-scalar equality, isolated single-channel perturbation, or exact scoring identity after an update.
- Concrete tensor-path and finite sequence probes do not establish all-input behavior or causal isolation. Exploratory history-length slices are mixed, with small long-history groups.
- Next discriminating experiment: run fixed-weight equality and single-channel perturbation CPU diagnostics, then compare incumbent, scalar scale, channel scale, and a parameter-matched non-scaling per-channel control across preregistered paired independent seeds under the frozen protocol. Report category and aggregate HR@10 with uncertainty.

## 审核和返回修改


### reviewer

原始判定：`approve`。[完整记录](../../records/research/rounds/round-4/reviewer/response.json)

阻塞项：



已检查：

- The supplied baseline, candidate, and control source texts match their stated SHA-256 hashes. Reviewed implementation_hash 3c092a825dc1fdf07fdd5504c20522e62448d66944987372a16a79001e89329d.
- After removing each declared parameter addition and residual-scale multiplication, both changed sources are identical to the baseline. The candidate and control differ only in scale shape.
- model.py uses a learned [1,1,D] scale; control-1.py uses a learned [1,1,1] scalar. Both initialize to 1 and multiply only the existing branch output before residual addition, as proposed.
- The unchanged encoder, forward, and candidate-scoring paths retain the stated scoring identity in source.

限制：

- /workspace/model was absent, so this review used the source texts embedded in /workspace/context.json.
- CPU contract checks, including numerical equality, gradients, and behavior after an update, are pending host admission. Source approval does not establish runtime correctness or causal isolation.

本轮没有 role_rejected 事件。

## 实际调用与训练成本

| 调用目录 | 输入 token | 缓存输入 token | 输出 token | 用时秒 |
| --- | ---: | ---: | ---: | ---: |
| [analyst](../../records/research/rounds/round-4/analyst/request.json) | 432,206 | 376,576 | 6,938 | 161.2 |
| [experiment](../../records/research/rounds/round-4/experiment/request.json) | 235,150 | 209,536 | 1,817 | 59.7 |
| [researcher](../../records/research/rounds/round-4/researcher/request.json) | 485,327 | 422,912 | 3,941 | 75.2 |
| [reviewer](../../records/research/rounds/round-4/reviewer/request.json) | 257,338 | 219,264 | 2,908 | 59.2 |

缓存输入是输入 token 的子集；reasoning token 是输出的子集，不重复相加。这些计数不能直接换算 Pro 订阅额度。

| 训练任务 | 最佳 epoch | 完成 epoch | 分配 GPU 秒 |
| --- | ---: | ---: | ---: |
| [amazon-next-v9-round-4-control-1-CDs_and_Vinyl-42](../../records/study/jobs/amazon-next-v9-round-4-control-1-CDs_and_Vinyl-42/completed.json) | 85 | 105 | 572.6 |
| [amazon-next-v9-round-4-control-1-Electronics-42](../../records/study/jobs/amazon-next-v9-round-4-control-1-Electronics-42/completed.json) | 15 | 35 | 235.2 |
| [amazon-next-v9-round-4-control-1-Industrial_and_Scientific-42](../../records/study/jobs/amazon-next-v9-round-4-control-1-Industrial_and_Scientific-42/completed.json) | 60 | 80 | 523.9 |
| [amazon-next-v9-round-4-control-1-Movies_and_TV-42](../../records/study/jobs/amazon-next-v9-round-4-control-1-Movies_and_TV-42/completed.json) | 55 | 75 | 484.8 |
| [amazon-next-v9-round-4-model-CDs_and_Vinyl-42](../../records/study/jobs/amazon-next-v9-round-4-model-CDs_and_Vinyl-42/completed.json) | 45 | 65 | 442.3 |
| [amazon-next-v9-round-4-model-Electronics-42](../../records/study/jobs/amazon-next-v9-round-4-model-Electronics-42/completed.json) | 55 | 75 | 503.2 |
| [amazon-next-v9-round-4-model-Industrial_and_Scientific-42](../../records/study/jobs/amazon-next-v9-round-4-model-Industrial_and_Scientific-42/completed.json) | 70 | 90 | 592.6 |
| [amazon-next-v9-round-4-model-Movies_and_TV-42](../../records/study/jobs/amazon-next-v9-round-4-model-Movies_and_TV-42/completed.json) | 55 | 75 | 501.8 |

完整训练曲线在对应任务的 curve.jsonl；此表只计本轮新版本，复用基线不重复计费。

## 提案原文与源码差异

### 机制

Compare channel-specific residual scaling with the incumbent and a generated learned scalar-scale control. The candidate-control contrast probes whether channel specificity adds value beyond learning overall block strength; the incumbent contrast probes practical ranking benefit. No mechanism conclusion follows from the existing baseline alone.

### 预期观测

Hypothesis: seed-42 unweighted four-category mean HR@10 exceeds 0.60425, strictly more than 0.005 above the observed incumbent mean of 0.59925, and exceeds the scalar-scale control. Report all four category effects and available history-length slices, including regressions; no slice benefit is established in advance.

### 证伪条件

Reject implementation validity if CPU contracts fail. Reject the ranking prediction if candidate mean HR@10 is at most 0.60425 or at most the generated control mean. If candidate and scalar control improve similarly, channel specificity remains unsupported. A seed-42 pass remains provisional.

### 计划正确性检查

- Verify candidate and control edits are confined to the declared block methods.
- Verify initial candidate, control and incumbent block outputs agree on identical fixed weights and inputs.
- Verify equal channel scales reproduce the scalar control and a single-channel perturbation changes only that residual channel.
- Verify finite outputs and gradients, causal future-token invariance, batch independence, input immutability and zero padding gradients before and after a CPU parameter update.
- Verify forward uses encode_sequence and score_candidates equals item_emb(candidates) @ encode_sequence(seq)[0,-1] after the update.
- Audit independent initialization and identical frozen training and evaluation manifests.

- [候选相对基线的 diff](candidate.diff) · [候选完整源码](../../records/research/rounds/round-4/experiment/model/model.py)
- [对照相对基线的 diff](control.diff) · [对照完整源码](../../records/research/rounds/round-4/experiment/model/control-1.py)

## 原始证据导航

- [研究员完整提案](../../records/research/rounds/round-4/researcher/response.json)
- [研究员实际上下文](../../records/research/rounds/round-4/researcher/context.json)
- [实验员返回](../../records/research/rounds/round-4/experiment/response.json)
- [本轮全部归档](../../records/research/rounds/round-4)
- [拒绝事件摘录](rejections.json)
- [六项指标与机制比较](../../records/research/rounds/round-4/development-readout.json)
- [分析员完整返回与知识写入](../../records/research/rounds/round-4/analyst/response.json)
- [晋升记录](../../records/research/rounds/round-4/promotion.json)
