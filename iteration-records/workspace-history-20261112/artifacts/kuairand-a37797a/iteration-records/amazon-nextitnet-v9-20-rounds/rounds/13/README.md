# 第 13 轮 残差输入条件门控

[返回总览](../../README.md) · [机器可读汇总](../../rounds.json)

状态：**未晋升**。当轮基线：`amazon-next-v9-round-8-control-1`。

## 假设与实验设计

用 block 输入决定更新强度优于用更新量自身决定。

| 版本 | 改动 |
| --- | --- |
| 候选 | 增量 0.1*tanh(g*tanh(x))*y。 |
| 机制对照 | 将公式中的 x 替换为 y。 |

基线和机制对照用途不同：前者判断是否值得保留，后者帮助区分解释。所有已训练版本使用 seed 42；并不承诺新增结构与基线逐参数初始化完全一致。

## 实验结果

| 品类 | 基线 HR@10 | 候选 HR@10 | 对照 HR@10 | 候选减基线 | 候选减对照 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Movies_and_TV | 0.67400 | 0.65300 | 0.64400 | -0.02100 | +0.00900 |
| Industrial_and_Scientific | 0.56400 | 0.56400 | 0.55800 | +0.00000 | +0.00600 |
| Electronics | 0.46400 | 0.46600 | 0.46500 | +0.00200 | +0.00100 |
| CDs_and_Vinyl | 0.67900 | 0.68000 | 0.67900 | +0.00100 | +0.00100 |
| **四品类平均** | **0.59525** | **0.59075** | **0.58650** | **-0.00450** | **+0.00425** |

Host 晋升：`无`；选择策略 `baseline-single-seed-v1`；confirmation_executed = `False`。

## 中文归因与审阅关注点

输入门控优于对照但低于基线，不能证明有实际净收益。

分析员结论标签：`unknown`；建议：`discard`。这里的 keep 是保留开发集记录供 host 选择，不等于机制确认。

### 分析员原始理由

The host-audited seed-42 development mean HR@10 was 0.59075 for the candidate, 0.59525 for the incumbent, and 0.5865 for the generated control. The candidate exceeded the control by 0.00425 but fell below the incumbent by 0.0045. The host marked it ineligible for confirmation. The positive control contrast does not establish that input conditioning caused a benefit.

### 尚未排除的解释

- The candidate-control contrast may reflect different input and update tensor distributions, optimization, or checkpoint selection rather than useful input conditioning.
- The single seed and dependent category slices do not establish a reproducible effect. Movies_and_TV fell 0.021 below the incumbent; Industrial_and_Scientific was unchanged.
- Host synthetic admission and training checks do not establish the proposed mechanism. The supplied evidence does not document the proposed fixed-tensor nonzero-weight formula contrast or every post-update scoring identity.
- No confirmation result was supplied. A discriminating follow-up would verify both formulas on identical fixed tensors at zero and matched nonzero gate weights, then compare independently initialized fits under the frozen development protocol.

## 审核和返回修改


### reviewer

原始判定：`approve`。[完整记录](../../records/research/rounds/round-13/reviewer/response.json)

阻塞项：



已检查：

- Reviewed implementation_hash 33dc02de602852d8f8151a778d978fec86b69fa652a4bef9af99bd01122bdcbb. Independently computed SHA-256 for all three embedded source texts; each matches its supplied source hash.
- Both new arms make exactly the edits allowed by the insertion contract: a zero-initialized hidden-width gate in NextItNetBlock.__init__ and a gated correction in NextItNetBlock.forward. All other source text matches the baseline.
- The candidate uses 0.1 × tanh(g × tanh(x)) × y; control-1.py uses 0.1 × tanh(g × tanh(y)) × y. Both reduce algebraically to x + y when g is zero.
- The unchanged forward and score_candidates paths both use encode_sequence; candidate scoring remains the item embedding dot product with its final encoded position. The existing causal convolutions and post-block padding masks are preserved.

限制：

- /workspace/model was absent, so this review covers the source texts embedded in context.json, not a separate working tree.
- Fixed-tensor, gradient, and host CPU admission checks are pending. Source approval does not establish runtime correctness, causal isolation, or ranking gains.

本轮没有 role_rejected 事件。

## 实际调用与训练成本

| 调用目录 | 输入 token | 缓存输入 token | 输出 token | 用时秒 |
| --- | ---: | ---: | ---: | ---: |
| [analyst](../../records/research/rounds/round-13/analyst/request.json) | 416,859 | 361,344 | 4,041 | 53.2 |
| [experiment](../../records/research/rounds/round-13/experiment/request.json) | 224,986 | 197,248 | 1,898 | 43.5 |
| [researcher](../../records/research/rounds/round-13/researcher/request.json) | 621,085 | 555,136 | 5,548 | 73.9 |
| [reviewer](../../records/research/rounds/round-13/reviewer/request.json) | 264,634 | 233,728 | 3,339 | 56.8 |

缓存输入是输入 token 的子集；reasoning token 是输出的子集，不重复相加。这些计数不能直接换算 Pro 订阅额度。

| 训练任务 | 最佳 epoch | 完成 epoch | 分配 GPU 秒 |
| --- | ---: | ---: | ---: |
| [amazon-next-v9-round-13-control-1-CDs_and_Vinyl-42](../../records/study/jobs/amazon-next-v9-round-13-control-1-CDs_and_Vinyl-42/completed.json) | 75 | 95 | 699.8 |
| [amazon-next-v9-round-13-control-1-Electronics-42](../../records/study/jobs/amazon-next-v9-round-13-control-1-Electronics-42/completed.json) | 15 | 35 | 252.6 |
| [amazon-next-v9-round-13-control-1-Industrial_and_Scientific-42](../../records/study/jobs/amazon-next-v9-round-13-control-1-Industrial_and_Scientific-42/completed.json) | 45 | 65 | 463.0 |
| [amazon-next-v9-round-13-control-1-Movies_and_TV-42](../../records/study/jobs/amazon-next-v9-round-13-control-1-Movies_and_TV-42/completed.json) | 60 | 80 | 518.6 |
| [amazon-next-v9-round-13-model-CDs_and_Vinyl-42](../../records/study/jobs/amazon-next-v9-round-13-model-CDs_and_Vinyl-42/completed.json) | 95 | 115 | 708.3 |
| [amazon-next-v9-round-13-model-Electronics-42](../../records/study/jobs/amazon-next-v9-round-13-model-Electronics-42/completed.json) | 15 | 35 | 259.0 |
| [amazon-next-v9-round-13-model-Industrial_and_Scientific-42](../../records/study/jobs/amazon-next-v9-round-13-model-Industrial_and_Scientific-42/completed.json) | 55 | 75 | 516.0 |
| [amazon-next-v9-round-13-model-Movies_and_TV-42](../../records/study/jobs/amazon-next-v9-round-13-model-Movies_and_TV-42/completed.json) | 85 | 105 | 715.0 |

完整训练曲线在对应任务的 curve.jsonl；此表只计本轮新版本，复用基线不重复计费。

## 提案原文与源码差异

### 机制

Compare input-conditioned scaling of causal convolution updates with both the audited incumbent and a matched update-conditioned gate. Fixed-tensor CPU checks test the intended architectural distinction; frozen ranking comparisons test its practical effect. The graph's prior item-identity and local-change counterevidence motivates a distinct block-level test, while leaving causal attribution provisional.

### 预期观测

The falsifiable development prediction is seed-42 unweighted four-category mean HR@10 strictly above 0.60025, exceeding the observed incumbent mean 0.59525 by more than 0.005, and strictly above the generated control. Report each category effect and available history-length slices with counts; no slice benefit is established in advance.

### 证伪条件

Reject implementation admission if formulas, causality, masking, or scoring contracts fail. Reject development advancement if candidate mean HR@10 is at most 0.60025, does not exceed the generated control, or fails host eligibility. If it qualifies, apply the frozen confirmation procedure. A ranking gain alone does not establish the proposed mechanism.

### 计划正确性检查

- Confine edits to the declared methods in model.py and control-1.py; preserve imports, helpers, signatures, and module settings.
- Verify both arms reproduce incumbent outputs at zero gate weights and compute their distinct formulas with identical nonzero weights on fixed x and y.
- Check empty, singleton, dense, and right-aligned histories; finite outputs and gradients; future-token perturbation invariance; zero padding gradients; and batch independence.
- After a synthetic parameter update, verify the forward and score_candidates identities against encode_sequence.
- Audit independent per-category initialization and the frozen execution manifest.

- [候选相对基线的 diff](candidate.diff) · [候选完整源码](../../records/research/rounds/round-13/experiment/model/model.py)
- [对照相对基线的 diff](control.diff) · [对照完整源码](../../records/research/rounds/round-13/experiment/model/control-1.py)

## 原始证据导航

- [研究员完整提案](../../records/research/rounds/round-13/researcher/response.json)
- [研究员实际上下文](../../records/research/rounds/round-13/researcher/context.json)
- [实验员返回](../../records/research/rounds/round-13/experiment/response.json)
- [本轮全部归档](../../records/research/rounds/round-13)
- [拒绝事件摘录](rejections.json)
- [六项指标与机制比较](../../records/research/rounds/round-13/development-readout.json)
- [分析员完整返回与知识写入](../../records/research/rounds/round-13/analyst/response.json)
- [晋升记录](../../records/research/rounds/round-13/promotion.json)
