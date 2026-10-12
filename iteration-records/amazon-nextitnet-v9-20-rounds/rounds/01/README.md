# 第 01 轮 当前物品坐标对齐

[返回总览](../../README.md) · [机器可读汇总](../../rounds.json)

状态：**未晋升**。当轮基线：`amazon-next-v8-round-1-control-1`。

## 假设与实验设计

补充当前物品 embedding 可改善身份信息保留。

| 版本 | 改动 |
| --- | --- |
| 候选 | 增加 0.10 倍有界当前物品信号。 |
| 机制对照 | 将相同信号的特征坐标循环旋转半个宽度。 |

基线和机制对照用途不同：前者判断是否值得保留，后者帮助区分解释。所有已训练版本使用 seed 42；并不承诺新增结构与基线逐参数初始化完全一致。

## 实验结果

| 品类 | 基线 HR@10 | 候选 HR@10 | 对照 HR@10 | 候选减基线 | 候选减对照 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Movies_and_TV | 0.64800 | 0.66100 | 0.64100 | +0.01300 | +0.02000 |
| Industrial_and_Scientific | 0.56100 | 0.57000 | 0.57000 | +0.00900 | +0.00000 |
| Electronics | 0.48600 | 0.45300 | 0.45100 | -0.03300 | +0.00200 |
| CDs_and_Vinyl | 0.68400 | 0.68000 | 0.66700 | -0.00400 | +0.01300 |
| **四品类平均** | **0.59475** | **0.59100** | **0.58225** | **-0.00375** | **+0.00875** |

Host 晋升：`无`；选择策略 `baseline-single-seed-v1`；confirmation_executed = `False`。

## 中文归因与审阅关注点

两者均低于基线；候选优于对照只能说明这两种注入有差异，不能证明增加身份信号有净收益。

分析员结论标签：`unknown`；建议：`discard`。这里的 keep 是保留开发集记录供 host 选择，不等于机制确认。

### 分析员原始理由

Discard under the supplied eligibility rule: audited seed-42 mean HR@10 is 0.591 for the candidate versus 0.59475 for the incumbent, below the proposed 0.59975 target; eligible_for_confirmation is false. The candidate exceeds the rotated control (0.58225) by 0.00875, but that descriptive contrast does not establish coordinate alignment. Candidate effects versus the incumbent are +0.013 for Movies_and_TV, +0.009 for Industrial_and_Scientific, -0.033 for Electronics, and -0.004 for CDs_and_Vinyl. The Industrial_and_Scientific candidate–control contrast is null. Audited receipts report fresh initialization and the frozen development protocol for the candidate and generated control; source admission and synthetic checks support bounded implementation validity, not a causal mechanism. No confirmation result is supplied.

### 尚未排除的解释

- The incumbent convolution path may already preserve useful current-item information; the proposed deficit was not measured directly.
- Representation scale, optimization, interaction of rotated coordinates with trained features, initialization, or checkpoint selection could explain the candidate–control difference.
- An archived fixed-tensor cross-arm test of exact rotation and equal per-position signal norms was not supplied. The implementation author reported no synthetic checks; host admission and finite synthetic probes do not establish causal isolation.
- The proposed internally padded and singleton-history checks and post-update scoring equality are not fully documented by the supplied evidence. Executed tensor paths and pre/post training contracts have finite-input scope.
- Single-seed public development and exploratory history-length slices do not establish replication, a history-length benefit, or confirmation performance.

## 审核和返回修改


### reviewer

原始判定：`approve`。[完整记录](../../records/research/rounds/round-1/reviewer/response.json)

阻塞项：



已检查：

- For implementation_hash 542232f7827d3206e85d618a92ec9b5b6d0e98cf075260d8091ef9e6b613eba3, the embedded baseline.py, model.py, and control-1.py texts independently match their declared SHA-256 hashes.
- Both edits are confined to NextItNetModel.encode_sequence; code outside that method matches the baseline.
- The candidate adds the specified 0.10 bounded, scaled current-item embedding signal after the incumbent blend. The control adds the same signal after the specified half-width feature rotation, preserving its norm. Both mask padding positions and add no parameters.
- The preceding-context computation, forward, and score_candidates are unchanged. Forward and candidate scoring both use encode_sequence output.

限制：

- Host CPU contracts, synthetic behavior checks, and training and evaluation remain pending. This source approval does not establish runtime correctness, performance, or causal isolation.
- The supplied implementation_hash identifies the reviewed bundle; its bundle-hash derivation was not supplied for independent recomputation.

本轮没有 role_rejected 事件。

## 实际调用与训练成本

| 调用目录 | 输入 token | 缓存输入 token | 输出 token | 用时秒 |
| --- | ---: | ---: | ---: | ---: |
| [analyst](../../records/research/rounds/round-1/analyst/request.json) | 307,484 | 260,736 | 3,910 | 82.0 |
| [experiment](../../records/research/rounds/round-1/experiment/request.json) | 210,662 | 183,552 | 2,222 | 69.0 |
| [researcher](../../records/research/rounds/round-1/researcher/request.json) | 637,550 | 564,480 | 4,575 | 120.7 |
| [reviewer](../../records/research/rounds/round-1/reviewer/request.json) | 261,245 | 230,912 | 3,802 | 79.4 |

缓存输入是输入 token 的子集；reasoning token 是输出的子集，不重复相加。这些计数不能直接换算 Pro 订阅额度。

| 训练任务 | 最佳 epoch | 完成 epoch | 分配 GPU 秒 |
| --- | ---: | ---: | ---: |
| [amazon-next-v9-round-1-control-1-CDs_and_Vinyl-42](../../records/study/jobs/amazon-next-v9-round-1-control-1-CDs_and_Vinyl-42/completed.json) | 60 | 80 | 528.7 |
| [amazon-next-v9-round-1-control-1-Electronics-42](../../records/study/jobs/amazon-next-v9-round-1-control-1-Electronics-42/completed.json) | 20 | 40 | 233.9 |
| [amazon-next-v9-round-1-control-1-Industrial_and_Scientific-42](../../records/study/jobs/amazon-next-v9-round-1-control-1-Industrial_and_Scientific-42/completed.json) | 70 | 90 | 595.6 |
| [amazon-next-v9-round-1-control-1-Movies_and_TV-42](../../records/study/jobs/amazon-next-v9-round-1-control-1-Movies_and_TV-42/completed.json) | 55 | 75 | 510.2 |
| [amazon-next-v9-round-1-model-CDs_and_Vinyl-42](../../records/study/jobs/amazon-next-v9-round-1-model-CDs_and_Vinyl-42/completed.json) | 95 | 115 | 833.3 |
| [amazon-next-v9-round-1-model-Electronics-42](../../records/study/jobs/amazon-next-v9-round-1-model-Electronics-42/completed.json) | 5 | 25 | 242.8 |
| [amazon-next-v9-round-1-model-Industrial_and_Scientific-42](../../records/study/jobs/amazon-next-v9-round-1-model-Industrial_and_Scientific-42/completed.json) | 55 | 75 | 532.9 |
| [amazon-next-v9-round-1-model-Movies_and_TV-42](../../records/study/jobs/amazon-next-v9-round-1-model-Movies_and_TV-42/completed.json) | 105 | 125 | 780.3 |

完整训练曲线在对应任务的 curve.jsonl；此表只计本轮新版本，复用基线不重复计费。

## 提案原文与源码差异

### 机制

Compare the incumbent, the aligned current-item signal, and its equal-norm rotated control. The baseline contrast tests ranking benefit; the candidate-control contrast tests the provisional coordinate-alignment explanation. The sole supplied graph observation is current-baseline-430dd0003cb20fc2553e1a60; no graph relation supplies causal support.

### 预期观测

The provisional target is seed-42 unweighted mean HR@10 strictly above 0.59975, versus the observed baseline mean 0.59475, and above the generated rotated-signal control. Report all four category effects and available history-length slices, including regressions; no slice benefit is established in advance.

### 证伪条件

Reject implementation validity if equal-norm rotation, causality, masking, or scoring checks fail. The ranking prediction fails if candidate mean HR@10 is at most 0.59975 or is at most the generated control mean. A qualifying seed-42 result remains provisional and does not establish the alignment mechanism.

### 计划正确性检查

- Verify edits in model.py and control-1.py are confined to NextItNetModel.encode_sequence.
- On fixed synthetic tensors, verify candidate and control added signals have equal per-position norms and differ by the specified feature rotation.
- Check dense, right-aligned, internally padded, singleton, and empty histories for finite masked outputs and unchanged input IDs.
- Check future-token perturbations cannot affect earlier outputs, padding gradients are zero, and batch permutation equivariance holds before and after a synthetic update.
- After a synthetic update, verify score_candidates equals item_emb(candidates) @ encode_sequence(seq)[0,-1] and forward uses those same representations.
- Audit independent initialization, unchanged parameter count, and execution of the frozen training and evaluation protocol.

- [候选相对基线的 diff](candidate.diff) · [候选完整源码](../../records/research/rounds/round-1/experiment/model/model.py)
- [对照相对基线的 diff](control.diff) · [对照完整源码](../../records/research/rounds/round-1/experiment/model/control-1.py)

## 原始证据导航

- [研究员完整提案](../../records/research/rounds/round-1/researcher/response.json)
- [研究员实际上下文](../../records/research/rounds/round-1/researcher/context.json)
- [实验员返回](../../records/research/rounds/round-1/experiment/response.json)
- [本轮全部归档](../../records/research/rounds/round-1)
- [拒绝事件摘录](rejections.json)
- [六项指标与机制比较](../../records/research/rounds/round-1/development-readout.json)
- [分析员完整返回与知识写入](../../records/research/rounds/round-1/analyst/response.json)
- [晋升记录](../../records/research/rounds/round-1/promotion.json)
