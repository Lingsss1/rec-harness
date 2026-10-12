# 第 07 轮 局部状态变化残差

[返回总览](../../README.md) · [机器可读汇总](../../rounds.json)

状态：**未晋升**。当轮基线：`amazon-next-v8-round-1-control-1`。

## 假设与实验设计

相邻状态之差可以补充局部兴趣变化。

| 版本 | 改动 |
| --- | --- |
| 候选 | 投影 x_t-x_previous，加入读出。 |
| 机制对照 | 相同容量投影 x_t+x_previous。 |

基线和机制对照用途不同：前者判断是否值得保留，后者帮助区分解释。所有已训练版本使用 seed 42；并不承诺新增结构与基线逐参数初始化完全一致。

## 实验结果

| 品类 | 基线 HR@10 | 候选 HR@10 | 对照 HR@10 | 候选减基线 | 候选减对照 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Movies_and_TV | 0.64800 | 0.64400 | 0.66100 | -0.00400 | -0.01700 |
| Industrial_and_Scientific | 0.56100 | 0.55600 | 0.54800 | -0.00500 | +0.00800 |
| Electronics | 0.48600 | 0.45700 | 0.45300 | -0.02900 | +0.00400 |
| CDs_and_Vinyl | 0.68400 | 0.66500 | 0.68800 | -0.01900 | -0.02300 |
| **四品类平均** | **0.59475** | **0.58050** | **0.58750** | **-0.01425** | **-0.00700** |

Host 晋升：`无`；选择策略 `baseline-single-seed-v1`；confirmation_executed = `False`。

## 中文归因与审阅关注点

差值版本更差；输入分布、优化及现有卷积已捕获变化均是替代解释。

分析员结论标签：`provisional`；建议：`discard`。这里的 keep 是保留开发集记录供 host 选择，不等于机制确认。

### 分析员原始理由

Discard from advancement: the audited seed-42 public development mean HR@10 was 0.5805 for the candidate, 0.5875 for the local-sum control, and 0.59475 for the incumbent. The candidate was below both comparators, missed its proposed >0.59975 prediction, and the host eligibility flag was false. Category contrasts were mixed; neither new arm exceeded the incumbent in aggregate. Host admission and completed frozen-protocol training support that these are development outcomes, but they do not establish a causal mechanism or confirmation performance.

### 尚未排除的解释

- The incumbent may already encode useful local transitions; the added projection may instead change optimization or overfit.
- Matched parameter counts do not equalize the distributions of local-difference and local-sum inputs. Independent initialization and validation checkpoint selection may also contribute to this single-seed contrast.
- The supplied evidence does not expose explicit fixed-state nonzero-projection formula results, zero-projection equivalence to the incumbent, or post-update forward and score_candidates identity results. Finite synthetic admission and training checks do not establish those claims.
- Exploratory history-length slices have positive, zero, and negative effects, but no paired candidate-control history-length contrast is supplied. No confirmation or final-test result is supplied.

## 审核和返回修改


### reviewer

原始判定：`approve`。[完整记录](../../records/research/rounds/round-7/reviewer/response.json)

阻塞项：



已检查：

- For implementation_hash eac7364d227ed4c21027854fe84f3aae9cd27e97d9d7ec7aa7d1cd078c119231, independently recomputed SHA-256 for all three supplied source texts; each matches its declared source hash.
- Compared baseline.py with model.py and control-1.py. Changes are confined to NextItNetModel.__init__ and encode_sequence, as required by the insertion contract. The candidate and control differ only in local difference versus local sum.
- Both arms use the same zero-initialized, bias-free projection, 0.1 residual scale, immediately preceding valid-token mask, and incumbent context blend. Source inspection shows causal predecessor access and zero residual at padding positions.
- Forward and score_candidates still consume encode_sequence; the scoring implementation is unchanged.

限制：

- Host CPU contract checks are pending, including fixed-state formula, gradient, masking, and scoring checks after an update. No training or ranking result was reviewed. Source approval does not establish correctness or causal isolation.

本轮没有 role_rejected 事件。

## 实际调用与训练成本

| 调用目录 | 输入 token | 缓存输入 token | 输出 token | 用时秒 |
| --- | ---: | ---: | ---: | ---: |
| [analyst](../../records/research/rounds/round-7/analyst/request.json) | 366,023 | 314,880 | 4,667 | 78.7 |
| [experiment](../../records/research/rounds/round-7/experiment/request.json) | 214,163 | 186,240 | 2,289 | 53.1 |
| [researcher](../../records/research/rounds/round-7/researcher/request.json) | 655,852 | 589,184 | 5,541 | 106.0 |
| [reviewer](../../records/research/rounds/round-7/reviewer/request.json) | 345,999 | 310,272 | 4,911 | 92.6 |

缓存输入是输入 token 的子集；reasoning token 是输出的子集，不重复相加。这些计数不能直接换算 Pro 订阅额度。

| 训练任务 | 最佳 epoch | 完成 epoch | 分配 GPU 秒 |
| --- | ---: | ---: | ---: |
| [amazon-next-v9-round-7-control-1-CDs_and_Vinyl-42](../../records/study/jobs/amazon-next-v9-round-7-control-1-CDs_and_Vinyl-42/completed.json) | 95 | 115 | 727.8 |
| [amazon-next-v9-round-7-control-1-Electronics-42](../../records/study/jobs/amazon-next-v9-round-7-control-1-Electronics-42/completed.json) | 5 | 25 | 136.5 |
| [amazon-next-v9-round-7-control-1-Industrial_and_Scientific-42](../../records/study/jobs/amazon-next-v9-round-7-control-1-Industrial_and_Scientific-42/completed.json) | 25 | 45 | 303.2 |
| [amazon-next-v9-round-7-control-1-Movies_and_TV-42](../../records/study/jobs/amazon-next-v9-round-7-control-1-Movies_and_TV-42/completed.json) | 95 | 115 | 734.4 |
| [amazon-next-v9-round-7-model-CDs_and_Vinyl-42](../../records/study/jobs/amazon-next-v9-round-7-model-CDs_and_Vinyl-42/completed.json) | 60 | 80 | 498.4 |
| [amazon-next-v9-round-7-model-Electronics-42](../../records/study/jobs/amazon-next-v9-round-7-model-Electronics-42/completed.json) | 5 | 25 | 232.4 |
| [amazon-next-v9-round-7-model-Industrial_and_Scientific-42](../../records/study/jobs/amazon-next-v9-round-7-model-Industrial_and_Scientific-42/completed.json) | 50 | 70 | 478.8 |
| [amazon-next-v9-round-7-model-Movies_and_TV-42](../../records/study/jobs/amazon-next-v9-round-7-model-Movies_and_TV-42/completed.json) | 50 | 70 | 463.7 |

完整训练曲线在对应任务的 curve.jsonl；此表只计本轮新版本，复用基线不重复计费。

## 提案原文与源码差异

### 机制

Compare a local-change residual with both the audited incumbent and an equal-capacity local-sum residual. The CPU contrast verifies that the two branches compute the intended distinct signals; the frozen ranking contrast tests their practical effect but cannot by itself isolate the cause of any gain.

### 预期观测

The falsifiable ranking prediction is seed-42 unweighted four-category mean HR@10 strictly above 0.59975, which is more than 0.005 above the observed 0.59475 incumbent mean, and above the generated control. Report all four category effects and available history-length slices with counts; no slice benefit is established in advance.

### 证伪条件

Reject implementation admission if the fixed-state formulas, causality, masking, or scoring contracts fail. Reject advancement if candidate mean HR@10 is at most 0.59975, does not exceed the generated control, or the host eligibility flag is false. If development qualifies, apply the frozen confirmation procedure; an aggregate gain alone does not establish that local change caused it.

### 计划正确性检查

- Verify edits are confined to NextItNetModel.__init__ and encode_sequence in model.py and control-1.py; preserve all unrelated code and signatures.
- On fixed states and a fixed nonzero projection, check the candidate's local-difference formula and the control's local-sum formula against explicit CPU calculations.
- Check empty, singleton, dense, and right-aligned histories; future-token perturbation invariance; finite outputs and gradients; zero padding gradients; and batch independence before and after a synthetic update.
- Verify both arms equal the incumbent at zero projection initialization and diverge on suitable fixed states after assigning the same nonzero projection.
- Verify forward and score_candidates identities after an update, and audit fresh per-category initialization and the frozen execution manifest.

- [候选相对基线的 diff](candidate.diff) · [候选完整源码](../../records/research/rounds/round-7/experiment/model/model.py)
- [对照相对基线的 diff](control.diff) · [对照完整源码](../../records/research/rounds/round-7/experiment/model/control-1.py)

## 原始证据导航

- [研究员完整提案](../../records/research/rounds/round-7/researcher/response.json)
- [研究员实际上下文](../../records/research/rounds/round-7/researcher/context.json)
- [实验员返回](../../records/research/rounds/round-7/experiment/response.json)
- [本轮全部归档](../../records/research/rounds/round-7)
- [拒绝事件摘录](rejections.json)
- [六项指标与机制比较](../../records/research/rounds/round-7/development-readout.json)
- [分析员完整返回与知识写入](../../records/research/rounds/round-7/analyst/response.json)
- [晋升记录](../../records/research/rounds/round-7/promotion.json)
