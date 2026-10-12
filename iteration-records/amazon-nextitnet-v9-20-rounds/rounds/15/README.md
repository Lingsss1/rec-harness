# 第 15 轮 撤销归一化前历史注入

[返回总览](../../README.md) · [机器可读汇总](../../rounds.json)

状态：**晋升候选**。当轮基线：`amazon-next-v9-round-14-model`。

## 假设与实验设计

新 dilation 顺序下，第 8 轮的额外注入可能带来干扰。

| 版本 | 改动 |
| --- | --- |
| 候选 | 最后 block 后、LayerNorm 前的前缀均值系数从 0.1 改为 0。 |
| 机制对照 | 相同系数改为 0.2。 |

基线和机制对照用途不同：前者判断是否值得保留，后者帮助区分解释。所有已训练版本使用 seed 42；并不承诺新增结构与基线逐参数初始化完全一致。

## 实验结果

| 品类 | 基线 HR@10 | 候选 HR@10 | 对照 HR@10 | 候选减基线 | 候选减对照 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Movies_and_TV | 0.65400 | 0.66500 | 0.66800 | +0.01100 | -0.00300 |
| Industrial_and_Scientific | 0.57900 | 0.57200 | 0.56800 | -0.00700 | +0.00400 |
| Electronics | 0.46700 | 0.47700 | 0.47100 | +0.01000 | +0.00600 |
| CDs_and_Vinyl | 0.68700 | 0.68500 | 0.68900 | -0.00200 | -0.00400 |
| **四品类平均** | **0.59675** | **0.59975** | **0.59900** | **+0.00300** | **+0.00075** |

Host 晋升：`amazon-next-v9-round-15-model`；选择策略 `baseline-single-seed-v1`；confirmation_executed = `False`。

## 中文归因与审阅关注点

候选晋升，但去掉和加倍都改善，且二者只差 0.00075；不能证明单调的少历史更好或过平滑解释。原 0.85/0.15 读出保留。

分析员结论标签：`provisional`；建议：`keep`。这里的 keep 是保留开发集记录供 host 选择，不等于机制确认。

### 分析员原始理由

The host reports that the candidate passed its baseline-only development eligibility rule. Seed-42 mean HR@10 was 0.59975 for the candidate, 0.59900 for the generated control, and 0.59675 for the incumbent. The candidate missed the proposal's >0.60175 target. Mixed category effects and absent confirmation leave scientific benefit and mechanism attribution provisional; performance selection belongs to the host.

### 尚未排除的解释

- Both new arms improved on the incumbent, while the candidate exceeded the control by only 0.00075. Initialization, optimization, and checkpoint selection remain plausible explanations.
- The candidate lost to the incumbent in Industrial_and_Scientific and CDs_and_Vinyl, and to the control in Movies_and_TV and CDs_and_Vinyl. The fixed dilation order may interact with context strength.
- Exploratory history slices include null and negative effects, sparse long-history bins, and no control-by-history comparison.
- The implementation author's local CPU checks were unexecuted. Host synthetic checks do not establish an explicit fixed-weight three-coefficient formula comparison, all-input semantics, or causal isolation.
- No confirmation result or independent-seed replication is supplied. The next discriminating experiment is a fixed-weight formula check followed by prespecified, paired coefficient-by-dilation comparisons across independent frozen-protocol seeds.

## 审核和返回修改


### reviewer

原始判定：`approve`。[完整记录](../../records/research/rounds/round-15/reviewer/response.json)

阻塞项：



已检查：

- For implementation_hash 85ad19cab7c68387afaca6f970add58dbe94fa0dd9d4f24679e15057567adc3c, independently recomputed SHA-256 hashes match the supplied baseline, candidate, and control source hashes.
- Each arm changes only line 248 of NextItNetModel.encode_sequence: the baseline coefficient is 0.1, the candidate is 0.0, and control-1.py is 0.2, as the frozen insertion contract specifies.
- Both arms retain the masked, strictly preceding mean, the 0.85/0.15 readout blend, and unchanged forward and score_candidates dot-product scoring.

限制：

- Host CPU semantic checks and execution-manifest checks are pending; this source approval does not establish runtime correctness, ranking benefit, or causal isolation.
- The model directory was not supplied. Review used the source texts embedded in context.json.

本轮没有 role_rejected 事件。

## 实际调用与训练成本

| 调用目录 | 输入 token | 缓存输入 token | 输出 token | 用时秒 |
| --- | ---: | ---: | ---: | ---: |
| [analyst](../../records/research/rounds/round-15/analyst/request.json) | 385,483 | 333,568 | 5,911 | 73.2 |
| [experiment](../../records/research/rounds/round-15/experiment/request.json) | 172,838 | 149,248 | 1,914 | 38.5 |
| [researcher](../../records/research/rounds/round-15/researcher/request.json) | 480,507 | 412,288 | 5,225 | 82.5 |
| [reviewer](../../records/research/rounds/round-15/reviewer/request.json) | 306,471 | 275,328 | 3,296 | 58.4 |

缓存输入是输入 token 的子集；reasoning token 是输出的子集，不重复相加。这些计数不能直接换算 Pro 订阅额度。

| 训练任务 | 最佳 epoch | 完成 epoch | 分配 GPU 秒 |
| --- | ---: | ---: | ---: |
| [amazon-next-v9-round-15-control-1-CDs_and_Vinyl-42](../../records/study/jobs/amazon-next-v9-round-15-control-1-CDs_and_Vinyl-42/completed.json) | 115 | 135 | 865.4 |
| [amazon-next-v9-round-15-control-1-Electronics-42](../../records/study/jobs/amazon-next-v9-round-15-control-1-Electronics-42/completed.json) | 40 | 60 | 402.5 |
| [amazon-next-v9-round-15-control-1-Industrial_and_Scientific-42](../../records/study/jobs/amazon-next-v9-round-15-control-1-Industrial_and_Scientific-42/completed.json) | 50 | 70 | 470.2 |
| [amazon-next-v9-round-15-control-1-Movies_and_TV-42](../../records/study/jobs/amazon-next-v9-round-15-control-1-Movies_and_TV-42/completed.json) | 105 | 125 | 741.2 |
| [amazon-next-v9-round-15-model-CDs_and_Vinyl-42](../../records/study/jobs/amazon-next-v9-round-15-model-CDs_and_Vinyl-42/completed.json) | 120 | 140 | 858.8 |
| [amazon-next-v9-round-15-model-Electronics-42](../../records/study/jobs/amazon-next-v9-round-15-model-Electronics-42/completed.json) | 40 | 60 | 410.4 |
| [amazon-next-v9-round-15-model-Industrial_and_Scientific-42](../../records/study/jobs/amazon-next-v9-round-15-model-Industrial_and_Scientific-42/completed.json) | 75 | 95 | 638.6 |
| [amazon-next-v9-round-15-model-Movies_and_TV-42](../../records/study/jobs/amazon-next-v9-round-15-model-Movies_and_TV-42/completed.json) | 120 | 140 | 858.4 |

完整训练曲线在对应任务的 curve.jsonl；此表只计本轮新版本，复用基线不重复计费。

## 提案原文与源码差异

### 机制

The incumbent and both new arms differ only in final-block pre-normalization context strength. CPU checks establish the intended computation; frozen ranking comparisons test usefulness, while mechanism attribution remains provisional.

### 预期观测

Provisional prediction: seed-42 unweighted four-category mean HR@10 exceeds 0.60175, strictly more than 0.005 above the observed 0.59675 incumbent, and exceeds the generated 0.2 control. Report all four category effects and available history-length slices with counts; no slice gain is established in advance.

### 证伪条件

Reject implementation admission if source scope, coefficient formulas, causality, masking, or scoring checks fail. Reject development advancement if candidate mean HR@10 is at most 0.60175, does not exceed the generated control, or fails host eligibility. Run the frozen confirmation procedure only after development qualification.

### 计划正确性检查

- Diff both arms against the supplied source and verify that only the declared coefficient in NextItNetModel.encode_sequence changes.
- With fixed CPU weights and synthetic sequences, verify coefficients 0.0, 0.1, and 0.2 against explicit preceding-mean formulas.
- Check empty, singleton, dense, and right-aligned histories; future-token invariance; padding masks and gradients; finite outputs and gradients; and batch independence before and after a synthetic update.
- Verify forward and score_candidates identities after the update and audit independent initialization and frozen execution manifests.

- [候选相对基线的 diff](candidate.diff) · [候选完整源码](../../records/research/rounds/round-15/experiment/model/model.py)
- [对照相对基线的 diff](control.diff) · [对照完整源码](../../records/research/rounds/round-15/experiment/model/control-1.py)

## 原始证据导航

- [研究员完整提案](../../records/research/rounds/round-15/researcher/response.json)
- [研究员实际上下文](../../records/research/rounds/round-15/researcher/context.json)
- [实验员返回](../../records/research/rounds/round-15/experiment/response.json)
- [本轮全部归档](../../records/research/rounds/round-15)
- [拒绝事件摘录](rejections.json)
- [六项指标与机制比较](../../records/research/rounds/round-15/development-readout.json)
- [分析员完整返回与知识写入](../../records/research/rounds/round-15/analyst/response.json)
- [晋升记录](../../records/research/rounds/round-15/promotion.json)
