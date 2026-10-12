# 第 06 轮 内容条件卷积门控

[返回总览](../../README.md) · [机器可读汇总](../../rounds.json)

状态：**未晋升**。当轮基线：`amazon-next-v8-round-1-control-1`。

## 假设与实验设计

门控根据内容选择卷积分支输出可改善预测。

| 版本 | 改动 |
| --- | --- |
| 候选 | 门控卷积读取归一化分支输入，乘数为 2*sigmoid(gate)。 |
| 机制对照 | 同样门控只读取全 1 张量。 |

基线和机制对照用途不同：前者判断是否值得保留，后者帮助区分解释。所有已训练版本使用 seed 42；并不承诺新增结构与基线逐参数初始化完全一致。

## 实验结果

| 品类 | 基线 HR@10 | 候选 HR@10 | 对照 HR@10 | 候选减基线 | 候选减对照 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Movies_and_TV | 0.64800 | 0.66000 | 0.65900 | +0.01200 | +0.00100 |
| Industrial_and_Scientific | 0.56100 | 0.55600 | 0.54700 | -0.00500 | +0.00900 |
| Electronics | 0.48600 | 0.47300 | 0.45400 | -0.01300 | +0.01900 |
| CDs_and_Vinyl | 0.68400 | 0.67100 | 0.67200 | -0.01300 | -0.00100 |
| **四品类平均** | **0.59475** | **0.59000** | **0.58300** | **-0.00475** | **+0.00700** |

Host 晋升：`无`；选择策略 `baseline-single-seed-v1`；confirmation_executed = `False`。

## 中文归因与审阅关注点

动态门控优于静态门控，但仍低于基线；不能以击败较弱对照证明新增模块有用。

分析员结论标签：`unknown`；建议：`discard`。这里的 keep 是保留开发集记录供 host 选择，不等于机制确认。

### 分析员原始理由

The audited seed-42 development mean HR@10 is 0.590 for the candidate, 0.59475 for the incumbent, and 0.583 for the static-gate control. The candidate exceeds the control by 0.007 but falls below the incumbent by 0.00475 and misses the preregistered 0.59975 target. The supplied readout marks it ineligible for confirmation, so the selection is discard under the supplied rule. The development contrast does not establish a causal mechanism or a confirmation outcome.

### 尚未排除的解释

- Independent initialization, optimization, or public-validation checkpoint selection could explain the candidate-control difference.
- The static-gate control may be weaker than the incumbent; the candidate’s advantage over it does not establish useful content conditioning.
- The candidate loses to the incumbent in three of four categories. Its contrasts with the control are near null in Movies_and_TV (+0.001) and CDs_and_Vinyl (-0.001).
- No supplied direct test establishes trained gate sensitivity to content, initial block equivalence to the incumbent, or scoring identity after a nonzero update and save/load. Finite synthetic checks do not establish those diagnostics.
- No independent-seed replication or confirmation result is supplied.

## 审核和返回修改


### reviewer

原始判定：`approve`。[完整记录](../../records/research/rounds/round-6/reviewer/response.json)

阻塞项：



已检查：

- For implementation_hash 48ddfd6526b5a8f8df183deb23aec93b0fd48e35281bc38fe5501c30b937e129, independently recomputed SHA-256 hashes match all three supplied source texts.
- Changes from baseline are confined to NextItNetBlock.__init__ and NextItNetBlock.forward in model.py and control-1.py; unrelated source, encoding, scoring, and training helpers are unchanged.
- Both arms add the same zero-initialized causal convolution and multiply the existing activated second-branch output by 2*sigmoid(gate output), giving a multiplier of one at initialization.
- The candidate gate receives the normalized branch activation. The generated control receives torch.ones_like of that activation and is therefore content-invariant at a fixed position.
- The gate kernel and dilation match conv2. Forward and candidate scoring continue to use encode_sequence, and candidate scoring retains the right-aligned tail readout.

限制：

- Host CPU admission checks are pending. Source review does not establish runtime correctness, learned gate sensitivity, ranking gains, or causal isolation.
- Causal zero padding can make the control gate position-dependent near the sequence boundary; it remains content-invariant at each fixed position.

本轮没有 role_rejected 事件。

## 实际调用与训练成本

| 调用目录 | 输入 token | 缓存输入 token | 输出 token | 用时秒 |
| --- | ---: | ---: | ---: | ---: |
| [analyst](../../records/research/rounds/round-6/analyst/request.json) | 311,850 | 266,624 | 4,538 | 71.4 |
| [experiment](../../records/research/rounds/round-6/experiment/request.json) | 209,295 | 182,784 | 2,040 | 47.2 |
| [researcher](../../records/research/rounds/round-6/researcher/request.json) | 598,825 | 525,824 | 4,745 | 95.3 |
| [reviewer](../../records/research/rounds/round-6/reviewer/request.json) | 326,580 | 290,944 | 5,052 | 122.8 |

缓存输入是输入 token 的子集；reasoning token 是输出的子集，不重复相加。这些计数不能直接换算 Pro 订阅额度。

| 训练任务 | 最佳 epoch | 完成 epoch | 分配 GPU 秒 |
| --- | ---: | ---: | ---: |
| [amazon-next-v9-round-6-control-1-CDs_and_Vinyl-42](../../records/study/jobs/amazon-next-v9-round-6-control-1-CDs_and_Vinyl-42/completed.json) | 60 | 80 | 588.1 |
| [amazon-next-v9-round-6-control-1-Electronics-42](../../records/study/jobs/amazon-next-v9-round-6-control-1-Electronics-42/completed.json) | 5 | 25 | 186.9 |
| [amazon-next-v9-round-6-control-1-Industrial_and_Scientific-42](../../records/study/jobs/amazon-next-v9-round-6-control-1-Industrial_and_Scientific-42/completed.json) | 35 | 55 | 349.7 |
| [amazon-next-v9-round-6-control-1-Movies_and_TV-42](../../records/study/jobs/amazon-next-v9-round-6-control-1-Movies_and_TV-42/completed.json) | 85 | 105 | 744.3 |
| [amazon-next-v9-round-6-model-CDs_and_Vinyl-42](../../records/study/jobs/amazon-next-v9-round-6-model-CDs_and_Vinyl-42/completed.json) | 55 | 75 | 571.6 |
| [amazon-next-v9-round-6-model-Electronics-42](../../records/study/jobs/amazon-next-v9-round-6-model-Electronics-42/completed.json) | 35 | 55 | 487.1 |
| [amazon-next-v9-round-6-model-Industrial_and_Scientific-42](../../records/study/jobs/amazon-next-v9-round-6-model-Industrial_and_Scientific-42/completed.json) | 45 | 65 | 556.4 |
| [amazon-next-v9-round-6-model-Movies_and_TV-42](../../records/study/jobs/amazon-next-v9-round-6-model-Movies_and_TV-42/completed.json) | 95 | 115 | 846.9 |

完整训练曲线在对应任务的 curve.jsonl；此表只计本轮新版本，复用基线不重复计费。

## 提案原文与源码差异

### 机制

Compare the content-conditioned causal gate in model.py with the unchanged incumbent and the parameter-matched static-gate control in control-1.py. Only the candidate gate receives the block activation as input.

### 预期观测

Provisional prediction: candidate mean HR@10 exceeds the audited 0.59475 baseline by more than the frozen 0.005 minimum and exceeds the generated control. Industrial_and_Scientific and Electronics are priority slices to inspect because their observed baseline HR@10 values are 0.561 and 0.486; no slice gain is established.

### 证伪条件

Reject advancement if CPU contracts fail, candidate seed-42 four-category mean HR@10 is at most 0.59975, or candidate mean is at most the generated control mean. A passing development contrast alone leaves the content-conditioning mechanism unconfirmed.

### 计划正确性检查

- Verify initial candidate and control block outputs equal the incumbent for identical weights and inputs.
- Verify candidate gate sensitivity to activation content and control gate invariance at fixed position after a nonzero gate update.
- Verify causal input gradients, zero padding gradients, finite empty-history output, batch independence, and right-aligned tail readout before and after updates.
- Verify forward scoring and candidate scoring identities after a nonzero update and after save/load.

- [候选相对基线的 diff](candidate.diff) · [候选完整源码](../../records/research/rounds/round-6/experiment/model/model.py)
- [对照相对基线的 diff](control.diff) · [对照完整源码](../../records/research/rounds/round-6/experiment/model/control-1.py)

## 原始证据导航

- [研究员完整提案](../../records/research/rounds/round-6/researcher/response.json)
- [研究员实际上下文](../../records/research/rounds/round-6/researcher/context.json)
- [实验员返回](../../records/research/rounds/round-6/experiment/response.json)
- [本轮全部归档](../../records/research/rounds/round-6)
- [拒绝事件摘录](rejections.json)
- [六项指标与机制比较](../../records/research/rounds/round-6/development-readout.json)
- [分析员完整返回与知识写入](../../records/research/rounds/round-6/analyst/response.json)
- [晋升记录](../../records/research/rounds/round-6/promotion.json)
