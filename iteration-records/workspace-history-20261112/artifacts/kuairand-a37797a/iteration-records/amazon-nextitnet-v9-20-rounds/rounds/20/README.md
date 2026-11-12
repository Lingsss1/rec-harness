# 第 20 轮 中间层可学习投影

[返回总览](../../README.md) · [机器可读汇总](../../rounds.json)

状态：**训练前失败**。当轮基线：`amazon-next-v9-round-15-model`。

## 假设与实验设计

用可学习投影提取第 4 层信息可能比固定混合更有效。

| 版本 | 改动 |
| --- | --- |
| 候选 | 第 4 层状态经归一化、零初始化投影和 tanh 加入读出。 |
| 机制对照 | 相同投影作用于最终层状态。 |

基线和机制对照用途不同：前者判断是否值得保留，后者帮助区分解释。所有已训练版本使用 seed 42；并不承诺新增结构与基线逐参数初始化完全一致。

## 实验结果

没有训练读数，没有分析员结论，也没有晋升；不能将实现准入失败当成假设被实验证伪。

## 中文归因与审阅关注点

未训练。初次包含不支持的 Raise，重试包含未批准的 list 调用，均未获准执行；不能判为性能或机制失败。

## 审核和返回修改

- 2026-10-11T09:01:33.945345+00:00：experiment，unsupported source syntax: Raise
- 2026-10-11T09:02:59.503971+00:00：experiment，unapproved callable: list
本轮没有独立 reviewer 返回；流程在进入该角色前已失败。
## 实际调用与训练成本

| 调用目录 | 输入 token | 缓存输入 token | 输出 token | 用时秒 |
| --- | ---: | ---: | ---: | ---: |
| [experiment](../../records/research/rounds/round-20/experiment/request.json) | 311,552 | 280,832 | 4,293 | 83.3 |
| [experiment-rejected-1](../../records/research/rounds/round-20/experiment-rejected-1/request.json) | 250,425 | 221,696 | 2,501 | 52.5 |
| [researcher](../../records/research/rounds/round-20/researcher/request.json) | 811,057 | 724,864 | 5,350 | 129.9 |

缓存输入是输入 token 的子集；reasoning token 是输出的子集，不重复相加。这些计数不能直接换算 Pro 订阅额度。

## 提案原文与源码差异

### 机制

Compare an intermediate-block readout residual with both the audited incumbent and an equal-capacity final-block readout residual. CPU tests establish that the branches use distinct intended states; frozen ranking comparisons test predictive value.

### 预期观测

Provisional: candidate mean HR@10 exceeds the incumbent's observed 0.59975 by more than 0.005 and exceeds the generated control. Inspect all four category effects, especially Movies and TV, where a prior dilation-order candidate regressed; no category gain is asserted.

### 证伪条件

Reject implementation admission if branch identity, causality, masking, or scoring checks fail. Reject development advancement if candidate mean HR@10 is at most 0.60475, is at most the generated control mean, or fails host eligibility. Apply the frozen confirmation procedure only after qualification. Even a qualifying ranking gain does not by itself prove the proposed mechanism.

### 计划正确性检查

- With identical nonzero test projection weights and fixed final state, perturbing the captured intermediate state changes only the candidate branch.
- Check causal input gradients, zero padding gradients, finite empty histories, right-aligned tail readout, batch independence, and output shape.
- After a synthetic update, verify score_candidates(seq,candidates) equals item_emb(candidates) @ encode_sequence(seq)[0,-1] and forward uses those same representations.

- [候选相对基线的 diff](candidate.diff) · [候选完整源码](../../records/research/rounds/round-20/experiment/model/model.py)
- [对照相对基线的 diff](control.diff) · [对照完整源码](../../records/research/rounds/round-20/experiment/model/control-1.py)

## 原始证据导航

- [研究员完整提案](../../records/research/rounds/round-20/researcher/response.json)
- [研究员实际上下文](../../records/research/rounds/round-20/researcher/context.json)
- [实验员返回](../../records/research/rounds/round-20/experiment/response.json)
- [本轮全部归档](../../records/research/rounds/round-20)
- [拒绝事件摘录](rejections.json)
