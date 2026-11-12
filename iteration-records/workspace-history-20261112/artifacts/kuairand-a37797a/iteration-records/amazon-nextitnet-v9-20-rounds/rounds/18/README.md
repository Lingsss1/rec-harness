# 第 18 轮 正负历史注入再次对比

[返回总览](../../README.md) · [机器可读汇总](../../rounds.json)

状态：**训练前失败**。当轮基线：`amazon-next-v9-round-15-model`。

## 假设与实验设计

用 +0.1 与 -0.1 的符号对比检验历史融合方向。

| 版本 | 改动 |
| --- | --- |
| 候选 | 归一化前注入 +0.1 倍前缀均值。 |
| 机制对照 | 注入 -0.1 倍。 |

基线和机制对照用途不同：前者判断是否值得保留，后者帮助区分解释。所有已训练版本使用 seed 42；并不承诺新增结构与基线逐参数初始化完全一致。

## 实验结果

没有训练读数，没有分析员结论，也没有晋升；不能将实现准入失败当成假设被实验证伪。

## 中文归因与审阅关注点

未训练。两次提交都与第 14 轮已有候选重复，被源码去重拦截；没有支持或反驳机制的实验结果。

## 审核和返回修改

- 2026-10-11T06:32:41.883519+00:00：experiment，new source model.py duplicates previously evaluated model amazon-next-v9-round-14-model; implement a distinct falsifiable candidate/control within the frozen proposal. Renaming a file or model does not make it new; cosmetic edits do not establish a distinct mechanism.
- 2026-10-11T06:33:23.181773+00:00：experiment，new source model.py duplicates previously evaluated model amazon-next-v9-round-14-model; implement a distinct falsifiable candidate/control within the frozen proposal. Renaming a file or model does not make it new; cosmetic edits do not establish a distinct mechanism.
本轮没有独立 reviewer 返回；流程在进入该角色前已失败。
## 实际调用与训练成本

| 调用目录 | 输入 token | 缓存输入 token | 输出 token | 用时秒 |
| --- | ---: | ---: | ---: | ---: |
| [experiment](../../records/research/rounds/round-18/experiment/request.json) | 228,083 | 211,328 | 1,705 | 38.5 |
| [experiment-rejected-1](../../records/research/rounds/round-18/experiment-rejected-1/request.json) | 235,117 | 208,896 | 1,987 | 45.2 |
| [researcher](../../records/research/rounds/round-18/researcher/request.json) | 693,128 | 609,792 | 5,998 | 100.6 |

缓存输入是输入 token 的子集；reasoning token 是输出的子集，不重复相加。这些计数不能直接换算 Pro 订阅额度。

## 提案原文与源码差异

### 机制

Compare positive pre-normalization predecessor context with the audited zero-multiplier incumbent and an equal-capacity negative-sign control. Fixed-state CPU checks identify the computation; frozen ranking comparisons test usefulness without proving causal attribution.

### 预期观测

Provisional prediction: the candidate's seed-42 unweighted four-category mean HR@10 exceeds 0.60475, strictly more than 0.005 above the observed 0.59975 incumbent mean, and exceeds the generated negative-sign control. Report all four paired category effects and available history-length slices with counts; no category or slice gain is established.

### 证伪条件

Reject source admission if the formulas, causality, masking or scoring contracts fail. Do not advance on development if candidate mean HR@10 is at most 0.60475, is at most the generated control mean, or fails host eligibility. If both signs improve similarly, the proposed positive-context explanation is weakened. Apply only the frozen confirmation procedure to a qualifying candidate; development selection alone does not confirm the mechanism.

### 计划正确性检查

- Verify edits in model.py and control-1.py are confined to NextItNetModel.encode_sequence and that the only candidate-control formula difference is the +0.1 versus -0.1 multiplier.
- On fixed CPU states, compare both insertions with explicit strictly preceding masked means; verify equal incumbent outputs when no predecessor exists and opposite pre-normalization deviations otherwise.
- Check empty, singleton, dense and right-aligned histories; future-token invariance; finite outputs and gradients; zero padding gradients; batch independence and unchanged input IDs before and after a synthetic update.
- After the update, verify forward score identities, score_candidates dot-product identity, tail readout and independent category/seed initialization.
- Audit execution manifests against the frozen protocol rather than source helper defaults.

- [候选相对基线的 diff](candidate.diff) · [候选完整源码](../../records/research/rounds/round-18/experiment/model/model.py)
- [对照相对基线的 diff](control.diff) · [对照完整源码](../../records/research/rounds/round-18/experiment/model/control-1.py)

## 原始证据导航

- [研究员完整提案](../../records/research/rounds/round-18/researcher/response.json)
- [研究员实际上下文](../../records/research/rounds/round-18/researcher/context.json)
- [实验员返回](../../records/research/rounds/round-18/experiment/response.json)
- [本轮全部归档](../../records/research/rounds/round-18)
- [拒绝事件摘录](rejections.json)
