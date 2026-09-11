# 第二轮查询：方向与范数固定权重诊断

**q4.buy 在这组指定替换中主要依赖第二轮查询的方向变化。** 只换回第一轮方向，ΔIPW BCE 为 +1.42098e-5；只换范数为 +1.18736e-7。方向、联合替换、方向减范数三个区间均超过预定 ε，范数及非加性交互区间均落在 ±ε 内。这里范数影响很小，并非精确为零。其余七个模型／任务组合未通过联合门槛，不能推出只有购买需要循环。

两家模型的 native 与 both_first 在全部 5514 行上分别精确复现旧 native/reuse 概率及 BCE，fallback 为 0。这是对同一 checkpoint 效应的进一步分解，不是独立训练复现；新增训练收益尚未建立。既有任务间增益差异应查阅 [直接配对任务分析](../recurrence-task-effects/RESULTS.md)。

原 q4/q8、固定 5514 行验证子集、4647 用户、CPU FP32；四条件共享输入与权重。没有新增训练、seed 或 test 读取。

正 ΔBCE 表示指定替换更有害。全部 40 列用同一组 10000 次用户抽样；下表为预定 family40 的 99.875% 区间，ε=1e-6。95% 探索区间保留在 JSON。

| 模型 | 任务 | 对照 | ΔIPW BCE | 99.875% 区间 | ε方向 |
|---|---|---|---:|---|---|
| ctx_q4 | is_click | direction_minus_native | +0.000164118914 | [-0.000210156155, +0.000713902904] | inconclusive |
| ctx_q4 | cart | direction_minus_native | +2.71413394e-05 | [-5.54033752e-05, +7.75016659e-05] | inconclusive |
| ctx_q4 | fav | direction_minus_native | +0.00012665503 | [-2.47779063e-05, +0.000357195583] | inconclusive |
| ctx_q4 | buy | direction_minus_native | +1.42098151e-05 | [+6.81505396e-06, +2.18778228e-05] | positive |
| ctx_q4 | is_click | norm_minus_native | +6.41534543e-08 | [-2.12825636e-06, +2.25153808e-06] | inconclusive |
| ctx_q4 | cart | norm_minus_native | -6.77098091e-07 | [-5.65134763e-06, +6.2883022e-07] | inconclusive |
| ctx_q4 | fav | norm_minus_native | +5.8482347e-07 | [-2.08873789e-07, +1.25311925e-06] | inconclusive |
| ctx_q4 | buy | norm_minus_native | +1.18736023e-07 | [+5.6919404e-08, +2.04063593e-07] | inconclusive |
| ctx_q4 | is_click | both_minus_native | +0.000164056163 | [-0.000210001252, +0.000713307247] | inconclusive |
| ctx_q4 | cart | both_minus_native | +2.63857039e-05 | [-5.96770231e-05, +7.80062561e-05] | inconclusive |
| ctx_q4 | fav | both_minus_native | +0.000127024328 | [-2.42281988e-05, +0.000356905472] | inconclusive |
| ctx_q4 | buy | both_minus_native | +1.43116563e-05 | [+6.91113852e-06, +2.19864601e-05] | positive |
| ctx_q4 | is_click | direction_minus_norm | +0.000164054761 | [-0.000209872571, +0.000713676145] | inconclusive |
| ctx_q4 | cart | direction_minus_norm | +2.78184374e-05 | [-5.17895879e-05, +7.68565821e-05] | inconclusive |
| ctx_q4 | fav | direction_minus_norm | +0.000126070206 | [-2.54705867e-05, +0.000356777076] | inconclusive |
| ctx_q4 | buy | direction_minus_norm | +1.40910791e-05 | [+6.71209165e-06, +2.17099968e-05] | positive |
| ctx_q4 | is_click | interaction | -1.26904996e-07 | [-1.21402757e-06, +1.07422247e-06] | inconclusive |
| ctx_q4 | cart | interaction | -7.85373686e-08 | [-3.43669503e-07, +2.2157057e-07] | inconclusive |
| ctx_q4 | fav | interaction | -2.15525219e-07 | [-7.5483351e-07, +1.2679447e-07] | inconclusive |
| ctx_q4 | buy | interaction | -1.68948594e-08 | [-4.69682207e-08, +1.00578454e-08] | inconclusive |
| explore_dual_no_feedback | is_click | direction_minus_native | +0.000333522488 | [-9.84766056e-05, +0.000847844717] | inconclusive |
| explore_dual_no_feedback | cart | direction_minus_native | +0.000301331709 | [-0.000298898166, +0.0018280072] | inconclusive |
| explore_dual_no_feedback | fav | direction_minus_native | -5.30892679e-05 | [-0.000124743518, +3.51544023e-05] | inconclusive |
| explore_dual_no_feedback | buy | direction_minus_native | -1.48665452e-05 | [-4.24198593e-05, +1.27458148e-05] | inconclusive |
| explore_dual_no_feedback | is_click | norm_minus_native | +2.40587878e-07 | [-3.17064121e-06, +3.50389066e-06] | inconclusive |
| explore_dual_no_feedback | cart | norm_minus_native | -8.24919186e-07 | [-6.54031576e-06, +1.21942951e-06] | inconclusive |
| explore_dual_no_feedback | fav | norm_minus_native | +2.9506226e-07 | [-4.58766188e-07, +8.04115184e-07] | inconclusive |
| explore_dual_no_feedback | buy | norm_minus_native | -2.84408079e-07 | [-6.42986527e-07, -7.01696345e-08] | inconclusive |
| explore_dual_no_feedback | is_click | both_minus_native | +0.000333330322 | [-9.87759914e-05, +0.000845246986] | inconclusive |
| explore_dual_no_feedback | cart | both_minus_native | +0.000300394855 | [-0.000298227363, +0.00182462621] | inconclusive |
| explore_dual_no_feedback | fav | both_minus_native | -5.29029172e-05 | [-0.000124114295, +3.49836346e-05] | inconclusive |
| explore_dual_no_feedback | buy | both_minus_native | -1.50687218e-05 | [-4.28608294e-05, +1.25704294e-05] | inconclusive |
| explore_dual_no_feedback | is_click | direction_minus_norm | +0.0003332819 | [-9.91488845e-05, +0.000849485789] | inconclusive |
| explore_dual_no_feedback | cart | direction_minus_norm | +0.000302156628 | [-0.00029971858, +0.001834453] | inconclusive |
| explore_dual_no_feedback | fav | direction_minus_norm | -5.33843301e-05 | [-0.000125422815, +3.52099231e-05] | inconclusive |
| explore_dual_no_feedback | buy | direction_minus_norm | -1.45821371e-05 | [-4.18658345e-05, +1.29947172e-05] | inconclusive |
| explore_dual_no_feedback | is_click | interaction | -4.32753782e-07 | [-1.72453071e-06, +7.86992357e-07] | inconclusive |
| explore_dual_no_feedback | cart | interaction | -1.11935056e-07 | [-2.56139723e-06, +3.25269189e-06] | inconclusive |
| explore_dual_no_feedback | fav | interaction | -1.08711611e-07 | [-3.80232081e-07, +1.55613211e-07] | inconclusive |
| explore_dual_no_feedback | buy | interaction | +8.22315086e-08 | [-6.27531346e-09, +1.88436849e-07] | inconclusive |

联合门槛：

```json
[
  {
    "family": "ctx_q4",
    "task": "is_click",
    "specified_direction_replacement_more_harmful": false,
    "direction_larger_in_harmful_joint_replacement": false,
    "conditional_dominance_with_small_norm_and_interaction": false
  },
  {
    "family": "ctx_q4",
    "task": "cart",
    "specified_direction_replacement_more_harmful": false,
    "direction_larger_in_harmful_joint_replacement": false,
    "conditional_dominance_with_small_norm_and_interaction": false
  },
  {
    "family": "ctx_q4",
    "task": "fav",
    "specified_direction_replacement_more_harmful": false,
    "direction_larger_in_harmful_joint_replacement": false,
    "conditional_dominance_with_small_norm_and_interaction": false
  },
  {
    "family": "ctx_q4",
    "task": "buy",
    "specified_direction_replacement_more_harmful": true,
    "direction_larger_in_harmful_joint_replacement": true,
    "conditional_dominance_with_small_norm_and_interaction": true
  },
  {
    "family": "explore_dual_no_feedback",
    "task": "is_click",
    "specified_direction_replacement_more_harmful": false,
    "direction_larger_in_harmful_joint_replacement": false,
    "conditional_dominance_with_small_norm_and_interaction": false
  },
  {
    "family": "explore_dual_no_feedback",
    "task": "cart",
    "specified_direction_replacement_more_harmful": false,
    "direction_larger_in_harmful_joint_replacement": false,
    "conditional_dominance_with_small_norm_and_interaction": false
  },
  {
    "family": "explore_dual_no_feedback",
    "task": "fav",
    "specified_direction_replacement_more_harmful": false,
    "direction_larger_in_harmful_joint_replacement": false,
    "conditional_dominance_with_small_norm_and_interaction": false
  },
  {
    "family": "explore_dual_no_feedback",
    "task": "buy",
    "specified_direction_replacement_more_harmful": false,
    "direction_larger_in_harmful_joint_replacement": false,
    "conditional_dominance_with_small_norm_and_interaction": false
  }
]
```

这测量固定模型对指定检索变化的依赖，不能证明训练 AUC 提升。某任务通过而另一个未通过，不是任务间效应差异检验；区间跨零也不是无效或等效。方向与范数替换没有按相同几何幅度校准。

实际导出及聚合合计 wall 170.614 秒、CPU 591.671 秒，保守峰 RSS 2.653 GiB。训练更新与 GPU 时间为 0；共 44112 condition-row forwards。

小型合成 stage fixture 的成本另见 reports/probe-offline.json，不计作真实数据或训练实验。
