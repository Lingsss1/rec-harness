## 181. taobao-second-query-direction-norm-5514

- 假设：Second-round task dependence may reflect query direction rather than score scale.
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "experiment_id": "taobao-second-query-direction-norm-5514",
  "timestamp": "2026-09-09T08:54:30.421574+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "family": "second_normalized_query_direction_norm_fixed_checkpoint",
  "code_id": "25ad69fe695a02d9c93b4464dd4951dbf9c75f8faa073b41287ce624431cf76c",
  "hypothesis": "Second-round task dependence may reflect query direction rather than score scale.",
  "change": "Original q4/q8, four q_norm-output conditions: native, first direction, first norm, both first; native/both exactly reproduce accepted saved outputs.",
  "budget": {
    "training_updates": 0,
    "new_training_seeds": [],
    "condition_row_forwards": 44112,
    "bootstrap_repetitions": 10000,
    "diagnostic_seed": 20262027
  },
  "metrics": {
    "q4_buy": [
      {
        "index": 3,
        "family": "ctx_q4",
        "contrast": "direction_minus_native",
        "task": "buy",
        "estimate": 1.4209815146104735e-05,
        "exploratory_ci95": [
          9.630137677801703e-06,
          1.9103530481695267e-05
        ],
        "family_ci99875": [
          6.8150539615716595e-06,
          2.187782277387799e-05
        ],
        "epsilon": 1e-06,
        "direction": "positive",
        "within_epsilon": false
      },
      {
        "index": 7,
        "family": "ctx_q4",
        "contrast": "norm_minus_native",
        "task": "buy",
        "estimate": 1.1873602300644224e-07,
        "exploratory_ci95": [
          7.730031934032012e-08,
          1.6693851830328818e-07
        ],
        "family_ci99875": [
          5.69194040327133e-08,
          2.0406359337706627e-07
        ],
        "epsilon": 1e-06,
        "direction": "inconclusive",
        "within_epsilon": true
      },
      {
        "index": 11,
        "family": "ctx_q4",
        "contrast": "both_minus_native",
        "task": "buy",
        "estimate": 1.4311656309733062e-05,
        "exploratory_ci95": [
          9.738095312462311e-06,
          1.922054970522917e-05
        ],
        "family_ci99875": [
          6.911138519482839e-06,
          2.1986460117554574e-05
        ],
        "epsilon": 1e-06,
        "direction": "positive",
        "within_epsilon": false
      },
      {
        "index": 15,
        "family": "ctx_q4",
        "contrast": "direction_minus_norm",
        "task": "buy",
        "estimate": 1.4091079123098342e-05,
        "exploratory_ci95": [
          9.523660949275898e-06,
          1.898621592838911e-05
        ],
        "family_ci99875": [
          6.712091653122236e-06,
          2.170999681939045e-05
        ],
        "epsilon": 1e-06,
        "direction": "positive",
        "within_epsilon": false
      },
      {
        "index": 19,
        "family": "ctx_q4",
        "contrast": "interaction",
        "task": "buy",
        "estimate": -1.689485937813561e-08,
        "exploratory_ci95": [
          -3.427779724919373e-08,
          -8.890910491542981e-10
        ],
        "family_ci99875": [
          -4.6968220655578777e-08,
          1.0057845432720967e-08
        ],
        "epsilon": 1e-06,
        "direction": "inconclusive",
        "within_epsilon": true
      }
    ],
    "conditional_comparisons": [
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
    ],
    "geometry": {
      "ctx_q4": {
        "valid_vectors": 88224,
        "fallback_vectors": 0,
        "norm2_over_norm1_quantiles_005_05_095": [
          0.9977528657075765,
          1.000060973297503,
          1.0033178298815306
        ],
        "cosine_quantiles_005_05_095": [
          0.8473691344261169,
          0.904393881559372,
          0.9441036880016327
        ],
        "scope": "Post-export descriptive unweighted vector geometry; no new task selection or inferential tests"
      },
      "explore_dual_no_feedback": {
        "valid_vectors": 176448,
        "fallback_vectors": 0,
        "norm2_over_norm1_quantiles_005_05_095": [
          0.9978875006444082,
          1.0005027009323166,
          1.0029324521784693
        ],
        "cosine_quantiles_005_05_095": [
          0.8351314008235932,
          0.9021984040737152,
          0.9441814690828323
        ],
        "scope": "Post-export descriptive unweighted vector geometry; no new task selection or inferential tests"
      }
    },
    "cost": {
      "wall_seconds": 170.6136488765478,
      "cpu_seconds": 591.671294977,
      "peak_rss_bytes": 2848133120,
      "gpu_seconds": 0,
      "training_updates": 0,
      "condition_row_forwards": 44112,
      "rss_scope": "Conservative parent plus child process high-water marks",
      "artifact_bytes": 32210827
    }
  },
  "validity": {
    "valid": true,
    "violations": [],
    "test_evaluated": false,
    "scope": "Original two checkpoints,5514 selected validation rows,4647 users,IPW/family40 bootstrap; conditional specified-replacement effects only."
  },
  "decision": "provisional",
  "confidence": "same_checkpoint_pathway_decomposition_not_new_training_gain",
  "global_best_changed": false,
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-query-direction-norm/RESULTS.md",
    "workspace/qformer-loop/taobao-transfer/recurrence-query-direction-norm/reports/interpretation.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-query-direction-norm/reports/result-review.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-query-direction-norm/reports/direction-norm/aggregate/aggregate.json"
  ],
  "next_question": "Design a distinct directional retrieval change with native score scale preserved; current dependence alone does not predict trained gains."
}
```

</details>
