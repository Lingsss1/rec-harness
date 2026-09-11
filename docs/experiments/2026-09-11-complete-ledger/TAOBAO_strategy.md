# Taobao strategy.json 快照

此处保留历史判断，不把待执行计划当作已完成实验。

```json
{
  "dataset": "taobao-unirank-release-v1",
  "primary_metric": "macro_AUC",
  "primary_baseline": "taobao_cross8",
  "current_best": {
    "family": "psgd_precond_every_step",
    "macro_auc": 0.8286235325196747,
    "per_task": {
      "is_click": {
        "logloss": 0.18200398487853026,
        "AUC": 0.6677897783914419
      },
      "cart": {
        "logloss": 0.01763503120391706,
        "AUC": 0.8670190266410652
      },
      "fav": {
        "logloss": 0.013270321997338044,
        "AUC": 0.8823612288873869
      },
      "buy": {
        "logloss": 0.006621879772426182,
        "AUC": 0.8973240961588049
      }
    },
    "seed": 20262027,
    "confidence": "single_seed_selection_uncorrected"
  },
  "completed_training_runs": 194,
  "active_hypothesis": null,
  "next_queue": [
    {
      "family": "round_accumulation_factorial",
      "state": "completed",
      "directory": "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation",
      "proposal": "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/study-design.json",
      "protocol": "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/protocol.json",
      "diagnostic_protocol": "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/diagnostic-protocol.json",
      "formal_training_started": 4,
      "planned_training_runs": 4,
      "formal_session": 41931,
      "formal_launcher_pid": 1275851,
      "smoke_updates_verified": 8,
      "smoke_verification_recovery": "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/reports/recovered-smoke-review.json",
      "additional_training_seeds": [],
      "reference_diagnostic_complete": "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/reports/diagnostic-exports/mature_r2/completion.json",
      "diagnostic_candidate_id": null,
      "diagnostic_task_id": null,
      "formal_training_completed": 4,
      "results": "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/RESULTS.md",
      "interpretation": "首轮累积路径对点击有局部收益，且依历史长度和槽而异；分轮与携带的训练交互为正，但分轮＋携带整体AUC显著低于原两轮，不能晋升。保留机制候选与Cross8对照候选，不增加seed。"
    },
    {
      "family": "endpoint_read",
      "state": "completed",
      "proposal": "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/proposal.json",
      "planned_training_runs": 3,
      "additional_training_seeds": [],
      "reused_factorial_control": "segment-read/first_even",
      "diagnostic_task_id": null,
      "diagnostic_candidate_id": null,
      "reason": "Independent exact endpoint intervention; no new estimator required, no ready handoffs directory exists; not derived from rejected gradient probe.",
      "smokes_completed": 3,
      "smoke_updates": 6,
      "formal_training_started": 3,
      "formal_session": 75924,
      "formal_launcher_pid": 1249807,
      "formal_authorization_sha256": "23b18e68185190f0e7f9130a18b60cb4c750eb004ce3da407cdd7ca662c5bbf4",
      "protocol": "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/protocol.json",
      "formal_training_completed": 3,
      "results": "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/RESULTS.md",
      "interpretation": "even_last保留研究候选：相对mature_r2 macro+0.001435345但CI跨零，点击改善/购买BCE退步，延迟+0.607%；odd_last和odd_no_last低于原循环，discard。偶数取样的末条保留效应为正，但奇偶末条效应交互CI跨零；无确认宏观提升或第二轮纯检索机制结论。"
    },
    {
      "family": "equal_count_segment_read_placement",
      "state": "completed",
      "formal_training_completed": 8,
      "results": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/RESULTS.md",
      "kept_research_candidates": [
        "first_odd"
      ],
      "confirmed_parent_improvement": false
    },
    {
      "family": "first_read_moment_second_query",
      "state": "completed",
      "implementation_directory": "workspace/qformer-loop/taobao-transfer/recurrence-moment-query",
      "proposal": "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/proposal.json",
      "protocol": "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/protocol.json",
      "planned_training_runs": 4,
      "formal_training_started": 4,
      "smoke_started": 4,
      "additional_training_seeds": [],
      "diagnostic_task_id": null,
      "diagnostic_candidate_id": null,
      "reason": "Distinct first-read second-moment statistic, not a nearby old mixing/coverage/value residual scan. No evidence from rejected gradient diagnostic used.",
      "falsification": "Reject dispersion explanation unless weighted variance beats mean/uniform variance/constant; retain original mature_r2 quality/cost policy and matched Cross8 comparison.",
      "formal_training_completed": 4,
      "smoke_completed": 4,
      "formal_session": 45912,
      "formal_launcher_pid": 1143932,
      "formal_workers": {
        "weighted_variance": 1143935,
        "weighted_mean": 1143936,
        "uniform_variance": 1143937,
        "constant": 1143938
      },
      "smoke_review": "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/reports/smoke-review.json",
      "smoke_review_sha256": "6f46a4dd8be302af39fa01a634c1acfdc4c44131ed755f1189f5adbfc4818956",
      "formal_authorization_sha256": "3465814805b1291f768da1d0b6a8a44fd4da815abc3aa0bb8cae5f584836d2ff",
      "results": "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/RESULTS.md",
      "interpretation": "四个第一轮统计量→第二轮查询模型均劣于 mature_r2，全部探索性配对 macro 区间为负，推理耗时增加约16%。加权方差优于均匀方差，但与均值/常量的 macro 区间跨零，未建立独立方差条件收益。保留原循环和 Cross8 参考，四组不作为优化候选，不改变已发布协议下的旧 incumbent。"
    },
    {
      "family": "second_work_value_content",
      "state": "completed",
      "protocol": "workspace/qformer-loop/taobao-transfer/recurrence-value-content/protocol.json",
      "hypothesis": "Second-work conditional value content: event versus pooled × entry/work/nonzero constant. No established improvement.",
      "falsification": "Six full-budget fits fail mature_r2 quality/cost policy; source effect must beat entry and constant controls before attributing work conditioning.",
      "budget": {
        "formal_runs": 6,
        "updates_each": 2193,
        "smoke_updates": 12,
        "seed": 20262027
      },
      "diagnostic_task_id": null,
      "diagnostic_candidate_id": null,
      "formal_started": 6,
      "smoke_completed": 6,
      "formal_training_completed": 6,
      "results": "workspace/qformer-loop/taobao-transfer/recurrence-value-content/RESULTS.md",
      "interpretation": "六组均未达到 mature_r2 的质量或近似持平/成本门槛。逐事件工作条件相对入口/常量的 macro 区间跨零；聚合后工作条件相对二者均为负区间。逐事件工作条件优于聚合后工作条件，但仍劣于原循环且更慢；没有建立更好的第二轮信息利用。"
    },
    {
      "experiment": "recurrence-single-history-read",
      "state": "completed",
      "implementation_directory": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read",
      "planned_training_runs": 8,
      "formal_training_started": 8,
      "additional_training_seeds": [],
      "protocol": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/protocol.json",
      "protocol_sha256": "82c32ac9a05cedfdc7b72ab1e76e6846e8c381f8f1c2112ed2514c1224e8596c",
      "diagnostic_task_id": null,
      "diagnostic_candidate_id": null,
      "reference_training_reused": [
        "mature_r2",
        "cross8_mature",
        "hist_shallow_deep"
      ],
      "falsification": "Reject models that fail original macro/cost policy; conditional-gate within-family gains alone do not prove improvement overA.",
      "formal_training_completed": 8,
      "smoke_completed": 8,
      "smoke_review": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/smoke-review.json",
      "smoke_review_sha256": "4c73083f05f4da8c68a870c064a0067ca0dfee1b4fd1a07d2be3388f85e46004",
      "formal_authorization_sha256": "c716eb20655e6fcf06d8f6cd02a6cb57f5ec4fea485910512f7ca9b21285a7d9",
      "results": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/RESULTS.md",
      "interpretation": "All eight single-history variants have lower macro point AUC and higher latency than matched mature R2; discard all as optimization candidates. MLP10 is best new but delta -0.00091332 CI crosses zero and latency +17.25%. Work-gate minus entry-gate macro CI crosses zero, with click/cart/fav AUC deterioration and uncertain buy AUC increase; buy BCE improves within that pair. Preserve A, three references and the old released incumbent; next mechanism remains unverified."
    },
    {
      "experiment": "q4_matched_depth_by_history_maturity",
      "state": "completed",
      "proposed_cells": [
        "released_r1",
        "released_r2",
        "mature_r1",
        "mature_r2"
      ],
      "planned_training_runs": 4,
      "additional_training_seeds": [],
      "priority": "next_after_N_closed",
      "hypothesis": "Does conservatively hiding unmatured cart/fav/buy labels change task-specific independently trained R2-R1 gains under common strict history boundaries?",
      "design_note": "workspace/qformer-loop/taobao-transfer/history-action-availability-audit/NEXT_MATCHED_DEPTH_CONTROL.md",
      "proposed_training_runs": 4,
      "implementation_directory": "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth",
      "scope_note": "workspace/qformer-loop/taobao-transfer/recurrence-task-query/root-scope-and-next.md",
      "formal_training_started": 4,
      "formal_training_completed": 4,
      "results": "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/RESULTS.md"
    },
    {
      "experiment": "history_versions_and_matched_cross8",
      "state": "completed",
      "implementation_directory": "workspace/qformer-loop/taobao-transfer/recurrence-history-versions",
      "planned_training_runs": 4,
      "additional_training_seeds": [],
      "parent_reused": "workspace/qformer-loop/taobao-transfer/recurrence-maturity-depth/runs/mature_r2",
      "experiments": [
        "hist_deep_deep",
        "hist_shallow_deep",
        "hist_deep_shallow",
        "cross8_mature"
      ],
      "interpretation": "All eight fits completed. Mature-minus-released recurrence interaction +0.00303483 CI [0.00004703,0.00629221] is exploratory; mature R2-R1 gain CI crosses zero. SD beats DD/DS but SD-A +0.00028929 CI [-0.00264109,0.00308386], latency ratio1.4665 with click/cart/buy tradeoffs. Keep A and references; retain SD only as a research tradeoff, discard DD/DS and released R2 as candidates. Old protocol incumbent unchanged.",
      "diagnostic_task_id": null,
      "diagnostic_candidate_id": null,
      "formal_training_started": 4,
      "formal_training_completed": 4,
      "results": "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/RESULTS.md"
    },
    {
      "experiment": "native_first_work_update_sample_matching_wq_only",
      "state": "conditional_deferred_low_decision_utility",
      "hypothesis": "Does sample-matched r=z1-e from the original first work update affect the native second retrieval independently of w_qi and residual? One proposed donor intervention on second w_q only; no improved training formula is established.",
      "evidence": [
        "workspace/qformer-loop/taobao-transfer/recurrence-field-role/NEXT_QUERY_CONDITION_AUDIT.md",
        "workspace/qformer-loop/taobao-transfer/recurrence-field-role/RESULTS.md"
      ],
      "requirements": "Scientific utility review completed: defer execution. Any later activation needs a concrete candidate and separately frozen donor/control, native identities, task contrasts and bounded cost.",
      "additional_training_seeds": [],
      "planned_training_runs": 0,
      "diagnostic_task_id": null,
      "diagnostic_candidate_id": null,
      "tool_scope": "No implementation, model/forward/bootstrap/GPU/training, tool candidate or handoff dispatched. Native optimizer shadow remains conditional and untriggered.",
      "utility_review": "workspace/qformer-loop/taobao-transfer/recurrence-field-role/NEXT_DIAGNOSTIC_UTILITY_REVIEW.md",
      "utility_review_sha256": "9c64ea95b5fdd41b3edc57603436c3855774d90074ccd5fc757382c2ffe7c107",
      "reason": "Distinct from prior interventions, but no current concrete Q2 candidate would be selected or rejected by this donor perturbation; conditional dependence alone has not translated into trained improvement.",
      "when": "Reconsider only if an explicit sample-independent first-work-update simplification is proposed; choose a direct candidate-matched intervention then."
    },
    {
      "experiment": "native_optimizer_layout_shadow",
      "state": "conditional_not_dispatched",
      "candidate_id": "probe-302bf4e60eb7a4e126b2d1b1",
      "when": "Before claiming a new-parameter variant preserves native optimizer dynamics",
      "requirements": "Separate native two-condition design and budget;real shadow/utility gates pending;no new training seeds"
    }
  ],
  "user_scope": "Improve second-cycle retrieval conditions using rec-harness diagnostics and memory; fixed two cycles, no prediction-to-work feedback, one training seed, eight GPUs; keep near-baseline efficient candidates.",
  "protocol_caveats": "workspace/qformer-loop/taobao-transfer/reports/protocol.json",
  "test_evaluated": false,
  "tradeoff_candidates": [
    "t3_post_half",
    "t2_cross8_w384_loop1",
    "taobao_cross8_clip1",
    "psgd_cross8_clip1",
    "psgd_post_half",
    "psgd_t3_tail_half",
    "middle_r2",
    "middle_bounded_interaction"
  ],
  "latest_user_priority": "ITT/LoopCTR information flow research used to design history-version controls; preserve mature-input depth interaction and all four tasks, compare matched Cross8, use eight distinct GPUs without new training seeds.",
  "active_round": null,
  "retained_recurrence_candidate": {
    "family": "explore_dual_no_feedback",
    "macro_auc": 0.8285212133742255,
    "reason": "Near best single; not an equivalence claim"
  },
  "retained_small_recurrence_candidate": {
    "family": "ctx_q4",
    "macro_auc": 0.8284725654262436,
    "parameter_count": 51724228,
    "reason": "near_original_recurrence_and_lower_measured_forward_latency_not_statistical_equivalence"
  },
  "phase": "round_accumulation_completed",
  "diagnostic_queue": [],
  "exhausted_families": [
    "tested_ffn_ratio2_8_and_entry3_capacity_sweep",
    "tested_difference_only_rank16_additive_source",
    "tested_q4_work_q8_entry_rank16_route_restrictions",
    "tested_rank16_bound025_entry_work_mixing_q4_q8_channel_query",
    "tested_q4_q8_shared_task_early_final_readout",
    "tested_q4_q8_second_query_norm_full_projection_untie",
    "tested_q4_q8_bound025_norm_preserving_extrapolation_and_key_feedback",
    "tested_q4_q8_strength1_signed_mean_loo_second_read_coverage",
    "tested_q4_q8_second_work_item_action_key_value_field_roles",
    "tested_q4_q8_second_query_shared_post_work_relay_all_own_uniform_other",
    "tested_task_query_bias_q4_q8_shared_task_four_branches"
  ],
  "pending_query_experiment": {
    "parent": "ctx_q4",
    "formula": "q2=(e+z1)/2 + B(GELU(A(LN(z1-e))))",
    "rank": 16,
    "initialization": "B zero; preserve original output; isolate new-module RNG consumption",
    "controls": [
      "same parameter correction conditioned on e",
      "same parameter correction conditioned on z1",
      "original q4 baseline reused"
    ],
    "budget": "Same seed20262027,2193updates,PSGDKron every-step,clip1; required q4 difference retry completed, audited and consolidated; no additional training seeds",
    "falsification": "No macroAUC improvement over q4 and unacceptable task/compute tradeoff => discard; do not infer gain from sensitivity",
    "evidence": "workspace/qformer-loop/taobao-transfer/query-conditioning/reports/paired-interpretation.json",
    "status": "all_eight_sources_completed_including_required_difference_retry"
  },
  "latest_memory_source_outcome": "workspace/qformer-loop/taobao-transfer/recurrence-memory-sources/reports/interpretation.json",
  "query_correction_attribution_scope": "Second-work input correction affects retrieval, interaction and residual; isolate these before assigning retrieval mechanism.",
  "query_correction_diagnostic_review": "workspace/qformer-loop/taobao-transfer/recurrence-query-correction/reports/design-review.json",
  "query_correction_repair": "workspace/qformer-loop/taobao-transfer/recurrence-query-correction-v2/reports/repair-audit.json",
  "pending_query_route_experiment": {
    "folder": "workspace/qformer-loop/taobao-transfer/recurrence-query-routes",
    "sources": {
      "q4": "work",
      "q8": "entry"
    },
    "routes": [
      "retrieval_only",
      "interaction_only",
      "residual_only",
      "attention_only"
    ],
    "state": "completed_all_eight_discarded_by_prospective_queue_policy",
    "evidence": "workspace/qformer-loop/taobao-transfer/query-correction-diagnostics/reports/interpretation.json"
  },
  "active_rounds": [],
  "latest_query_route_outcome": "workspace/qformer-loop/taobao-transfer/recurrence-query-routes/reports/interpretation.json",
  "latest_difference_retry_outcome": "workspace/qformer-loop/taobao-transfer/recurrence-query-difference-retry/reports/interpretation.json",
  "pending_query_mixing_experiment": {
    "folder": "workspace/qformer-loop/taobao-transfer/recurrence-query-mixing",
    "state": "completed_audited_consolidated_interpreted_and_diagnosed",
    "analysis_plan": "workspace/qformer-loop/taobao-transfer/recurrence-query-mixing/PLAN.md",
    "paired_comparisons": 12,
    "additional_training_seeds": [],
    "observed_at_utc": "2026-09-09T04:32:13.100781+00:00",
    "continuation": "workspace/qformer-loop/taobao-transfer/recurrence-query-mixing/reports/continuation.json",
    "consolidation_entry": [
      "workspace/ai4ai-runtime/opd-native-env/bin/python",
      "-B",
      "workspace/qformer-loop/taobao-transfer/recurrence-query-mixing/verify_consolidation_inputs.py",
      "--run-consolidation"
    ],
    "interpretation_plan": "workspace/qformer-loop/taobao-transfer/recurrence-query-mixing/reports/interpretation-plan.json",
    "quality_summary": "workspace/qformer-loop/taobao-transfer/recurrence-query-mixing/reports/quality-summary.json",
    "training_completed_not_yet_in_ledger": 0,
    "interpretation": "workspace/qformer-loop/taobao-transfer/recurrence-query-mixing/reports/interpretation.json",
    "diagnostic_outcome": "workspace/qformer-loop/taobao-transfer/query-mixing-diagnostics/reports/interpretation.json"
  },
  "latest_query_mixing_outcome": "workspace/qformer-loop/taobao-transfer/recurrence-query-mixing/reports/interpretation.json",
  "query_mixing_optimization_candidates": [],
  "query_mixing_quality_compute_tradeoffs": [],
  "latest_query_mixing_diagnostic_outcome": "workspace/qformer-loop/taobao-transfer/query-mixing-diagnostics/reports/interpretation.json",
  "latest_second_read_attention_outcome": "workspace/qformer-loop/taobao-transfer/memory/tool-evolution/candidates/taobao-second-read-attention-v1/artifacts/outcome.json",
  "second_read_attention_constraint": {
    "source_episode": "taobao-second-read-attention-originals-5514",
    "candidate_id": "taobao-second-read-attention-v1",
    "preserve_q4_buy_second_probability_redistribution": true,
    "other_7_tasks": "inconclusive",
    "no_reuse_or_novelty_training_from_this_probe": true,
    "new_normalization_architecture": "separate research hypothesis, not a claimed consequence or improvement from this diagnostic"
  },
  "backup_research": "workspace/qformer-loop/taobao-transfer/recurrence-research/ROUND_SPECIFIC_RETRIEVAL_NOTE.md",
  "latest_task_effects_outcome": "workspace/qformer-loop/taobao-transfer/recurrence-task-effects/reports/interpretation.json",
  "last_consolidated_episode": "round-accumulation--distinct_no_carry--s20262027",
  "task_effects_constraint": {
    "only_some_tasks_claim_supported": false,
    "task_effect_difference_exploratory": true,
    "metric_gain_is_not_literal_external_information": true,
    "original_readout_complementarity_unverified": true,
    "task_specific_early_final_readout_tested_no_gain": true
  },
  "pending_readout_retrieval": {
    "folder": "workspace/qformer-loop/taobao-transfer/recurrence-readout-retrieval",
    "protocol": "workspace/qformer-loop/taobao-transfer/recurrence-readout-retrieval/protocol.json",
    "state": "completed_audited_consolidated_all_eight_discarded",
    "planned_training_runs": 8,
    "started_training_runs": 8,
    "formal_tool_session": 80732,
    "smoke_review": "workspace/qformer-loop/taobao-transfer/recurrence-readout-retrieval/reports/smoke-review.json",
    "postprocess_identity": "workspace/qformer-loop/taobao-transfer/recurrence-readout-retrieval/reports/postprocess-identity.json",
    "postprocessing_state": "completed_all_ten_pairs_eleven_latency_two_readout_diagnostics",
    "training_completed_not_yet_in_ledger": 0,
    "formal_tool_session_exit_code": 0,
    "quality_audit_session": 63217,
    "interpretation": "workspace/qformer-loop/taobao-transfer/recurrence-readout-retrieval/reports/interpretation.json"
  },
  "gpu_execution_constraint": "Launch GPU model subprocesses with approved require_escalated execution; sandbox nvidia-smi subprocess returns driver error9 although direct tool nvidia-smi can work.",
  "latest_readout_retrieval_outcome": "workspace/qformer-loop/taobao-transfer/recurrence-readout-retrieval/reports/interpretation.json",
  "readout_retrieval_optimization_candidates": [],
  "latest_optimizer_layout_outcome": "workspace/qformer-loop/taobao-transfer/optimizer-layout-replay/reports/interpretation.json",
  "latest_direction_norm_outcome": "workspace/qformer-loop/taobao-transfer/recurrence-query-direction-norm/reports/interpretation.json",
  "direction_norm_constraint": {
    "preserve_q4_buy_second_direction": true,
    "epsilon": 1e-06,
    "scope": "Specified first-direction/first-norm replacements on original fixed checkpoint/subset",
    "other_7_joint_gates": "not_passed_not_no_benefit",
    "no_new_training_candidate": true,
    "same_saved_native_and_reuse_replicated": true
  },
  "pending_directional_retrieval": {
    "folder": "workspace/qformer-loop/taobao-transfer/recurrence-directional-retrieval-v2",
    "protocol": "workspace/qformer-loop/taobao-transfer/recurrence-directional-retrieval-v2/protocol.json",
    "state": "completed_audited_interpreted_consolidated",
    "planned_training_runs": 8,
    "started_training_runs": 8,
    "smoke_tool_session": 82983,
    "parent_failed_smoke": "workspace/qformer-loop/taobao-transfer/recurrence-directional-retrieval/reports/implementation-failure.json",
    "postprocess_identity": "workspace/qformer-loop/taobao-transfer/recurrence-directional-retrieval-v2/reports/postprocess-identity.json",
    "new_training_seeds": [],
    "training_completed_not_yet_in_ledger": 0,
    "formal_tool_session": 35368,
    "smoke_tool_exit_code": 0,
    "smoke_review": "workspace/qformer-loop/taobao-transfer/recurrence-directional-retrieval-v2/reports/smoke-review.json",
    "postprocessing_state": "all_twelve_pairs_and_eleven_latency_models_completed",
    "implementation_failures_in_ledger": 6,
    "implementation_failure_episode_ids": [
      "dir_q4_uniform_key-s20262027-20260909T092130215879Z-smoke-failed",
      "dir_q4_attended_key-s20262027-20260909T092130157444Z-smoke-failed",
      "dir_q4_centered_key-s20262027-20260909T092130159863Z-smoke-failed",
      "dir_q8_uniform_key-s20262027-20260909T092130151184Z-smoke-failed",
      "dir_q8_attended_key-s20262027-20260909T092130167891Z-smoke-failed",
      "dir_q8_centered_key-s20262027-20260909T092130154504Z-smoke-failed"
    ],
    "formal_tool_exit_code": 0,
    "quality_audit_session": 2003,
    "quality_audit_exit_code": 0,
    "paired_tool_session": 4422,
    "latency_tool_session": 2586,
    "source_point_outcome": "all_eight_below_corresponding_original_macro_auc; fixed policy yields no retained candidate from points",
    "paired_tool_exit_code": 0,
    "latency_tool_exit_code": 0,
    "interpretation": "workspace/qformer-loop/taobao-transfer/recurrence-directional-retrieval-v2/reports/interpretation.json"
  },
  "latest_directional_retrieval_outcome": "workspace/qformer-loop/taobao-transfer/recurrence-directional-retrieval-v2/reports/interpretation.json",
  "directional_retrieval_optimization_candidates": [],
  "directional_retrieval_constraint": "q4/q8 analyzed separately; user-paired intervals exclude training-seed variance and selection/multiplicity correction. No posthoc click target switch, only-buy rule, or general no-recurrence conclusion.",
  "pending_coverage_retrieval": {
    "folder": "workspace/qformer-loop/taobao-transfer/recurrence-coverage-retrieval",
    "state": "completed_audited_interpreted_consolidated",
    "protocol": "workspace/qformer-loop/taobao-transfer/recurrence-coverage-retrieval/protocol.json",
    "planned_training_runs": 8,
    "started_training_runs": 8,
    "completed_training_runs": 8,
    "training_completed_not_yet_in_ledger": 0,
    "new_training_seeds": [],
    "test_evaluated": false,
    "smoke_tool_session": 98773,
    "native_fixture_tool_session": 46593,
    "native_fixture_exit_code": 0,
    "frozen_identity": "workspace/qformer-loop/taobao-transfer/recurrence-coverage-retrieval/reports/frozen-identity.json",
    "postprocess_identity": "workspace/qformer-loop/taobao-transfer/recurrence-coverage-retrieval/reports/postprocess-identity.json",
    "postprocessing_state": "all_16_pairs_and_eleven_latency_models_completed",
    "native_independent_review": "workspace/qformer-loop/taobao-transfer/recurrence-coverage-retrieval/reports/native-gate-independent-review.json",
    "formal_tool_session": 49697,
    "smoke_tool_exit_code": 0,
    "smoke_review": "workspace/qformer-loop/taobao-transfer/recurrence-coverage-retrieval/reports/smoke-review.json",
    "smoke_independent_review": "workspace/qformer-loop/taobao-transfer/recurrence-coverage-retrieval/reports/smoke-independent-review.json",
    "formal_tool_exit_code": 0,
    "formal_completed_observed_at": "2026-09-09T11:51:54.053455+00:00",
    "quality_audit_tool_session": 16421,
    "quality_audit_tool_exit_code": 0,
    "paired_tool_session": 10337,
    "latency_tool_session": 6254,
    "latency_tool_exit_code": 0,
    "quality_summary_sha256": "ab3e9cad65194b7d34656032a6f18736c72826ecb77cb106797a588eb558bb05",
    "latency_completion_sha256": "99875d4e84ac6262cc4594bde6f006e127f5b6cac0d67f1dcf0c471b91cabb69",
    "interpretation": "workspace/qformer-loop/taobao-transfer/recurrence-coverage-retrieval/reports/interpretation.json"
  },
  "latest_coverage_retrieval_outcome": "workspace/qformer-loop/taobao-transfer/recurrence-coverage-retrieval/reports/interpretation.json",
  "coverage_retrieval_optimization_candidates": [],
  "pending_field_role": {
    "folder": "workspace/qformer-loop/taobao-transfer/recurrence-field-role",
    "state": "completed_audited_interpreted_consolidated",
    "protocol": "workspace/qformer-loop/taobao-transfer/recurrence-field-role/protocol.json",
    "planned_training_runs": 8,
    "started_training_runs": 8,
    "completed_training_runs": 8,
    "training_completed_not_yet_in_ledger": 0,
    "smokes_started": 8,
    "new_training_seeds": [],
    "test_evaluated": false,
    "registered_at": "2026-09-09T12:42:16.118079+00:00",
    "code_cache_identity": "workspace/qformer-loop/taobao-transfer/recurrence-field-role/reports/code-cache-identity.json",
    "code_cache_import_tool_session": 12450,
    "code_cache_import_exit_code": 0,
    "protocol_frozen": true,
    "diagnostic_task_id": null,
    "smoke_tool_session": 50500,
    "frozen_identity": "workspace/qformer-loop/taobao-transfer/recurrence-field-role/reports/frozen-identity.json",
    "frozen_identity_sha256": "fd4864b7cee9ff535ab5c5832720ca34067cfda82820616bc1b44a8e61584c92",
    "native_fixture_tool_session": 19834,
    "native_fixture_exit_code": 0,
    "native_independent_review": "workspace/qformer-loop/taobao-transfer/recurrence-field-role/reports/native-gate-independent-review.json",
    "formal_tool_session": 81694,
    "smoke_tool_exit_code": 0,
    "smokes_completed": 8,
    "smoke_review": "workspace/qformer-loop/taobao-transfer/recurrence-field-role/reports/smoke-review.json",
    "smoke_independent_review": "workspace/qformer-loop/taobao-transfer/recurrence-field-role/reports/smoke-independent-review.json",
    "postprocess_identity": "workspace/qformer-loop/taobao-transfer/recurrence-field-role/reports/postprocess-identity.json",
    "postprocessing_state": "all_16_pairs_eleven_latency_models_and_42_combination_intervals_completed",
    "formal_started_at": "2026-09-09T13:01:33.334022+00:00",
    "formal_tool_exit_code": 0,
    "formal_completed_observed_at": "2026-09-09T14:05:50.941207+00:00",
    "quality_audit_tool_session": 21191,
    "quality_audit_tool_exit_code": 0,
    "quality_summary": "workspace/qformer-loop/taobao-transfer/recurrence-field-role/reports/quality-summary.json",
    "quality_summary_sha256": "03f49184cbdfde17bf200dae10caa28b5a2619c9d9aa9ba88bd2daf526927af5",
    "paired_analysis_tool_session": 65918,
    "latency_tool_session": 87225,
    "analysis_started_at": "2026-09-09T14:07:36.253584+00:00",
    "interpretation": "workspace/qformer-loop/taobao-transfer/recurrence-field-role/reports/interpretation.json"
  },
  "latest_field_role_outcome": "workspace/qformer-loop/taobao-transfer/recurrence-field-role/reports/interpretation.json",
  "field_role_optimization_candidates": [],
  "latest_query_condition_utility_review": "workspace/qformer-loop/taobao-transfer/recurrence-field-role/NEXT_DIAGNOSTIC_UTILITY_REVIEW.md",
  "pending_work_relay": {
    "state": "completed_audited_interpreted_consolidated",
    "planned_training_runs": 8,
    "formal_training_started": 8,
    "completed_training_runs": 8,
    "training_completed_not_yet_in_ledger": 0,
    "code_cache_import_session": 82474,
    "code_cache_state": "completed",
    "gpu_or_training_dispatched": true,
    "smokes_started": 8,
    "smoke_session": 33355,
    "native_stage_gate": "passed",
    "native_stage_gate_session": 22706,
    "source_freeze_sha256": "74509817a0a7699c51e3033ed830c9eac937e35d921e09edca94790738ebbfe5",
    "postprocess_freeze_sha256": "6c7efeb07c9e5159d4f14d3ee516dbaac9d89c7cc47304e46d9f674597628f0a",
    "smokes_completed": 8,
    "smoke_exit_code": 0,
    "formal_session": 86340,
    "smoke_independent_review": "workspace/qformer-loop/taobao-transfer/recurrence-work-relay/reports/smoke-independent-review.json",
    "formal_exit_code": 0,
    "formal_completion_chunk": "29da46",
    "postprocessing_state": "all_14_pairs_and_eleven_latency_models_completed",
    "audit_session": 13462,
    "interpretation": "workspace/qformer-loop/taobao-transfer/recurrence-work-relay/reports/interpretation.json"
  },
  "latest_work_relay_outcome": "workspace/qformer-loop/taobao-transfer/recurrence-work-relay/reports/interpretation.json",
  "work_relay_optimization_candidates": [],
  "pending_task_query": {
    "state": "completed_audited_interpreted_consolidated",
    "planned_training_runs": 4,
    "formal_training_started": 4,
    "completed_training_runs": 4,
    "training_completed_not_yet_in_ledger": 0,
    "smokes_started": 4,
    "smokes_completed": 4,
    "native_stage_gate": "passed",
    "code_cache_state": "completed",
    "gpu_or_training_dispatched": true,
    "additional_training_seeds": [],
    "model_offline": "workspace/qformer-loop/taobao-transfer/recurrence-task-query/reports/model-offline.json",
    "runner_offline": "workspace/qformer-loop/taobao-transfer/recurrence-task-query/reports/runner-offline.json",
    "protocol": "workspace/qformer-loop/taobao-transfer/recurrence-task-query/protocol.json",
    "smoke_session": 44894,
    "native_stage_gate_session": 53158,
    "code_cache_import_session": 18918,
    "source_freeze_sha256": "98fffb3abaaa5556323f5928779ad6dd2e73389268f5238aa2dbb0201921371c",
    "postprocess_freeze_sha256": "8124560a1c3358002330fe5993c4d6afd1223433cc01e0469777609263643450",
    "smoke_exit_code": 0,
    "formal_session": 7977,
    "formal_workers": {
      "taskq_q4_shared": 933637,
      "taskq_q4_task": 933638,
      "taskq_q8_shared": 933639,
      "taskq_q8_task": 933640
    },
    "smoke_independent_review": "workspace/qformer-loop/taobao-transfer/recurrence-task-query/reports/smoke-independent-review.json",
    "smoke_independent_review_sha256": "b1ac5b1aea19f548bd3ec697fb61b9f9fe7697e8699ad447fe2a35a9807ee4d2",
    "interpretation_scope": "Released offline features only; close all four frozen runs and paired/cost analysis, with the independently audited historical-action maturity limitation attached.",
    "postprocessing_state": "all_6_pairs_and_seven_latency_models_completed",
    "interpretation": "workspace/qformer-loop/taobao-transfer/recurrence-task-query/reports/interpretation.json"
  },
  "latest_history_action_availability_audit": {
    "state": "completed_independently_reviewed",
    "role": "auxiliary_data_quality_audit",
    "directory": "workspace/qformer-loop/taobao-transfer/history-action-availability-audit",
    "evidence_sha256": {
      "workspace/qformer-loop/taobao-transfer/history-action-availability-audit/RESULTS.md": "fc5c785cb5e5fdeb0c17cdec21a8201750cc285d2d2fcefd7de662653531fcdf",
      "workspace/qformer-loop/taobao-transfer/history-action-availability-audit/audit_sample.py": "a10adcbbb365c99d4b4c0362bd26fd69bef94bd7e0b3453e0fab57301ceb930a",
      "workspace/qformer-loop/taobao-transfer/history-action-availability-audit/footer-plan.json": "0660abf218510b6aaf24dbe6a0d1457c5850e045437576329998836fe149a128",
      "workspace/qformer-loop/taobao-transfer/history-action-availability-audit/reports/source-review.json": "132c37eb3c989cd660fe3d7c82d439d3670b3bd950e62138ebf2949cb1919e2e",
      "workspace/qformer-loop/taobao-transfer/history-action-availability-audit/reports/result-independent-review.json": "d6ce9522986b211a3bdf6c553120c37ae4f580de84e1651f7ecef56411d82a79",
      "workspace/qformer-loop/taobao-transfer/history-action-availability-audit/reports/sample-1024/summary.json": "8838de60ebfcf0cb567a5fb1f09feb9f861c3ec4c17ff435adbf4be32d59d7bd",
      "workspace/qformer-loop/taobao-transfer/history-action-availability-audit/reports/sample-1024/completion.json": "668bc01f25aab67c937f728eb28b3592e376026e907d741e44a94c7eb733d3da",
      "workspace/qformer-loop/taobao-transfer/history-action-availability-audit/reports/sample-execution-identity.json": "78f42c0d5bd9c2bed566b3bbd9599444329153441ce91898f48e3b7d84221281",
      "workspace/qformer-loop/taobao-transfer/history-action-availability-audit/reports/sample-execution-completion.json": "06a3e0baf02409be44a0a4a42eed2d486208013b8c3fc41e48d6c3332933aa0f",
      "workspace/qformer-loop/taobao-transfer/history-action-availability-audit/MATURITY_CONTROL_FEASIBILITY.md": "06da69ebb4f1110e558b4d9cf06b3f671518c699a0c8ef2f3a3d8ab9e524012f",
      "workspace/qformer-loop/taobao-transfer/recurrence-research/TASKWISE_RECURRENCE_EVIDENCE.md": "860520cc9694b07154465c3cce2796ac37d65cd69a4959b308377c5ce731b7b0"
    },
    "sample_rows": 1024,
    "history_occurrences": 62140,
    "unfinished_window_occurrences": 30475,
    "unfinished_window_fraction": 0.4904248471194078,
    "target_rows_with_unfinished_window": 980,
    "target_row_fraction": 0.95703125,
    "same_timestamp_history_occurrences": 2704,
    "target_rows_with_same_timestamp_history": 795,
    "future_timestamp_history_occurrences": 0,
    "exact_window_boundary_occurrences": 0,
    "behavior_tasks": [
      "cart",
      "fav",
      "buy"
    ],
    "window_seconds": 86400,
    "click_scope": "Co-occurrence only; click is not the same 24h behavior-window label.",
    "validity_boundary": "Fixed systematic validation sample of history occurrences; candidate unmatured windows are not actual leakage rate, affected-prediction rate, causal AUC harm or a population estimate. Matched behavior-event and click-arrival times are unavailable; target time has no independent row timestamp check.",
    "prior_results_scope": "Existing comparisons remain unchanged outcomes under the released offline feature protocol; common inputs do not establish online availability or causal retrieval information gain.",
    "execution": {
      "tool_session": 14202,
      "tool_exit_code": 0,
      "tool_completion_chunk": "de370d",
      "wall_seconds": 1.7158713936805725,
      "child_cpu_seconds": 1.670418,
      "child_peak_rss_bytes": 147333120
    },
    "parquet_api_read_bytes": 134823983,
    "column_row_group_reads": 64,
    "model_forwards": 0,
    "training_updates": 0,
    "additional_training_seeds": [],
    "test_split_files_read": false,
    "accounting": "Auxiliary input-quality evidence is registered here with complete receipts; no new formal-training experiment or ledger episode/TSV row, and no new lesson from one audit."
  },
  "history_action_availability_constraint": "Before interpreting task-specific loop gains as available-history information, use a matched maturity control. Pause new action-label score-prior architectures. Keep unknown distinct from mature zero; preserve original results, seed and data files.",
  "latest_task_query_outcome": "workspace/qformer-loop/taobao-transfer/recurrence-task-query/reports/interpretation.json",
  "task_query_optimization_candidates": [],
  "latest_matched_maturity_history_outcome": {
    "summary": "All eight fits completed. Mature-minus-released recurrence interaction +0.00303483 CI [0.00004703,0.00629221] is exploratory; mature R2-R1 gain CI crosses zero. SD beats DD/DS but SD-A +0.00028929 CI [-0.00264109,0.00308386], latency ratio1.4665 with click/cart/buy tradeoffs. Keep A and references; retain SD only as a research tradeoff, discard DD/DS and released R2 as candidates. Old protocol incumbent unchanged.",
    "formal_training_runs": 8,
    "report": "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/RESULTS.md",
    "paired": "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/reports/joint-paired/result.json",
    "latency": "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/reports/latency/completion.json",
    "candidates": {
      "released_r1": {
        "parent": null,
        "effect": null,
        "decision": "retain_reference",
        "reason": "Strict-past released-label matched R1 reference; do not compare directly to old release protocol.",
        "hypothesis": "Recurrence gain may depend on action maturity under matched strict history support."
      },
      "released_r2": {
        "parent": "released_r1",
        "effect": "D_released",
        "decision": "discard",
        "reason": "R2-R1 macro estimate and paired interval are negative; keep the completed result as evidence.",
        "hypothesis": "Recurrence gain may depend on action maturity under matched strict history support."
      },
      "mature_r1": {
        "parent": null,
        "effect": null,
        "decision": "retain_reference",
        "reason": "Matched maturity/depth R1 reference; R2 gain interval crosses zero.",
        "hypothesis": "Recurrence gain may depend on action maturity under matched strict history support."
      },
      "mature_r2": {
        "parent": "mature_r1",
        "effect": "D_mature",
        "decision": "retain_protocol_anchor",
        "reason": "Keep the existing mature-input recurrent anchor; no confirmed SD macro improvement and lower cost than SD.",
        "hypothesis": "Recurrence gain may depend on action maturity under matched strict history support."
      },
      "hist_deep_deep": {
        "parent": "mature_r2",
        "effect": "hist_deep_deep__minus__mature_r2",
        "decision": "discard",
        "reason": "Macro regression versus A has a negative paired interval and latency rises 46.63%.",
        "hypothesis": "The depth and order of history representations may change second-read utility; compare against matched Cross8."
      },
      "hist_shallow_deep": {
        "parent": "mature_r2",
        "effect": "hist_shallow_deep__minus__mature_r2",
        "decision": "retain_research_tradeoff",
        "reason": "Positive macro point meets candidate policy but interval crosses zero; latency rises 46.65%, cart/buy point estimates decline; no promotion.",
        "hypothesis": "The depth and order of history representations may change second-read utility; compare against matched Cross8."
      },
      "hist_deep_shallow": {
        "parent": "mature_r2",
        "effect": "hist_deep_shallow__minus__mature_r2",
        "decision": "discard",
        "reason": "Macro regression versus A has a negative paired interval and latency rises 46.35%.",
        "hypothesis": "The depth and order of history representations may change second-read utility; compare against matched Cross8."
      },
      "cross8_mature": {
        "parent": null,
        "effect": null,
        "decision": "retain_reference",
        "reason": "Fresh matched Cross8 baseline; architecture/width differences prevent isolating recurrence from A-X.",
        "hypothesis": "The depth and order of history representations may change second-read utility; compare against matched Cross8."
      }
    },
    "next_design": "workspace/qformer-loop/taobao-transfer/recurrence-history-versions/NEXT_EXPERIMENT.md",
    "new_supported_lessons": 0,
    "old_incumbent_changed": false
  },
  "latest_single_history_outcome": {
    "summary": "All eight single-history variants have lower macro point AUC and higher latency than matched mature R2; discard all as optimization candidates. MLP10 is best new but delta -0.00091332 CI crosses zero and latency +17.25%. Work-gate minus entry-gate macro CI crosses zero, with click/cart/fav AUC deterioration and uncertain buy AUC increase; buy BCE improves within that pair. Preserve A, three references and the old released incumbent; next mechanism remains unverified.",
    "formal_training_runs": 8,
    "reused_reference_training_runs": 3,
    "report": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/RESULTS.md",
    "paired": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/paired-analysis/result.json",
    "latency": "workspace/qformer-loop/taobao-transfer/recurrence-single-history-read/reports/latency/completion.json",
    "candidates": {
      "sa_01": {
        "parent": "mature_r2",
        "effect": "sa_01__minus__mature_r2",
        "decision": "discard",
        "reason": "Macro delta versus A -0.00245567, exploratory CI [-0.00498069,-0.00000191], forward latency ratio 1.238432. Fails both predeclared quality/near-candidate rules; retain all artifacts as negative research evidence, with no claim that every interval establishes harm.",
        "hypothesis": "One attention encoding used only by the second work read can improve retrieval."
      },
      "sa_11": {
        "parent": "mature_r2",
        "effect": "sa_11__minus__mature_r2",
        "decision": "discard",
        "reason": "Macro delta versus A -0.00214114, exploratory CI [-0.00474877,+0.00039578], forward latency ratio 1.233557. Fails both predeclared quality/near-candidate rules; retain all artifacts as negative research evidence, with no claim that every interval establishes harm.",
        "hypothesis": "One attention encoding reused by both work reads can improve retrieval."
      },
      "sa_10": {
        "parent": "mature_r2",
        "effect": "sa_10__minus__mature_r2",
        "decision": "discard",
        "reason": "Macro delta versus A -0.00194309, exploratory CI [-0.00499568,+0.00060710], forward latency ratio 1.240588. Fails both predeclared quality/near-candidate rules; retain all artifacts as negative research evidence, with no claim that every interval establishes harm.",
        "hypothesis": "One attention encoding used by the first work read while the second returns to H0 can improve retrieval."
      },
      "mlp_01": {
        "parent": "mature_r2",
        "effect": "mlp_01__minus__mature_r2",
        "decision": "discard",
        "reason": "Macro delta versus A -0.00783142, exploratory CI [-0.01137708,-0.00409488], forward latency ratio 1.173115. Fails both predeclared quality/near-candidate rules; retain all artifacts as negative research evidence, with no claim that every interval establishes harm.",
        "hypothesis": "A tokenwise MLP can supply useful second-read history content at lower cost than history attention."
      },
      "mlp_11": {
        "parent": "mature_r2",
        "effect": "mlp_11__minus__mature_r2",
        "decision": "discard",
        "reason": "Macro delta versus A -0.00573898, exploratory CI [-0.00935142,-0.00235989], forward latency ratio 1.173186. Fails both predeclared quality/near-candidate rules; retain all artifacts as negative research evidence, with no claim that every interval establishes harm.",
        "hypothesis": "Both work reads can benefit from one tokenwise history encoding."
      },
      "mlp_10": {
        "parent": "mature_r2",
        "effect": "mlp_10__minus__mature_r2",
        "decision": "discard",
        "reason": "Macro delta versus A -0.00091332, exploratory CI [-0.00361977,+0.00169206], forward latency ratio 1.172513. Fails both predeclared quality/near-candidate rules; retain all artifacts as negative research evidence, with no claim that every interval establishes harm.",
        "hypothesis": "Encoding the first work history only can improve the representation used to form the second query."
      },
      "mlp_gate_entry": {
        "parent": "mature_r2",
        "effect": "mlp_gate_entry__minus__mature_r2",
        "decision": "discard",
        "reason": "Macro delta versus A -0.00377920, exploratory CI [-0.00735817,-0.00072761], forward latency ratio 1.329651. Fails both predeclared quality/near-candidate rules; retain all artifacts as negative research evidence, with no claim that every interval establishes harm.",
        "hypothesis": "Entry-conditioned modulation can make the second-read history increment useful."
      },
      "mlp_gate_work": {
        "parent": "mature_r2",
        "effect": "mlp_gate_work__minus__mature_r2",
        "decision": "discard",
        "reason": "Macro delta versus A -0.00590641, exploratory CI [-0.00892655,-0.00283662], forward latency ratio 1.330854. Fails both predeclared quality/near-candidate rules; retain all artifacts as negative research evidence, with no claim that every interval establishes harm.",
        "hypothesis": "First-work-conditioned modulation can improve second-read history relative to entry conditioning."
      }
    },
    "new_supported_lessons": 0,
    "old_incumbent_changed": false
  },
  "latest_value_content_outcome": {
    "summary": "六组均未达到 mature_r2 的质量或近似持平/成本门槛。逐事件工作条件相对入口/常量的 macro 区间跨零；聚合后工作条件相对二者均为负区间。逐事件工作条件优于聚合后工作条件，但仍劣于原循环且更慢；没有建立更好的第二轮信息利用。",
    "formal_training_runs": 6,
    "reused_reference_training_runs": 2,
    "report": "workspace/qformer-loop/taobao-transfer/recurrence-value-content/RESULTS.md",
    "paired": "workspace/qformer-loop/taobao-transfer/recurrence-value-content/reports/paired-analysis/result.json",
    "latency": "workspace/qformer-loop/taobao-transfer/recurrence-value-content/reports/latency/completion.json",
    "candidates": {
      "event_entry": {
        "parent": "mature_r2",
        "effect": "event_entry__minus__mature_r2",
        "decision": "discard",
        "reason": "Macro delta -0.004339360, CI95 [-0.00789869835970372, -0.0008016800889363466], forward ratio 1.113860; misses both frozen quality and near policy.",
        "hypothesis": "Second-work value supplementation; event_entry conditional nonlinear content before/after original weighted aggregation."
      },
      "event_work": {
        "parent": "mature_r2",
        "effect": "event_work__minus__mature_r2",
        "decision": "discard",
        "reason": "Macro delta -0.002905209, CI95 [-0.005597721227477079, -0.00040624993050394914], forward ratio 1.116838; misses both frozen quality and near policy.",
        "hypothesis": "Second-work value supplementation; event_work conditional nonlinear content before/after original weighted aggregation."
      },
      "event_constant": {
        "parent": "mature_r2",
        "effect": "event_constant__minus__mature_r2",
        "decision": "discard",
        "reason": "Macro delta -0.004475306, CI95 [-0.007605460367732112, -0.0014428959967006203], forward ratio 1.114353; misses both frozen quality and near policy.",
        "hypothesis": "Second-work value supplementation; event_constant conditional nonlinear content before/after original weighted aggregation."
      },
      "pooled_entry": {
        "parent": "mature_r2",
        "effect": "pooled_entry__minus__mature_r2",
        "decision": "discard",
        "reason": "Macro delta -0.003351628, CI95 [-0.00633145636213481, -0.0003471266979163905], forward ratio 0.999264; misses both frozen quality and near policy.",
        "hypothesis": "Second-work value supplementation; pooled_entry conditional nonlinear content before/after original weighted aggregation."
      },
      "pooled_work": {
        "parent": "mature_r2",
        "effect": "pooled_work__minus__mature_r2",
        "decision": "discard",
        "reason": "Macro delta -0.007217189, CI95 [-0.010118841791676636, -0.0042218699572851605], forward ratio 1.002274; misses both frozen quality and near policy.",
        "hypothesis": "Second-work value supplementation; pooled_work conditional nonlinear content before/after original weighted aggregation."
      },
      "pooled_constant": {
        "parent": "mature_r2",
        "effect": "pooled_constant__minus__mature_r2",
        "decision": "discard",
        "reason": "Macro delta -0.004223502, CI95 [-0.007324589522790518, -0.000923203284268388], forward ratio 0.999204; misses both frozen quality and near policy.",
        "hypothesis": "Second-work value supplementation; pooled_constant conditional nonlinear content before/after original weighted aggregation."
      }
    },
    "new_supported_lessons": 0,
    "old_incumbent_changed": false
  },
  "latest_call_gradient_diagnostic": {
    "candidate_id": "probe-44b522d096f9a98b62e8ba95",
    "status": "rejected",
    "review": "workspace/qformer-loop/taobao-transfer/recurrence-gradient-credit/reports/root-shadow-review.json",
    "scope": "16 forwards/64 backwards completed; terminal iterator.close error prevented final restoration audit. No supported query opposition, insufficient buy positives. No handoff/replay/model conclusion."
  },
  "latest_moment_query_outcome": {
    "summary": "四个第一轮统计量→第二轮查询模型均劣于 mature_r2，全部探索性配对 macro 区间为负，推理耗时增加约16%。加权方差优于均匀方差，但与均值/常量的 macro 区间跨零，未建立独立方差条件收益。保留原循环和 Cross8 参考，四组不作为优化候选，不改变已发布协议下的旧 incumbent。",
    "formal_training_runs": 4,
    "reused_reference_training_runs": 2,
    "report": "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/RESULTS.md",
    "paired": "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/reports/paired-analysis/result.json",
    "latency": "workspace/qformer-loop/taobao-transfer/recurrence-moment-query/reports/latency/completion.json",
    "candidates": {
      "weighted_variance": {
        "parent": "mature_r2",
        "effect": "weighted_variance__minus__mature_r2",
        "decision": "discard",
        "reason": "Macro delta -0.002997810, CI95 [-0.005467256714010241, -0.0008875458971996674]; forward ratio 1.164332. Fails fixed positive/near quality-cost policy.",
        "hypothesis": "First-read weighted_variance projected into second-work w_q before q_norm may improve addressing under fixed two-cycle mature protocol."
      },
      "weighted_mean": {
        "parent": "mature_r2",
        "effect": "weighted_mean__minus__mature_r2",
        "decision": "discard",
        "reason": "Macro delta -0.005301424, CI95 [-0.008202795852167826, -0.0027167669438501324]; forward ratio 1.161172. Fails fixed positive/near quality-cost policy.",
        "hypothesis": "First-read weighted_mean projected into second-work w_q before q_norm may improve addressing under fixed two-cycle mature protocol."
      },
      "uniform_variance": {
        "parent": "mature_r2",
        "effect": "uniform_variance__minus__mature_r2",
        "decision": "discard",
        "reason": "Macro delta -0.006015812, CI95 [-0.008745644480710763, -0.003450261280446835]; forward ratio 1.162326. Fails fixed positive/near quality-cost policy.",
        "hypothesis": "First-read uniform_variance projected into second-work w_q before q_norm may improve addressing under fixed two-cycle mature protocol."
      },
      "constant": {
        "parent": "mature_r2",
        "effect": "constant__minus__mature_r2",
        "decision": "discard",
        "reason": "Macro delta -0.004005424, CI95 [-0.0072340066829167556, -0.0012263822228989322]; forward ratio 1.161950. Fails fixed positive/near quality-cost policy.",
        "hypothesis": "First-read constant projected into second-work w_q before q_norm may improve addressing under fixed two-cycle mature protocol."
      }
    },
    "new_supported_lessons": 0,
    "old_incumbent_changed": false
  },
  "segment_read_queue": {
    "state": "completed",
    "variants": [
      "first_prefix",
      "first_suffix",
      "first_even",
      "first_odd",
      "second_prefix",
      "second_suffix",
      "second_even",
      "second_odd"
    ],
    "formal_runs": 8,
    "formal_updates_each": 2193,
    "training_seed": 20262027,
    "additional_training_seeds": [],
    "proposal": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/proposal.json",
    "smoke_authorization": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/smoke-authorization.json",
    "formal_requires_all_native_smokes": true,
    "formal_session_id": 95487,
    "formal_launcher_pid": 1176530,
    "smokes_completed": 8,
    "smoke_updates": 16,
    "initializer_sha256": "fd5f866c35c5e29b37841aeed070f6ef5259c773de173df7549d149f87a8c09e",
    "formal_training_completed": 8,
    "results": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/RESULTS.md"
  },
  "latest_segment_read_outcome": {
    "summary": "八组中仅 first_odd 满足既定正macro点估计与成本门槛，保留研究候选而非确认改进。相对原循环+0.001237919，区间跨零，点击AUC/BCE退步；相对第二轮同样筛选+0.005011363且区间全正，主要体现收藏收益。其余七组未达到正点估计或近似持平门槛，discard。未证实纯第二轮查询中介、时间近期性或跨seed稳定收益。",
    "formal_training_runs": 8,
    "reused_reference_training_runs": 2,
    "report": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/RESULTS.md",
    "paired": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/paired-analysis/result.json",
    "latency": "workspace/qformer-loop/taobao-transfer/recurrence-segment-read/reports/latency/completion.json",
    "candidates": {
      "first_prefix": {
        "parent": "mature_r2",
        "effect": "first_prefix__minus__mature_r2",
        "decision": "discard",
        "reason": "Macro delta -0.003292829,CI95 [-0.006288188901790596, -0.0005409899799124383],latency ratio 1.005509. Fails positive and near-candidate quality policy; preserve negative evidence.",
        "hypothesis": "Equal-count first_prefix stored-history support may change useful two-round retrieval conditions; no pure Q2 mediation claim."
      },
      "first_suffix": {
        "parent": "mature_r2",
        "effect": "first_suffix__minus__mature_r2",
        "decision": "discard",
        "reason": "Macro delta -0.004871842,CI95 [-0.007814593774694572, -0.0022479055763103472],latency ratio 1.006671. Fails positive and near-candidate quality policy; preserve negative evidence.",
        "hypothesis": "Equal-count first_suffix stored-history support may change useful two-round retrieval conditions; no pure Q2 mediation claim."
      },
      "first_even": {
        "parent": "mature_r2",
        "effect": "first_even__minus__mature_r2",
        "decision": "discard",
        "reason": "Macro delta -0.004675091,CI95 [-0.007934792701201314, -0.0017019098487541366],latency ratio 1.013763. Fails positive and near-candidate quality policy; preserve negative evidence.",
        "hypothesis": "Equal-count first_even stored-history support may change useful two-round retrieval conditions; no pure Q2 mediation claim."
      },
      "first_odd": {
        "parent": "mature_r2",
        "effect": "first_odd__minus__mature_r2",
        "decision": "keep",
        "reason": "Macro delta +0.001237919,CI95 [-0.0013729652000838489, 0.003909334895993671],latency ratio 1.011287. Research candidate under frozen point/cost policy only; no confirmed parent improvement, click regression, no incumbent promotion.",
        "hypothesis": "Equal-count first_odd stored-history support may change useful two-round retrieval conditions; no pure Q2 mediation claim."
      },
      "second_prefix": {
        "parent": "mature_r2",
        "effect": "second_prefix__minus__mature_r2",
        "decision": "discard",
        "reason": "Macro delta -0.012856043,CI95 [-0.017204751230466074, -0.00938755160395674],latency ratio 1.006201. Fails positive and near-candidate quality policy; preserve negative evidence.",
        "hypothesis": "Equal-count second_prefix stored-history support may change useful two-round retrieval conditions; no pure Q2 mediation claim."
      },
      "second_suffix": {
        "parent": "mature_r2",
        "effect": "second_suffix__minus__mature_r2",
        "decision": "discard",
        "reason": "Macro delta -0.002208599,CI95 [-0.005286403531037203, 0.0005428426213684278],latency ratio 1.007043. Fails positive and near-candidate quality policy; preserve negative evidence.",
        "hypothesis": "Equal-count second_suffix stored-history support may change useful two-round retrieval conditions; no pure Q2 mediation claim."
      },
      "second_even": {
        "parent": "mature_r2",
        "effect": "second_even__minus__mature_r2",
        "decision": "discard",
        "reason": "Macro delta -0.002107763,CI95 [-0.005397456379620315, 0.0012215223930179182],latency ratio 1.012147. Fails positive and near-candidate quality policy; preserve negative evidence.",
        "hypothesis": "Equal-count second_even stored-history support may change useful two-round retrieval conditions; no pure Q2 mediation claim."
      },
      "second_odd": {
        "parent": "mature_r2",
        "effect": "second_odd__minus__mature_r2",
        "decision": "discard",
        "reason": "Macro delta -0.003773443,CI95 [-0.0063889316344838635, -0.0012820873871544369],latency ratio 1.013405. Fails positive and near-candidate quality policy; preserve negative evidence.",
        "hypothesis": "Equal-count second_odd stored-history support may change useful two-round retrieval conditions; no pure Q2 mediation claim."
      }
    },
    "new_supported_lessons": 0,
    "old_incumbent_changed": false
  },
  "latest_endpoint_read_outcome": {
    "summary": "even_last保留研究候选：相对mature_r2 macro+0.001435345但CI跨零，点击改善/购买BCE退步，延迟+0.607%；odd_last和odd_no_last低于原循环，discard。偶数取样的末条保留效应为正，但奇偶末条效应交互CI跨零；无确认宏观提升或第二轮纯检索机制结论。",
    "formal_training_runs": 3,
    "reused_reference_training_runs": 4,
    "report": "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/RESULTS.md",
    "paired": "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/paired-analysis-recovery/result.json",
    "latency": "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/latency-recovery/completion.json",
    "candidates": {
      "even_last": {
        "parent": "mature_r2",
        "effect": "even_last__minus__mature_r2",
        "decision": "keep",
        "reason": "Macro delta +0.001435345,CI95 [-0.001703624913421238, 0.004117847779228911];latency ratio1.006071. Research candidate only, macroCIcrosszero andbuyBCEregression;noincumbentpromotion.",
        "hypothesis": "Separate first-work stored-endpoint retention from interleaving phase at fixed support count, two work rounds and no prediction feedback."
      },
      "odd_last": {
        "parent": "mature_r2",
        "effect": "odd_last__minus__mature_r2",
        "decision": "discard",
        "reason": "Macro delta -0.004784711,CI95 [-0.007743902623144495, -0.00213613676149459];latency ratio1.005589. Below matched original recurrence; preserve negative outcome.",
        "hypothesis": "Separate first-work stored-endpoint retention from interleaving phase at fixed support count, two work rounds and no prediction feedback."
      },
      "odd_no_last": {
        "parent": "mature_r2",
        "effect": "odd_no_last__minus__mature_r2",
        "decision": "discard",
        "reason": "Macro delta -0.007211877,CI95 [-0.010450704717793483, -0.004335497843915446];latency ratio1.008237. Below matched original recurrence; preserve negative outcome.",
        "hypothesis": "Separate first-work stored-endpoint retention from interleaving phase at fixed support count, two work rounds and no prediction feedback."
      }
    },
    "new_supported_lessons": 0,
    "old_incumbent_changed": false
  },
  "latest_round_accumulation_outcome": {
    "summary": "首轮累积路径对点击有局部收益，且依历史长度和槽而异；分轮与携带的训练交互为正，但分轮＋携带整体AUC显著低于原两轮，不能晋升。保留机制候选与Cross8对照候选，不增加seed。",
    "formal_training_runs": 4,
    "reused_reference_training_runs": 2,
    "report": "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/RESULTS.md",
    "candidates": {
      "shared_carry": {
        "decision": "discard",
        "reason": "相对原两轮macro AUC下降0.006701且区间在零以下；未提供整体优化收益。",
        "hypothesis": "检验轮次区分的残差更新与首轮状态携带的交互，固定两轮、优化器和seed。"
      },
      "distinct_carry": {
        "decision": "keep",
        "reason": "仅作为点击/状态路由研究候选：条件携带收益和交互为正，点击改善；整体显著低于原两轮，不晋升。",
        "hypothesis": "检验轮次区分的残差更新与首轮状态携带的交互，固定两轮、优化器和seed。"
      },
      "shared_no_carry": {
        "decision": "keep",
        "reason": "作为对照候选：较同协议Cross8 macro AUC提高0.004332且区间为正，但低于原两轮点估计，不晋升。",
        "hypothesis": "检验轮次区分的残差更新与首轮状态携带的交互，固定两轮、优化器和seed。"
      },
      "distinct_no_carry": {
        "decision": "discard",
        "reason": "相对原两轮macro AUC下降0.010544且区间在零以下，轮次更新脱离携带路径没有收益。",
        "hypothesis": "检验轮次区分的残差更新与首轮状态携带的交互，固定两轮、优化器和seed。"
      }
    },
    "new_supported_lessons": 0,
    "old_incumbent_changed": false
  }
}

```
