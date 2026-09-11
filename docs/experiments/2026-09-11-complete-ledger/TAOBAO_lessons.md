# Taobao lessons.json 快照

此处保留历史判断，不把待执行计划当作已完成实验。

```json
{
  "lessons": [
    {
      "lesson_id": "taobao-transfer-r7-not-winner",
      "status": "provisional",
      "statement": "KuaiRec r7 winner did not beat Cross8 on four-task Taobao; Cross8clip1 is current strongest reference, r7 retains a modest inference speed tradeoff.",
      "supporting_episode_ids": [
        "taobao_cross8-s20262027-20260907T084102636174Z",
        "taobao_cross8_clip1-s20262027-20260907T084102654002Z",
        "taobao_cross11-s20262027-20260907T084102626054Z",
        "taobao_r7_loop2-s20262027-20260907T084102641511Z"
      ],
      "confidence": 0.65,
      "scope": "One existing seed, one epoch, released Taobao protocol",
      "falsification_condition": "Capacity/recurrence changes under the same Taobao budget may change the ranking",
      "scope_note": "Historical round-specific result; current incumbent is in strategy.json, not historical winner wording."
    },
    {
      "lesson_id": "taobao-round2-width-not-whole-loop",
      "status": "provisional",
      "statement": "Width384 single pass wins macroAUC; whole-history recurrence lowers macroAUC at widths256/384/512. Click supplies76.5% of best macro gain; latency rises23.5%.",
      "scope": "Fixed seed and released validation protocol",
      "supporting_episode_ids": [
        "t2_cross8_w256_loop2-s20262027-20260907T092620181179Z",
        "t2_cross8_w384_loop1-s20262027-20260907T092620130333Z",
        "t2_cross8_w384_loop2-s20262027-20260907T092619887297Z",
        "t2_cross8_w512_loop1-s20262027-20260907T092620048171Z",
        "t2_cross8_w512_loop2-s20262027-20260907T092620129939Z",
        "t2_r7_w384_loop2-s20262027-20260907T092620135016Z"
      ],
      "scope_note": "Historical round-specific result; current incumbent is in strategy.json, not historical winner wording."
    },
    {
      "lesson_id": "taobao-round3-local-damped-loop-wins",
      "status": "provisional",
      "statement": "Tail half update increases all4taskAUC at unchanged parameters; macro delta0.001386 CI[0.000520,0.002246], inference+17.2%. Entire stack recurrence loses.",
      "scope": "Fixed seed, selected candidate; released validation protocol",
      "scope_note": "Historical round-specific result; current incumbent is in strategy.json, not historical winner wording."
    },
    {
      "lesson_id": "taobao-round4-tail-sweep-no-gain",
      "status": "provisional",
      "statement": "All6width/strength/depth candidates below t3_tail_half across4AUCs; extra passes and width512 increase latency. Keep previous winner and focus requested optimizer substitution.",
      "scope": "Single seed fixed protocol",
      "artifacts": [
        "workspace/qformer-loop/taobao-transfer/round4/reports/summary.json",
        "workspace/qformer-loop/taobao-transfer/round4/reports/final-audit.json"
      ],
      "scope_note": "Historical round-specific result; current incumbent is in strategy.json, not historical winner wording."
    },
    {
      "lesson_id": "taobao-psgdkron-four-structures",
      "status": "provisional",
      "statement": "PSGDKron improves macro AUC within all four structures. Width384 single pass is highest observed at 0.827466136; extra tail/post recurrence does not improve its point estimate. Within-architecture optimizer gain user-bootstrap intervals are positive for Cross8, width384 single and post-half, but cross zero for tail-half.",
      "scope": "One training seed; fixed LR, clip1 and per-pair architecture; conditional checkpoint bootstrap, selection uncorrected",
      "artifact_paths": [
        "workspace/qformer-loop/taobao-transfer/optimizer-psgd-parallel/COMPARISON.md",
        "workspace/qformer-loop/taobao-transfer/optimizer-psgd-parallel/reports/final-audit.json"
      ],
      "scope_note": "Historical round-specific result; current incumbent is in strategy.json, not historical winner wording."
    },
    {
      "lesson_id": "taobao-q4-two-cycle-benefit",
      "statement": "Matched q4 R2 exceeds R1 by 0.001960375; paired-user CI excludes zero. Keep R2 for this candidate; this does not show recurrence beats the best single-pass architecture.",
      "status": "provisional",
      "confidence": "single_training_seed_fixed_checkpoint_bootstrap",
      "supporting_episode_ids": [
        "q4_r1-s20262027-20260908T150233962301Z",
        "taobao-q4-r1-r2-paired-bootstrap"
      ],
      "counterevidence_episode_ids": [],
      "scope": "Taobao width384 q4 no-feedback PSGDKron every-step fixed budget",
      "falsification_condition": "A matched query-formation retraining control erases the benefit; bootstrap does not establish seed robustness"
    },
    {
      "lesson_id": "taobao-local-capacity-sweep-negative",
      "statement": "Seven tested FFN/entry capacity variants failed to exceed their matching q4/q8 two-cycle references; lower priority of more nearby FFN scalar sweeps.",
      "status": "supported",
      "confidence": "multiple_architecture_outcomes_one_seed",
      "supporting_episode_ids": [
        "q4_postffn2-s20262027-20260908T150326065288Z",
        "q4_postffn8-s20262027-20260908T150329037662Z",
        "q4_ffn2-s20262027-20260908T150239241335Z",
        "q4_ns3-s20262027-20260908T150248111717Z",
        "q8_postffn2-s20262027-20260908T150330839374Z",
        "q8_postffn8-s20262027-20260908T150325024497Z",
        "q8_ffn2-s20262027-20260908T150232883029Z"
      ],
      "counterevidence_episode_ids": [],
      "scope": "Only tested ratios2/8 and entry3 in this basin",
      "falsification_condition": "A controlled capacity reallocation exceeds the matched baseline under unchanged protocol"
    },
    {
      "lesson_id": "taobao-query-information-not-yet-diagnosed",
      "statement": "Removing dynamic second-work retrieval and tested context reread both lose quality; these outcomes do not identify an optimal second query. Measure first-work information contribution before further formula sweeps.",
      "status": "provisional",
      "confidence": "mechanism_unresolved",
      "supporting_episode_ids": [
        "nf_fixed_memory-s20262027-20260908T115619206369Z",
        "ctx_read-s20262027-20260908T133655778759Z",
        "ctx_read_middle-s20262027-20260908T133653654700Z"
      ],
      "counterevidence_episode_ids": [],
      "scope": "Two-cycle no-feedback, second query currently (entry+first_work)/2",
      "falsification_condition": "Same-weight paired interventions show no sample-specific dependence on first_work or entry",
      "followup_note": "Paired diagnostic taobao-query-parts-paired-5514-q4-q8 now supports preserving both sources; optimal query formation remains unresolved."
    },
    {
      "lesson_id": "second-query-preserve-both-sources",
      "statement": "Preserve entry and first-work information; their example-specific contributions matter for fav/buy in both checkpoints; work shuffle additionally harms click. Cart remains inconclusive.",
      "status": "supported",
      "scope": "Fixed stratified subset and checkpoints, CPU float32, user bootstrap conditional on selected subset, not full-population selection variance or training-seed variance",
      "confidence": "conditional_mechanism_evidence_two_checkpoints_not_retraining_gain",
      "supporting_episode_ids": [
        "taobao-query-parts-paired-5514-q4-q8"
      ],
      "counterevidence_episode_ids": [],
      "falsification_condition": "Repetition under full validation or independent subset reverses paired dependence; this lesson does not predict adapter gain"
    },
    {
      "lesson_id": "taobao-second-cycle-memory-concatenation-negative",
      "statement": "All eight tested second-cycle memory concatenations lose against matching q4/q8 references; retain original memory paths for query-correction tests.",
      "status": "supported",
      "confidence": "eight_architecture_outcomes_one_seed",
      "scope": "One training seed; tested concatenation locations only, not evidence all memory-routing designs fail. Latency is repeated validation batch forward only.",
      "supporting_episode_ids": [
        "memory_q4_work_entry-s20262027-20260908T155435293535Z",
        "memory_q4_prediction_history-s20262027-20260908T155442368353Z",
        "memory_q4_prediction_states-s20262027-20260908T155437780821Z",
        "memory_q4_prediction_entry-s20262027-20260908T155428499084Z",
        "memory_q8_work_entry-s20262027-20260908T155431113128Z",
        "memory_q8_prediction_history-s20262027-20260908T155431290001Z",
        "memory_q8_prediction_states-s20262027-20260908T155422308090Z",
        "memory_q8_prediction_entry-s20262027-20260908T155429941953Z"
      ],
      "counterevidence_episode_ids": [],
      "falsification_condition": "A controlled different routing design exceeds its matched reference; these outcomes do not establish query correction will help."
    },
    {
      "lesson_id": "taobao-query-correction-source-depends-on-query-count",
      "status": "provisional",
      "statement": "Among completed candidates, work has the highest q4 point estimate while entry has the highest q8 point estimate. Their respective improvements over same-size mean controls have positive conditional user-bootstrap intervals; this does not establish a transferable source ranking or a gain over the original recurrence.",
      "scope": "All eight source candidates completed, including the audited q4 difference retry. One training seed, fixed checkpoint intervals, selection uncorrected. Whole-input corrections affect retrieval, interaction and residual together.",
      "confidence": "source_controls_in_two_query_sizes_single_training_seed",
      "supporting_episode_ids": [
        "query_q4_work-s20262027-20260908T171234834290Z",
        "query_q4_mean-s20262027-20260908T171242116320Z",
        "query_q8_entry-s20262027-20260908T171234804182Z",
        "query_q8_mean-s20262027-20260908T171226307157Z",
        "taobao-query-correction-v2-query_q4_mean-query_q4_work-paired-bootstrap",
        "taobao-query-correction-v2-query_q8_mean-query_q8_entry-paired-bootstrap"
      ],
      "counterevidence_episode_ids": [
        "query_q8_work-s20262027-20260908T171237243608Z",
        "query_q8_difference-s20262027-20260908T171234078895Z",
        "query_q4_difference-s20262027-20260908T184409455021Z"
      ],
      "missing_evidence_episode_ids": [],
      "falsification_condition": "A controlled query-count/source comparison reverses the observed source ranking; current evidence does not identify a transferable optimal source.",
      "resolved_execution_failure": {
        "successful_retry": "query_q4_difference-s20262027-20260908T184409455021Z",
        "original_failure_preserved": true
      }
    },
    {
      "lesson_id": "taobao-query-q4-work-near-reference-without-demonstrated-gain",
      "status": "provisional",
      "statement": "Retain q4 work correction as a near-reference optimization candidate: macro AUC delta -0.000006788835 versus ctx_q4, conditional CI [-0.000812322970, 0.000722271955]. All eight completed added-capacity variants do not improve their matched recurrence point estimates; extra capacity alone does not guarantee a gain.",
      "scope": "One training seed; fixed checkpoints and released validation protocol; selection uncorrected; no training-seed variance or equivalence claim. A cross-zero interval is not equivalence. q4_work raises click but lowers buy in point estimates. Repeated-batch forward latency is about 14.19 versus 14.32 ms; no reliable speedup claim.",
      "confidence": "one_matched_training_pair_with_conditional_user_bootstrap_and_seven_quality_outcomes",
      "supporting_episode_ids": [
        "query_q4_work-s20262027-20260908T171234834290Z",
        "taobao-query-correction-v2-ctx_q4-query_q4_work-paired-bootstrap"
      ],
      "counterevidence_episode_ids": [
        "query_q4_entry-s20262027-20260908T171244424318Z",
        "query_q4_mean-s20262027-20260908T171242116320Z",
        "query_q8_difference-s20262027-20260908T171234078895Z",
        "query_q8_entry-s20262027-20260908T171234804182Z",
        "query_q8_work-s20262027-20260908T171237243608Z",
        "query_q8_mean-s20262027-20260908T171226307157Z",
        "query_q4_difference-s20262027-20260908T184409455021Z"
      ],
      "falsification_condition": "A later audited matched result establishes a material improvement or regression; current evidence neither promotes the incumbent nor identifies the retrieval mechanism."
    },
    {
      "lesson_id": "correction-route-dependence-is-checkpoint-specific",
      "status": "provisional",
      "statement": "Fixed-checkpoint q4 retrieval correction helps fav/buy while residual correction hurts them; q8 retrieval does not repeat the same benefit. Subsequent matched route-restricted retraining did not improve all-route or original recurrence. Preserve the diagnostic observation, lower the associated architecture priority.",
      "confidence": "two_selected_fixed_checkpoints_conditional_subset_only",
      "scope": "Fixed 5514-row stratified subset, 4647 selected users, CPU FP32, IPW BCE; paired bootstrap conditional on checkpoint and subset, not sampling-design uncertainty or training-seed variance. Multiple interventions and candidate selection uncorrected. No test evaluation.",
      "supporting_episode_ids": [
        "taobao-correction-three-routes-q4work-q8entry-5514"
      ],
      "counterevidence_episode_ids": [],
      "falsification_condition": "The fixed-checkpoint intervention fails to reproduce under identical checkpoint/subset/protocol; do not interpret it as a prediction of retraining gain.",
      "retraining_followup_episode": "taobao-query-routes-final-interpretation",
      "proposed_retraining_gain_supported": false
    },
    {
      "lesson_id": "taobao-difference-only-additive-query-correction-negative",
      "status": "supported",
      "statement": "Difference-conditioned rank16 whole-input additive correction is below the corresponding mean-conditioned control in q4 and q8; both fixed-checkpoint macroAUC intervals are negative. Lower this specific source/offset design's priority.",
      "supporting_episode_ids": [
        "query_q4_difference-s20262027-20260908T184409455021Z",
        "query_q8_difference-s20262027-20260908T171234078895Z",
        "taobao-query-difference-retry-query_q4_mean-query_q4_difference-paired-bootstrap",
        "taobao-query-correction-v2-query_q8_mean-query_q8_difference-paired-bootstrap"
      ],
      "confidence": "two_matched_architecture_outcomes_one_training_seed",
      "scope": "Taobao two-cycle no-feedback, width384/rank16, fixed PSGDKron budget; no claim that differences are useless as features or multiplicative fusion directions",
      "counterevidence_episode_ids": [],
      "falsification_condition": "A matched difference-only additive correction exceeds its mean-conditioned control under the same budget"
    },
    {
      "lesson_id": "taobao-query-route-restriction-no-retraining-gain",
      "status": "supported",
      "statement": "All eight tested route restrictions are below matching all-route and original-recurrence point estimates; seven all-route intervals are negative, q4 retrieval crosses zero. No new candidate meets the predeclared quality/compute queue rule.",
      "supporting_episode_ids": [
        "route_q4_retrieval-s20262027-20260908T190038665636Z",
        "route_q4_interaction-s20262027-20260908T190043679512Z",
        "route_q4_residual-s20262027-20260908T190045767031Z",
        "route_q4_attention-s20262027-20260908T190044491955Z",
        "route_q8_retrieval-s20262027-20260908T190022268468Z",
        "route_q8_interaction-s20262027-20260908T190028413046Z",
        "route_q8_residual-s20262027-20260908T194602060181Z",
        "route_q8_attention-s20262027-20260908T190022893149Z",
        "taobao-query-routes-final-interpretation"
      ],
      "confidence": "eight_architecture_outcomes_single_training_seed_selection_uncorrected",
      "scope": "q4/work and q8/entry rank16 corrections: retrieval, interaction, residual, attention-only. Source and query count differ between conditions; no general no-recurrence conclusion",
      "counterevidence_episode_ids": [
        "taobao-correction-three-routes-q4work-q8entry-5514"
      ],
      "falsification_condition": "A controlled route-restricted retraining demonstrates useful matched quality/compute benefit"
    },
    {
      "lesson_id": "taobao-query-mixing-bounded-gate-outcomes",
      "status": "provisional",
      "statement": "Eight bounded rank16 mixing outcomes: 0 pass the predeclared near-reference quality/compute queue rule; 0 positive-quality candidates exceed its compute limit. This queue policy is not equivalence.",
      "supporting_episode_ids": [
        "taobao-query-mixing-final-interpretation"
      ],
      "confidence": "single_seed_selection_uncorrected",
      "scope": "q4/q8 and channel/query separately; only this rank16/bound0.25 budget, no general conclusion about recurrence",
      "falsification_condition": "Later matched training and compute evidence changes the observed quality/compute conclusions"
    },
    {
      "lesson_id": "taobao-query-mixing-conditioning-versus-static-control",
      "status": "provisional",
      "statement": "Of four matched joint-minus-constant pairs, 0 also have positive lower bounds against the original recurrence. All four pairs and intervals are retained; fixed-checkpoint gate diagnostics are separate evidence.",
      "supporting_episode_ids": [
        "taobao-query-mixing-final-interpretation"
      ],
      "confidence": "fixed_checkpoint_intervals_selection_uncorrected",
      "scope": "Same trainable parameter count within granularity, different conditioning/gradient paths and FLOPs; no isolated retrieval attribution",
      "falsification_condition": "Matched training reverses the conditional comparisons; shuffle sensitivity alone cannot establish training benefit"
    },
    {
      "lesson_id": "taobao-q4-channel-gate-correspondence-not-demonstrated-useful",
      "status": "provisional",
      "statement": "On the prospectively fixed q4 channel joint checkpoint, zeroing and shuffling the learned gate both reduce fav IPW BCE with negative conditional intervals; click/cart/buy intervals cross zero. Constant shuffle is exact, and constant zero/click shows only a tiny positive conditional effect. Nonzero gate variance does not establish useful sample conditioning; preserve full-training comparisons separately.",
      "supporting_episode_ids": [
        "taobao-query-mixing-q4-channel-gate-diagnostics-5514"
      ],
      "confidence": "fixed_checkpoint_and_subset_conditional_selection_uncorrected",
      "scope": "Two fixed q4 channel checkpoints;5514 stratified validation rows,4647 users,CPU FP32;not isolated retrieval or a q8/query-granularity result",
      "falsification_condition": "A separately validated matched checkpoint/subset intervention changes these task-specific findings; training gain must be established by training comparisons"
    },
    {
      "lesson_id": "taobao-q4-buy-second-attention-redistribution",
      "status": "provisional",
      "statement": "On original ctx_q4.buy, fixed-query direction replacement accounts for the harmful combined first-query replacement at the predeclared epsilon1e-6 scale: direction,both,direction-minus-norm family40 intervals exceed epsilon; norm and interaction intervals lie within +/-epsilon. Norm is small, not exactly zero. This decomposes the same checkpoint/subset effect; it is not independent training evidence or an only-buy rule.",
      "supporting_episode_ids": [
        "taobao-second-read-attention-originals-5514",
        "taobao-second-query-direction-norm-5514"
      ],
      "confidence": "same_checkpoint_subsample_mechanism_decomposition_no_independent_training_gain",
      "scope": "Original q4/q8 checkpoints,5514 fixed stratified rows,4647 users,CPU FP32; model quality/seed variance not established",
      "counterevidence": [
        "Other seven checkpoint/task results are inconclusive, including every q8 task",
        "Earlier learned query corrections mostly failed to beat the original training AUC; necessity of current retrieval does not imply a new correction will help",
        "New direction/norm decomposition has no between-task contrast and other seven checkpoint/task joint gates do not pass. The same checkpoints/subset are reused, not independent training replication."
      ],
      "falsification_condition": "A separately validated matching probability-only intervention changes the supported task direction; new architecture gains require the fixed training and full-validation comparison",
      "latest_evidence": "workspace/qformer-loop/taobao-transfer/recurrence-query-direction-norm/reports/interpretation.json"
    },
    {
      "lesson_id": "taobao-recurrence-benefit-depends-on-task-and-comparison",
      "statement": "In each fixed q4 trained-depth and q8 frozen-depth comparison, cart/buy AUC effects exceed click at exploratory 95% paired task-contrast intervals. Yet the q8 checkpoint benefits from its whole second cycle on all four AUC/BCE heads, while q4 click BCE worsens. Task/metric-specific benefit is a useful research clue; only-buy or task-label-only rules are unsupported.",
      "scope": "Taobao original q4/q8 saved validation predictions, 2871767 rows,268201 users; different depth interventions; uncorrected 95% intervals with no training-seed uncertainty",
      "status": "provisional",
      "confidence": "conditional_evidence_not_new_model_gain",
      "supporting_episode_ids": [
        "taobao-recurrence-task-effects-saved-validation"
      ],
      "counterevidence_episode_ids": [
        "taobao-second-read-attention-originals-5514"
      ],
      "counterevidence": "Attention-only replacement has a narrower supported q4.buy direction; q8 whole-cycle gain on all heads rules out generalizing that result to only-buy recurrence utility.",
      "falsification_condition": "A matched task-effect analysis changes these fixed-scope contrasts; any task-readout gain must beat the shared-mixture and original readout controls under the existing training and compute policy.",
      "use_count": 0,
      "successful_use_count": 0,
      "last_used_at": "2026-09-09T06:06:00.273415+00:00",
      "supersedes": null
    },
    {
      "lesson_id": "taobao-readout-retrieval-early-final-readout",
      "status": "supported",
      "statement": "Four tested shared/task early-final readout variants fall below corresponding originals; q8 task beats shared but not the original.",
      "scope": "Only tested q4/q8 variants, fixed single training seed and 2193 updates; not all readouts/retrieval architectures",
      "supporting_episode_ids": [
        "role_q4_read_shared-s20262027-20260909T063952359077Z",
        "role_q4_read_task-s20262027-20260909T063952341877Z",
        "role_q8_read_shared-s20262027-20260909T063952341972Z",
        "role_q8_read_task-s20262027-20260909T063952422184Z"
      ],
      "confidence": "repeated_valid_negative_outcomes_with_exploratory_fixed_checkpoint_intervals",
      "counterevidence": "Original task-dependent cycle benefits remain; q8 task-minus-shared is positive, without exceeding original.",
      "falsification_condition": "A materially different matched design exceeds the original under the predefined quality and compute policy; optimizer-only sensitivity is insufficient."
    },
    {
      "lesson_id": "taobao-readout-retrieval-second-query-untie",
      "status": "supported",
      "statement": "Four tested second-cycle query norm/projection unties fall below corresponding originals under the fixed native PSGDKron training protocol.",
      "scope": "Only tested q4/q8 variants, fixed single training seed and 2193 updates; not all readouts/retrieval architectures",
      "supporting_episode_ids": [
        "role_q4_query_norm-s20262027-20260909T063952376231Z",
        "role_q4_query_projection-s20262027-20260909T063952342021Z",
        "role_q8_query_norm-s20262027-20260909T063952342021Z",
        "role_q8_query_projection-s20262027-20260909T063952355394Z"
      ],
      "confidence": "repeated_valid_negative_outcomes_with_exploratory_fixed_checkpoint_intervals",
      "counterevidence": "Original task-dependent cycle benefits remain; q8 task-minus-shared is positive, without exceeding original.",
      "falsification_condition": "A materially different matched design exceeds the original under the predefined quality and compute policy; optimizer-only sensitivity is insufficient."
    },
    {
      "lesson_id": "taobao-initial-function-equality-does-not-match-optimizer-layout",
      "status": "provisional",
      "statement": "In the fixed synthetic CPU eager replay, adding parameter layouts changes existing updates through clip,common bucket Q initialization and old-member probes. All3 control restores old updates/state in both layouts, while new parameters update. Initial function equality alone does not establish matched optimizer dynamics.",
      "scope": "Local HeavyBall3.2.1;formula gradients andone step;no actual Taobao AUC magnitude or compiled result",
      "supporting_episode_ids": [
        "taobao-optimizer-layout-synthetic-replay"
      ],
      "confidence": "synthetic_only",
      "counterevidence": "Actual native compiled coupling and real-gradient magnitudes remain unmeasured;all eight trained losses retain their original decisions.",
      "falsification_condition": "Native matched replay fails the same controls or effect;model gain still requires fixed protocol full-validation training evidence."
    },
    {
      "lesson_id": "taobao-tested-directional-query-feedback-no-new-candidate",
      "status": "supported",
      "statement": "All eight tested q4/q8 bounded tangent query-extrapolation/key-feedback models fall below their own original macro AUC by more than0.0001 and increase measured forward latency; none passes the fixed quality/near queue policy.",
      "scope": "Only these eight independently trained structural variants, single fixed seed20262027,2193updates,PSGDKron; not a general rejection of directional retrieval or recurrence.",
      "supporting_episode_ids": [
        "dir_q4_extrapolate-s20262027-20260909T093929334811Z",
        "dir_q4_uniform_key-s20262027-20260909T093929427714Z",
        "dir_q4_attended_key-s20262027-20260909T093929333387Z",
        "dir_q4_centered_key-s20262027-20260909T093929350350Z",
        "dir_q8_extrapolate-s20262027-20260909T093929348738Z",
        "dir_q8_uniform_key-s20262027-20260909T093929349342Z",
        "dir_q8_attended_key-s20262027-20260909T093929344937Z",
        "dir_q8_centered_key-s20262027-20260909T093929355117Z"
      ],
      "confidence": "repeated_valid_negative_training_outcomes_under_one_fixed_protocol",
      "counterevidence": "Original task-specific second-query dependence remains; some candidate-vs-parent macro intervals may cross zero. Candidate-vs-candidate improvements cannot establish a gain over the original.",
      "falsification_condition": "A distinct matched structural design passes the predeclared parent/best-single quality and compute gates; nearby scalar scans and same-checkpoint dependency alone do not establish such a gain.",
      "use_count": 0,
      "successful_use_count": 0,
      "last_used_at": "2026-09-09T10:40:18.868443+00:00",
      "supersedes": null
    },
    {
      "lesson_id": "taobao-tested-signed-cross-slot-coverage-no-candidate",
      "status": "supported",
      "statement": "All eight tested q4/q8 strength1 signed mean/leave-own-slot-out first-read coverage biases reduce macro AUC versus their original recurrence, with all eight exploratory user-paired95%macro intervals below zero and8.13–9.46%higher measured forward latency. None passes the fixed quality/near queue policy; deprioritize nearby strength scans without new evidence.",
      "scope": "Only these eight separately trained variants under seed20262027,2193updates,PSGDKron every-step,clip1,two cycles/no feedback; not a general rejection of coverage, second-read attention or recurrence. No seed variance or selection/multiplicity correction.",
      "supporting_episode_ids": [
        "cov_q4_mean_positive-s20262027-20260909T111449014321Z",
        "cov_q4_mean_negative-s20262027-20260909T111449096617Z",
        "cov_q4_loo_positive-s20262027-20260909T111449024286Z",
        "cov_q4_loo_negative-s20262027-20260909T111449019377Z",
        "cov_q8_mean_positive-s20262027-20260909T111449020979Z",
        "cov_q8_mean_negative-s20262027-20260909T111449014936Z",
        "cov_q8_loo_positive-s20262027-20260909T111449019377Z",
        "cov_q8_loo_negative-s20262027-20260909T111449097724Z"
      ],
      "confidence": "repeated_valid_negative_structural_outcomes_under_one_fixed_training_protocol",
      "counterevidence": "q8 LOO beats mean for both signs inside this losing family; q4 mean-positive improves click versus its small parent while worsening the other three tasks. Original q4/q8 second-cycle dependence remains. None establishes a new macro/cost candidate or an only-task rule.",
      "falsification_condition": "A distinct controlled coverage or memory-role structure exceeds its original under the same predeclared quality/cost gates. A changed attention statistic, candidate-to-candidate advantage or posthoc task target alone is insufficient.",
      "artifact_paths": [
        "workspace/qformer-loop/taobao-transfer/recurrence-coverage-retrieval/RESULTS.md",
        "workspace/qformer-loop/taobao-transfer/recurrence-coverage-retrieval/reports/paired-result-independent-review.json",
        "workspace/qformer-loop/taobao-transfer/recurrence-coverage-retrieval/reports/training-latency-independent-review.json"
      ],
      "use_count": 0,
      "successful_use_count": 0,
      "last_used_at": "2026-09-09T12:23:45.032427+00:00",
      "supersedes": null
    },
    {
      "lesson_id": "taobao-tested-second-read-field-roles-no-candidate",
      "status": "supported",
      "statement": "All eight tested q4/q8 second-read item/full,full/action,item/action,action/item K/V field-role variants have negative macro-AUC point deltas below -0.0001 versus their original recurrence and 13.73–29.14% higher measured forward latency. None passes the fixed quality/near queue policy. Six exploratory parent macro intervals are wholly negative; q4 value_action and item_action cross zero. Deprioritize nearby raw-field role scans without new evidence.",
      "scope": "These eight separately trained models only: seed20262027,2193updates,PSGDKron every-step,clip1,two cycles/no prediction feedback,zero new parameters,original first full read and Q2. Not a rejection of recurrence, all K/V specialization, or all query conditioning; no training-seed variance or selection/multiplicity correction.",
      "supporting_episode_ids": [
        "field_q4_key_item-s20262027-20260909T130018194006Z",
        "field_q4_value_action-s20262027-20260909T130018189098Z",
        "field_q4_item_action-s20262027-20260909T130018174644Z",
        "field_q4_action_item-s20262027-20260909T130018191153Z",
        "field_q8_key_item-s20262027-20260909T130018196759Z",
        "field_q8_value_action-s20262027-20260909T130018202302Z",
        "field_q8_item_action-s20262027-20260909T130018207493Z",
        "field_q8_action_item-s20262027-20260909T130018194011Z"
      ],
      "confidence": "multiple_valid_fixed_protocol_structural_outcomes_with_independently_reproduced_saved_user_pair_intervals",
      "counterevidence": "The item/action combination has positive nonadditivity I in both parents (q4 +0.003245402, q8 +0.005071222, both exploratory CIs positive), but neither combined model beats its original. q4 value_action improves click AUC while worsening buy AUC; buy BCE improves, so this is an AUC ranking tradeoff, not proof of information loss. Original recurrence and second-query dependence evidence remains.",
      "falsification_condition": "A distinct controlled field-role or query-conditioning design improves its original under the same predeclared macro/cost gates. A positive I, candidate-to-candidate advantage, individual-task improvement, or zero-crossing interval alone is insufficient.",
      "artifact_paths": [
        "workspace/qformer-loop/taobao-transfer/recurrence-field-role/RESULTS.md",
        "workspace/qformer-loop/taobao-transfer/recurrence-field-role/reports/paired-result-independent-review.json",
        "workspace/qformer-loop/taobao-transfer/recurrence-field-role/reports/combination-result-independent-review.json",
        "workspace/qformer-loop/taobao-transfer/recurrence-field-role/reports/training-latency-independent-review.json",
        "workspace/qformer-loop/taobao-transfer/recurrence-field-role/reports/report-interpretation-independent-review.json"
      ],
      "use_count": 0,
      "successful_use_count": 0,
      "last_used_at": "2026-09-09T14:42:50.690451+00:00",
      "supersedes": null
    },
    {
      "lesson_id": "taobao-tested-second-query-work-relay-no-candidate",
      "status": "supported",
      "statement": "All eight tested q4/q8 all/own/uniform/other work-relay variants have negative macro-AUC point deltas below -0.0001 versus their original recurrence; all eight exploratory parent macro intervals are wholly negative. Forward latency rises 7.93–9.40%. None passes the predeclared quality/near queue policy. Deprioritize nearby shared-post work-relay scans without distinct evidence.",
      "scope": "These eight separately trained models only: seed20262027,2193updates,PSGDKron every-step,clip1,two cycles/no prediction feedback,zero new parameters. Only second w_q sees first-work contents relayed through original post weights; native w_qi/residual/H/K/V stay. Not a rejection of recurrence, all inter-slot communication or task-conditioned retrieval; intervals omit training-seed variance and multiplicity/selection correction.",
      "supporting_episode_ids": [
        "relay_q4_all-s20262027-20260909T155756344521Z",
        "relay_q4_own-s20262027-20260909T155756356296Z",
        "relay_q4_uniform-s20262027-20260909T155756355524Z",
        "relay_q4_other-s20262027-20260909T155756344080Z",
        "relay_q8_all-s20262027-20260909T155756390913Z",
        "relay_q8_own-s20262027-20260909T155756400980Z",
        "relay_q8_uniform-s20262027-20260909T155756355347Z",
        "relay_q8_other-s20262027-20260909T155756342306Z"
      ],
      "confidence": "multiple_valid_fixed_protocol_structural_outcomes_with_independently_reproduced_saved_user_pair_intervals",
      "counterevidence": "q4 all minus own improves fav AUC but worsens click AUC, with opposite-signed exploratory task intervals and a negative direct click-minus-fav effect interval; both models still lose to the original. q4 all is worse than uniform on macro; the other five internal macro intervals cross zero. These trained comparisons show task tradeoffs, not task-exclusive recurrence information or causal information quantity. Original recurrence and second-query dependence evidence remains.",
      "falsification_condition": "A distinct controlled query-conditioning design improves its original under the same predeclared macro/cost gates. Internal relative gain, one task improvement, or a zero-crossing interval alone is insufficient.",
      "artifact_paths": [
        "workspace/qformer-loop/taobao-transfer/recurrence-work-relay/RESULTS.md",
        "workspace/qformer-loop/taobao-transfer/recurrence-work-relay/reports/paired-result-independent-review.json",
        "workspace/qformer-loop/taobao-transfer/recurrence-work-relay/reports/training-latency-independent-review.json",
        "workspace/qformer-loop/taobao-transfer/recurrence-work-relay/reports/report-interpretation-independent-review.json"
      ],
      "use_count": 0,
      "successful_use_count": 0,
      "last_used_at": "2026-09-09T17:11:55.270069+00:00",
      "supersedes": null
    },
    {
      "lesson_id": "taobao-tested-four-branch-second-query-bias-no-candidate",
      "status": "supported",
      "statement": "All four tested q4/q8 shared/task projected-query bias variants lose macro AUC versus their own original recurrence; all four exploratory parent macro intervals are negative. Forward latency rises72.21–75.17%, and none meets the prespecified quality/near candidate policy. Deprioritize nearby four-branch query-bias scans without distinct evidence.",
      "scope": "These four independently trained fixedseed20262027/2193update/PSGDKron-every-step/clip1/two-cycle/no-feedback outcomes under released offline action inputs only. Each appends1536 bias parameters and performs four full later branches. Original constructor prefix/RNG and static mappings were checked, not optimizer trajectory equality. No training-seed or multiplicity/selection correction. A availability audit remains a limitation on request-time information interpretation.",
      "supporting_episode_ids": [
        "taskq_q4_shared-s20262027-20260909T173345728371Z",
        "taskq_q4_task-s20262027-20260909T173345700005Z",
        "taskq_q8_shared-s20262027-20260909T173345699338Z",
        "taskq_q8_task-s20262027-20260909T173345704656Z"
      ],
      "confidence": "multiple_valid_fixed_protocol_structural_outcomes_with_saved_user_paired_intervals",
      "counterevidence": "q4 shared improves click AUC while worsening buy AUC, with positive direct click-minus-buy effect interval. q8 task-minus-shared macro is positive but both remain below originalq8; q4 analogous internal contrast is negative. These show task-dependent tradeoffs, not a successful task-specific recurrence structure or task-exclusive information. Prior originalR2-R1 evidence remains scoped to released inputs.",
      "falsification_condition": "A distinct preregistered query design retains gain or meets the same relevant macro/cost policy against its matched original. Internal gain or a favorable selected task alone is insufficient; changed input maturity requires new matched controls.",
      "artifact_paths": [
        "workspace/qformer-loop/taobao-transfer/recurrence-task-query/RESULTS.md",
        "workspace/qformer-loop/taobao-transfer/recurrence-task-query/reports/paired-result-independent-review.json",
        "workspace/qformer-loop/taobao-transfer/recurrence-task-query/reports/training-latency-independent-review.json",
        "workspace/qformer-loop/taobao-transfer/recurrence-task-query/reports/report-interpretation-independent-review.json",
        "workspace/qformer-loop/taobao-transfer/recurrence-task-query/root-scope-and-next.md"
      ],
      "use_count": 0,
      "successful_use_count": 0,
      "last_used_at": "2026-09-09T18:47:41.585103+00:00",
      "supersedes": null
    }
  ],
  "last_consolidated_episode": "taskq_q8_task-s20262027-20260909T173345704656Z"
}

```
