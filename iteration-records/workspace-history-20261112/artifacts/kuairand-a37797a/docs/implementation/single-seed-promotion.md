# Single-seed Amazon promotion

New studies use `development_seeds: [42]`, `confirmation_seeds: []`, and
`selection_policy: baseline-single-seed-v1`. Autoresearch and online tool tasks
inherit the policy from the study unless explicitly configured otherwise.
Previously frozen studies, plans and results keep their old policy and seeds.

After source review, admission, training and independent rescoring, each new
candidate/control is compared with the incumbent on the same seed across all
categories. A strictly positive mean HR@10 gain qualifies. The highest development
mean wins, with ties resolved in favor of the candidate then frozen control order.
No confirmation jobs are run, and `confirmation_executed` remains false. The
controller records `promotion.json`, then updates its incumbent. An absent positive
gain retains the incumbent. The analyst cannot veto metric selection with a
mechanism interpretation, and model promotion is not mechanism confirmation.

For compatibility the development readout still exposes the existing
`eligible_for_confirmation` selection-gate field; with this explicitly named
single-seed policy it denotes eligibility for direct development selection,
not authorization to launch or claim an independent confirmation experiment.

Final evaluation remains sealed outside the optimization study. For explicitly
single-seed studies it uses the validation-selected seed-42 checkpoints, without
retraining or selecting by test results. The evaluation candidate-sampling seed
is a separate frozen protocol setting; it is not an additional training run.

A single-seed policy rejects multiple development seeds or any confirmation seeds
before a plan may run. Do not edit a running/frozen study to switch policies;
create a new study under these defaults instead.
