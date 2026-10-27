# NextItNet evaluator version 2

Version 2 changes model validity checks, not the dataset, training seed, BPR loss,
Adam, batching, evaluation rows, ranking metrics, checkpoint selection or budgets.

The version 1 checker incorrectly required `forward()` to equal the encoder's
dot product with the internal shared `item_emb` table. This rejected legitimate
separate output embeddings, target normalization and learned target projections.
The checker now treats shared-table equivalence and last-position readout as
optional architecture diagnostics. A false diagnostic does not reject a model.

Required checks remain: causal and batch-independent encoding; finite, correctly
shaped public scores; candidate order/set independence. Training scores must now
explicitly be finite and correctly shaped, independent of batch composition, and
unchanged when the same target moves between positive/negative argument roles.
Changing future history or an unrelated target must not affect earlier training
scores; this retains causal training without enforcing a particular scoring formula.
No internal embedding attribute is required. These checks are not a proof of
complete correctness or training/inference equivalence for every possible input.

Staged evaluator subprocess failures now include the underlying worker log tail
in stderr. The controller retains that diagnostic in the experiment record and
next agent packet. Agents can identify the actual failing constraint instead of
seeing only a generic subprocess failure. Check version 2 is recorded in worker
manifests, correctness reports, checkpoint identities and staged metric payloads.
The frozen-trainer test path also checks the loaded checkpoint before prediction.

Before editing, the complete earlier Python source and protocol documents were
archived with SHA-256 hashes in `artifacts/evaluator-fix-v2/legacy-source` and
`legacy-manifest.json`. Historical experiment registrations, sources, checkpoints,
scores and summaries are unchanged. Their version 1 results must retain the
version 1 label. Registration/hash guards deliberately prevent resuming old runs
with the changed evaluator: use a new run prefix and new registration for version 2.
For a matched comparison, all methods must use the same evaluator version.

`artifacts/evaluator-fix-v2/recheck.py` rechecks saved sources/checkpoints on CPU,
including the rejected C candidates, and compares original-baseline predictions
before/after the corrected checks. It adds no training or hidden-test calls and
writes its separate audit to `regression.json`. Passing an initialization-only
check does not imply successful training; rechecking cannot recover the different
search decisions that better error feedback might have produced.
