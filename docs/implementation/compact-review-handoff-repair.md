# Compact review handoff repair after Amazon v7

The v7 campaign completed the experiment flow in rounds 5 and 10. Rounds 1,
2, 3, 6, 8 and 9 ended with reviewer output-contract failures: the reviewer
approved the implementation but put positive source-review observations in
`findings`. The host interpreted every entry as blocking, including after a
format retry. These were harness/interface failures, not failed model training.

Round 7 was unresolved because CPU checks were unavailable at source review,
which precedes `admit_plan`. Round 4 mixed a mistaken inference from unchanged
helper defaults with a real contradiction in its frozen GELU test. The latter
must remain a blocker; this repair does not approve historical experiments.

The per-round source-record index is in
[the failure audit](../experiments/amazon-v7-harness-failure-audit.json).

## Changes

- Reviewer output has separate `checks`, `blocking_findings`, and `limitations`.
  An approval can include positive checks but cannot include any blockers.
  Source identity remains mandatory. Runtime, prefix replay and cycle finalization
  share the same validator. Substantive rejections still stop before training;
  they are not retried until approval.
- Review receives the actual host protocol and compact baseline receipt metadata,
  plus an explicit pre-admission stage. Pending host CPU checks are not represented
  as completed evidence. Contradictory tests and concrete code defects remain
  blockers, and generic admission does not prove a proposal-specific mechanism.
- Analyst receives required node attributes, an exact graph-scope template and a
  valid empty patch. Graph task IDs may differ from campaign task IDs. Missing
  scope/provenance errors identify the offending node and fields. The host never
  silently fills or changes a scientific graph patch.
- Research context no longer tells the researcher to edit the candidate.
- Historical cycle finalization accepts the old review shape conservatively:
  only an original approval with empty `findings` can pass. New calls require
  the new schema. Existing archives and their hashes are unchanged.

## Verification and limits

Validation on 2026-10-09: `python3 -m unittest discover -v -s tests`
completed 681 tests with zero failures (21 environment-dependent skips).

Regression tests cover positive reviews, contradictory approvals, wrong source
identities, substantive rejection, pre-admission protocol visibility, graph task
scope and immutable legacy review handling. The autonomous-controller tests
exercise a compact pipeline with mocked agents/workers, including rejection
before training.

This is a code/interface repair, not a new ten-round live Codex/GPU campaign.
Old failed rounds have not been reopened or counted as completed. Future research
can still correctly stop for source defects or invalid proposals. Per-agent
bounded working memory and cumulative token limits are separate, unimplemented
work; this patch does not claim to fix the token-volume problem.

Subsequent change: new compact campaigns now support up to two substantive
[returns for proposal/code revision](reviewer-return-for-revision.md) before
stopping. This replaces the stop-on-first-rejection behavior described above;
it does not reinterpret or reopen old campaign records.
