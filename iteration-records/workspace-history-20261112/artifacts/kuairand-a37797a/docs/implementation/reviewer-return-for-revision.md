# Reviewer return for revision

New Amazon compact-four campaigns default to `review_revision_limit: 2`.
Allowed values are integers 0–2; zero disables the return path. Existing frozen
campaigns without this field retain their original stop-on-rejection behavior.

A valid `request_changes` or `unresolved` response now returns the pre-registration
attempt to Researcher, then Experiment, then an independent Reviewer. Researcher
receives the blocking findings, prior design, and actual prior candidate/control
source. It may correct a proposal-specific contradiction or unnecessary check,
explaining the change in proposal.guardrails. Host protocol, seeds, IDs, budget
limits and source admissibility remain enforced. Experiment receives the revised
full proposal and feedback. Reviewer checks the actual revised source against
that proposal. Formatting retries remain a separate, existing bounded mechanism.

There are at most three source reviews: the original and two revisions. With no
format retries, the compact layout therefore uses four calls on its ordinary
success path and at most ten calls on a success after two returns. Campaign
role budgets still apply; no extra budget is silently granted. A third substantive
rejection ends the round before any training or registration.

Each returned attempt is moved into `review-repair/revision-N/`, including role
contexts, schemas, responses, source and format-retry logs. A journal entry and
CAS hashes bind every file before moving it. Recovery finishes interrupted moves
idempotently and preserves the count. Audit verifies superseded attempts even
when the round fails; successful cycle archives additionally bind the complete
revision history. Final registration binds only the final reviewed cards and
source. Registered experiments cannot be revised through this path.

Regression coverage includes first/second revision success, exhaustion without
training, repeated resume, immutable past attempts, interrupted archival, source
history audit and the no-revision legacy behavior. Tests use controlled agents
and workers; no real Codex call or GPU experiment is launched by this change.

Validation: the full 690-test run exposed five online configuration-binding
failures caused by the new default. After fixing the whitelist and default
binding, all 57 tests across autonomous pipeline, revision recovery, online
harness, online tools and tool-cost suites passed. The full suite was not rerun
after that narrow compatibility correction; 21 environment-dependent tests
were skipped in the full run.
