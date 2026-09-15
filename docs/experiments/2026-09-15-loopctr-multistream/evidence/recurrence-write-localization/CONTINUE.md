# Write-localization study

User: further research and update lineage for EVERY experiment. Requirement persisted in qformer-loop/AGENTS.md and memory/strategy.json.lineage_policy. Canonical interactive graph: qformer-loop/lineage/index.html, generator lineage/update.py. Graph contains505 recorded episodes +8 current planned/running models; separate architecture and comparison edges. Every worker start/terminal refreshes graph; closeout refreshes after ledger append.

Eight new fits, original released Cross8 action17/width256/q8/seed20262027/PSGDKron every-step/clip1/2193updates. No mature, no additional training seeds. first_only,second_only,collapsed_write,tied_write,no_static_routes,first_only_no_routes,second_only_no_routes,collapsed_no_routes. Structural parent previous dynamic_write. Two frozen prior references Cross8 and dynamic_write reused WITHOUT training replay. See DESIGN.md and lineage-plan.json.

Detached pipeline PID1437242, reports/dispatch.json, training workers on8 GPUs. Check reports/pipeline and per-model smoke/formal exit receipts; no automatic retries. Pipeline handles all8 smokes → all8 formal → paired statistics + latency → closeout/memory + lineage. Frozen91sources; do not edit files bound by authorization while work is active.

Offline model tests b633fe exit0; statistics tests51ce9e exit0. Full mode nonzero forward matches old trained architecture exactly; new variants zero parity, explicit two-stream forward/all-gradient oracle, no prediction feedback checked. Actual smoke verifies BF16, finite gradient and live parameters. All model initializers must match the previous reference original parameters.

10model statistics:2 old references+8new;45pairwise effects,21metrics each,500 shared user draws seed20260906, no test. Latency all10saved checkpoints on GPU4,1024rows,10warm+40timed=500forwards, no training. closeout appends exactly8 new fits if all successful.

Latest five-file memory CAS receipt reports/memory-start.json; baseline counts202formal,272Taobaoepisodes,22failures. Only strategy changed at start. Closeout expected210formal/280episodes. Global incumbent unchanged pending interpretation. On concurrent memory edits closeout refuses and preserves results for manual reconciliation.

All8 smokes passed. All8 formal workers started and graph updated to formal_running; review reports/formal-start-review.json. Full initializer of each new variant exactly matches previous dynamic_write (32de0c92...). Keep memory-start receipt unchanged while automatic closeout is pending; actual live phase in reports and graph.
