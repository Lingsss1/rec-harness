# Released Cross8 multistream experiments

User requires original Cross8, no mature or 24-hour filtering; no extra training seeds. Eight fresh fits: cross8, no_feedback, static, dynamic_read, dynamic_write, dynamic_bypass, dynamic_all, collapsed_all. Width256, queries8, 17 action IDs. PSGDKron every step, clip1, existing learning rates and seed.

Original scheduler PID1371138 paused after it had already started ALL eight smokes in two waves on GPU4–7. launch_eight.py preclaim guard failed (tool958854), without any mutations/model starts. Recovery launch_eight_recovery.py (session77324) waits for ALL eight existing smoke exits, terminates only the paused scheduler after worker cleanup, preserves original authorization then expands pool0–7 and starts eight formal fits. Do not resume the paused scheduler concurrently with recovery. No smoke replays. See reports/handoff-preclaim-incident.json.

Frozen DESIGN.md has stale width384/counts; actual configs are correct width256. See reports/config-clarification.json; do not edit frozen files while jobs run.

finish_measurements.py is the post-training pipeline: wait for real eight-fit completion, run paired user bootstrap (500 draws, 28 comparisons, 21 metrics, same existing seed) and saved-checkpoint latency. Check reports/postprocess for actual launch status. No model retraining for statistics.

Memory last write reports/memory-start.json: strategy only; formal totals remain194, episodes264 until actual new results close out. CAS all five memory files, preserve source receipts, append actual fresh fits only after completion. Previous record upload completed to separate research-records/rec-harness-upload branch.

All eight smokes passed; eight formal workers started GPUs0–7 (session77324). Postprocess waiting in session97203. Last memory CAS receipt is now reports/memory-formal-start.json; counts unchanged pending results.

FINAL: All 8 fits completed. Parent scheduler/waiter disappeared (cause unknown); verified every training artifact and recovered only aggregate markers. First paired statistics PID1429384 and latency PID1429385 completed. reports/postprocess-recovery/completion.json records both. Closeout committed 8 episodes; totals202 formal/272 episodes, failures22. Latest memory CAS receipt reports/closeout/completion.json. RESULTS.md, INTERPRETATION.md and MECHANISM_NOTES.md are final. No further fits started.
