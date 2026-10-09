# Candidate state and registry schema

`manage_candidate.py` stores mutable candidate state in `candidate.json` and immutable events in `memory/tool-evolution/registry.jsonl`.

## Lifecycle

```text
idea -> researched -> implemented -> synthetic_pass -> shadow -> accepted
  |          |             |              |           |
  +----------+-------------+--------------+-----------> rejected
accepted ------------------------------------------------> superseded
```

Use a new ID for a materially different estimator. Revisions that preserve the estimator and decision contract may continue under the same ID, with every gate change recorded.

## Required validation gates

- `paper_fidelity`
- `synthetic_correctness`
- `robustness`
- `shadow_real_data`
- `incremental_utility`
- `cost`

Each gate has `pending`, `pass`, or `fail`, an evidence path, note, and update timestamp. `accepted` is blocked until every gate passes and `acceptance.preset_before_results` is true.

## Event fields

Every registry line includes:

```json
{
  "timestamp": "ISO-8601 UTC",
  "candidate_id": "gradient-snr-v1",
  "event": "initialized|gate_recorded|transitioned|promoted",
  "from_status": null,
  "to_status": "idea",
  "gate": null,
  "result": null,
  "evidence": null,
  "note": "compact attributable fact"
}
```

Never delete or edit registry lines. Candidate state may be reconstructed by replaying them; `candidate.json` is a convenient current view.

## Promotion event

After integrating into the stable toolkit, append an event whose note contains the toolkit source path, promoted files, code hash/commit, tests run, and rollback identity. The registry records tool evolution; model research conclusions remain in the Harness updater's memory.
