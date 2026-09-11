# KuaiRec 失败与恢复记录

共 4 条。执行事故不等于科学假设被否定。

```json
{
  "failure_id": "round2-diagnostic-bce-autocast",
  "experiment_id": "round2-diagnostic-bce-autocast",
  "kind": "autocast_runtime_error",
  "signature": "torch.nn.functional.binary_cross_entropy and torch.nn.BCELoss are unsafe to autocast",
  "scope": {
    "component": "diagnostic backward",
    "precision": "bf16"
  },
  "reproduction": "initial diagnostic_round2.py applied BCE within model._get_amp_context()",
  "avoid": "Cast probabilities to float and compute BCE outside autocast; retain BF16 model forward",
  "artifact_path": "workspace/qformer-loop/reports/round2-middle2.log",
  "superseded_by": "diagnostics-round2-middle2-v2"
}
```

```json
{
  "failure_id": "round2-premature-diagnostic-analysis",
  "experiment_id": "round2-premature-diagnostic-analysis",
  "kind": "premature_input_read",
  "signature": "No such file or directory: hooks.json / states.npz",
  "avoid": "Wait for successful exporter completion and summary.json before consuming arrays",
  "artifact_path": "reports/round2/toolkit-post-training-premature-status.json",
  "superseded_by": "diagnostics-round2-middle2-damped50"
}
```

```json
{
  "failure_id": "round8-yaml-duplicate-anchor",
  "experiment_ids": [
    "round8-launch-yaml-r8_first_midnorm",
    "round8-launch-yaml-r8_first_half",
    "round8-launch-yaml-r8_first_double",
    "round8-launch-yaml-r8_first_prenorm",
    "round8-launch-yaml-r8_first_prenorm_half",
    "round8-launch-yaml-r8_first_midnorm_half",
    "round8-launch-yaml-r8_first_loop1",
    "round8-launch-yaml-r8_first_loop3"
  ],
  "kind": "configuration_parser_error",
  "signature": "yaml.composer.ComposerError: found duplicate anchor 'id001'",
  "reproduction": "Append a separately safe_dump-ed shallow-copy config dictionary to YAML already containing id001 anchors",
  "avoid": "Serialize independent config objects without colliding anchors; parse the entire merged YAML and verify old/new entries before freeze and launch",
  "training_updates": 0,
  "artifact_path": "workspace/qformer-loop/reports/round8/training-launch-failed/status.json",
  "recovery": "Regenerated appended entries from independent JSON objects; full merged parse and semantic equality checks passed"
}
```

```json
{
  "failure_id": "round18-yaml-duplicate-anchor",
  "experiment_ids": [
    "round18-launch-yaml-r18_base_norm_phase132",
    "round18-launch-yaml-r18_base_norm_mixed1232",
    "round18-launch-yaml-r18_field512_norm_phase132",
    "round18-launch-yaml-r18_field512_norm_mixed1232"
  ],
  "kind": "configuration_parser_error",
  "signature": "yaml.composer.ComposerError: found duplicate anchor 'id001'",
  "training_updates": 0,
  "avoid": "Serialize merged YAML document once; parse and compare every old and new entry before launch. Same failure as round8; enforce preflight.",
  "artifact_path": "workspace/qformer-loop/reports/round18/launch-failure.json"
}
```
