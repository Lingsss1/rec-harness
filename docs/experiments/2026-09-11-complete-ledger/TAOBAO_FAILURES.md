# Taobao 失败与恢复记录

共 20 条。执行事故不等于科学假设被否定。

```json
{
  "failure_id": "alternative-missing-accumulation",
  "kind": "model_contract_error",
  "training_updates": 0,
  "artifact": "round2/reports/alternative-cpu-initial-failure.txt",
  "repair": "Added accumulation_steps, CPU5 subsequently passed"
}
```

```json
{
  "failure_id": "alternative-poststep-gradient-fixture",
  "kind": "test_fixture_error",
  "training_updates": 0,
  "artifact": "round2/reports/alternative-cpu-gradient-fixture-failure.txt",
  "repair": "Inspect gradients after explicit backward, not after optimizer clears gradients"
}
```

```json
{
  "failure_id": "capacity-four-task-fixture",
  "kind": "test_fixture_error",
  "training_updates": 0,
  "artifact": "round2/reports/cpu-task-fixture-failure.txt",
  "repair": "Set scalar binary_classification task in four-task fixture; CPU/GPU5 passed"
}
```

```json
{
  "failure_id": "query-parts-item-index-contract",
  "kind": "diagnostic_export_contract",
  "training_updates": 0,
  "signature": "KeyError: item_id",
  "repair": "Use batch item_index, matching frozen training prediction exporter; retain item_id as NPZ archive key",
  "artifact_path": "qformer-loop/taobao-transfer/query-conditioning/reports/attempts/query-parts-item-key"
}
```

```json
{
  "failure_id": "query-correction-zero-a-psgd-init",
  "kind": "optimizer_initialization_contract",
  "formal_training_updates": 0,
  "partial_smoke_steps_may_have_updated_other_parameters": true,
  "affected_families": [
    "query_q4_difference",
    "query_q4_entry",
    "query_q4_mean",
    "query_q4_work",
    "query_q8_difference",
    "query_q8_entry",
    "query_q8_mean",
    "query_q8_work"
  ],
  "artifact": "workspace/qformer-loop/taobao-transfer/recurrence-query-correction/reports",
  "repair": "Defer A optimizer initialization until first nonzero gradient; preserve optimizer options and exact model initialization; validate two real-data steps."
}
```

```json
{
  "failure_id": "query-route-export-global-ceiling-budget",
  "kind": "diagnostic_contract",
  "training_updates": 0,
  "repair": "Use audited per-payload release budget 2193 steps and 17822296 rows; global ceil is invalid for this loader.",
  "artifact": "workspace/qformer-loop/taobao-transfer/query-correction-diagnostics/reports/attempts/global-ceiling-budget"
}
```

```json
{
  "failure_id": "query-route-export-symlink-root",
  "kind": "diagnostic_contract",
  "training_updates": 0,
  "repair": "Resolve metadata from data_root/dataset_id and valid input containment from resolved valid symlink target.",
  "artifact": "workspace/qformer-loop/taobao-transfer/query-correction-diagnostics/reports/attempts/symlink-data-root"
}
```

```json
{
  "failure_id": "query-correction-q4-difference-second-psgd-init",
  "kind": "optimizer_initialization_contract",
  "formal_training_updates": 0,
  "smoke_first_step_completed": true,
  "family": "query_q4_difference",
  "artifact": "workspace/qformer-loop/taobao-transfer/recurrence-query-correction-v2/reports/query_q4_difference/smoke.log",
  "repair": "Pending instrumented reproduction; preserve locked optimizer settings and completed seven candidates."
}
```

```json
{
  "failure_id": "readout-retrieval-latency-v1-input-tree",
  "kind": "measurement_input_adapter",
  "signature": "AttributeError: list object has no attribute detach",
  "training_updates": 0,
  "artifact_path": "workspace/qformer-loop/taobao-transfer/recurrence-readout-retrieval/reports/latency-failure-reconciliation.json",
  "avoid": "Fingerprint and transfer the full four-part nested tensor tree.",
  "resolved_by": "benchmark_latency_v2.py"
}
```

```json
{
  "failure_id": "readout-retrieval-latency-v2-process-namespace",
  "kind": "measurement_device_identity",
  "signature": "GPU has another compute process: [506366]",
  "training_updates": 0,
  "model_forwards_exact": null,
  "model_forwards_upper_bound": 50,
  "artifact_path": "workspace/qformer-loop/taobao-transfer/recurrence-readout-retrieval/reports/latency-v2-failure-reconciliation.json",
  "avoid": "Do not equate local PID with NVML PID; bind the exclusive context through verified allocation/release and retain strict same-UUID process-set checks.",
  "resolved_by": "benchmark_latency_v3.py"
}
```

```json
{
  "failure_id": "direction-norm-config-digest-convention-metadata",
  "experiment_id": "taobao-second-query-direction-norm-5514",
  "kind": "metadata_identity_convention",
  "signature": "Protocol fixed checkpoint/config differs from archived reference",
  "scope": {
    "stage": "read_only_preexecution_check",
    "model_forwards": 0,
    "dataset_traversals": 0
  },
  "likely_cause": "Historical structural report and canonical references hash identical JSON config with different serialization conventions.",
  "avoid": "Compare exact config values and explicitly bind canonical compact sorted JSON hash; retain the other convention only as provenance.",
  "artifact_path": "workspace/qformer-loop/taobao-transfer/recurrence-query-direction-norm/reports/metadata-reconciliation.json",
  "resolved": true,
  "scientific_hypothesis_failed": false
}
```

```json
{
  "failure_id": "directional-retrieval-cuda-autocast-source-query-dtype",
  "kind": "implementation_native_mixed_precision",
  "signature": "Source and raw Q must have matching nonempty shapes/dtypes/devices",
  "scope": {
    "failed_native_smokes": 6,
    "failed_smoke_optimizer_updates": 0,
    "formal_training_runs": 0
  },
  "artifact_path": "workspace/qformer-loop/taobao-transfer/recurrence-directional-retrieval/reports/implementation-failure.json",
  "avoid": "CUDA autocast dtype policy is operation-specific: test native LayerNorm FP32 plus SDPA BF16 sources; allow both floating dtypes before existing FP32 geometry instead of forcing original queries to lower precision.",
  "resolved_by": "workspace/qformer-loop/taobao-transfer/recurrence-directional-retrieval-v2/QFormerCross8DirectionalRetrieval.py",
  "tiny_cuda_verified": true,
  "full_corrected_native_smokes": "running",
  "scientific_hypothesis_failed": false
}
```

```json
{
  "failure_id": "directional-retrieval-cuda-autocast-source-query-dtype-resolved-v2",
  "supersedes_failure_id": "directional-retrieval-cuda-autocast-source-query-dtype",
  "kind": "implementation_fix_verification",
  "timestamp": "2026-09-09T09:45:45.375218+00:00",
  "resolved": true,
  "full_corrected_native_smokes": "all_eight_passed",
  "verification": "workspace/qformer-loop/taobao-transfer/recurrence-directional-retrieval-v2/reports/smoke-review.json",
  "independent_verification": "workspace/qformer-loop/taobao-transfer/recurrence-directional-retrieval-v2/reports/smoke-independent-review.json",
  "scientific_hypothesis_failed": false,
  "formal_quality": "pending_eight_active_runs",
  "scope": "Preserve the original failure record; append the correction and verification as a separate event."
}
```

```json
{
  "failure_id": "endpoint-measurement-import-and-analysis-interruption",
  "kind": "execution_incident_with_verified_recovery",
  "timestamp": "2026-09-10T15:24:31.235440+00:00",
  "error": "KeyError endpoint_read_model after200referenceforwards; pipeline terminated analysis-15",
  "resolved": true,
  "resolution": "Add actualfactorymodule import; preserve200referenceforwards andcontinue150new; oneexplicit500drawrecovery witholdattemptdrawcountunknown",
  "scientific_hypothesis_failed": false,
  "formal_training_runs_failed": 0,
  "evidence_sha256": {
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/latency/failure.json": "f2693b36d0fd1683a0aa4374d8a158c1c1ae8dabc1d43ab022fd30de824abf1e",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/measurement-pipeline/failure.json": "dabf8f634318242003b4663e45e4a4956bb057776cdaafab46e3b57bd8a7a218",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/latency-recovery/completion.json": "e255d284cf4cdc93e9d2841e9d822d6950771806d1855a83c998033c5f488583",
    "workspace/qformer-loop/taobao-transfer/recurrence-endpoint-read/reports/paired-analysis-recovery/completion.json": "4a37782006080bbc623b6cd920b470f4a1446d4f8313e9bfa243c0afc004893e"
  },
  "scope": "Incident record, not an extra formal fit or training seed. Original failure artifacts remain immutable."
}
```

```json
{
  "failure_id": "round-accumulation-preclaim-launch-lock",
  "kind": "execution_incident_with_verified_recovery",
  "timestamp": "2026-09-10T15:24:31.235440+00:00",
  "error": "Global inherited launcher lock prevented disjointGPU launch beforeworkers",
  "resolved": true,
  "resolution": "Study-local launcher lock, unchanged sharedphysicalGPUlock; nofitreplay",
  "scientific_hypothesis_failed": false,
  "formal_training_runs_failed": 0,
  "evidence_sha256": {
    "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/reports/launch-lock-collision/event.json": "d9c6b8cd69332286623d7eaacdb8c4a081db4f6427b5522c20a5b1e6866b57c9"
  },
  "scope": "Incident record, not an extra formal fit or training seed. Original failure artifacts remain immutable."
}
```

```json
{
  "failure_id": "round-accumulation-smoke-verifier-missing-params",
  "kind": "execution_incident_with_verified_recovery",
  "timestamp": "2026-09-10T15:24:31.235440+00:00",
  "error": "All4runners0after8updates butworkers126becauseouterinitializercheck omittedparams",
  "resolved": true,
  "resolution": "Revalidate allsavedchecks withmissingparamssupplied, preservefailedterminals,formalwrapperfix; no smokereplay",
  "scientific_hypothesis_failed": false,
  "formal_training_runs_failed": 0,
  "evidence_sha256": {
    "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/reports/launch-smoke/completion.json": "37882051c8b94d6ae8eaa996639178056b8331a5bb34c871f7a62a560f1143d7",
    "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/reports/recovered-smoke-review.json": "ae5612bb1c50bf9b209f454f2945d0ebaa3b5000a112ea174fb8e1c5fbe16559"
  },
  "scope": "Incident record, not an extra formal fit or training seed. Original failure artifacts remain immutable."
}
```

```json
{
  "failure_id": "round-accumulation-subset-float-user-schema",
  "kind": "execution_incident_with_verified_recovery",
  "timestamp": "2026-09-10T15:24:31.235440+00:00",
  "error": "Strict integer selector rejected exactintegralfloat64 encodeduserIDs",
  "resolved": true,
  "resolution": "Validatefinite safeintegerbounds andexactcast roundtrip beforefixedhashselection",
  "scientific_hypothesis_failed": false,
  "formal_training_runs_failed": 0,
  "evidence_sha256": {
    "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/reports/subset-schema-incident.json": "832a82dc2ca4e076248d5b1e77c600373155b908c28ab019c275cbafb0e30565",
    "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/reports/diagnostic-subset/manifest.json": "f36ef6145355dc832175e069c6aa9263f387db8106c5bb5f88d999b733893ad1"
  },
  "scope": "Incident record, not an extra formal fit or training seed. Original failure artifacts remain immutable."
}
```

```json
{
  "failure_id": "round-accumulation-preclaim-metadata-symlink",
  "kind": "execution_incident_with_verified_recovery",
  "timestamp": "2026-09-10T15:24:31.235440+00:00",
  "error": "Canonical inputauthorization lookup used unresolvedmeta_data path",
  "resolved": true,
  "resolution": "Resolvepath beforehashlookup; failedbeforeloaderpass",
  "scientific_hypothesis_failed": false,
  "formal_training_runs_failed": 0,
  "evidence_sha256": {
    "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/reports/data-preclaim-path-incident.json": "80247f6cc471f573ab81247480584aec75c197ceec67dd99ca80058e68880d7b"
  },
  "scope": "Incident record, not an extra formal fit or training seed. Original failure artifacts remain immutable."
}
```

```json
{
  "failure_id": "round-accumulation-native-item-index-schema",
  "kind": "execution_incident_with_verified_recovery",
  "timestamp": "2026-09-10T15:24:31.235440+00:00",
  "error": "Firstnativebatch export expecteditem_id instead ofactualitem_index",
  "resolved": true,
  "resolution": "Use originalrunner export mapping; preserve firstfailedclaim withnooutputbatch andcompleteonefullnativepass",
  "scientific_hypothesis_failed": false,
  "formal_training_runs_failed": 0,
  "evidence_sha256": {
    "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/reports/diagnostic-data/failure.json": "06b71c0c70602514224b70779bb2704cc068f7351f1cab00b5052434d102b857",
    "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/reports/diagnostic-data-v2/completion.json": "073e0cc968d4f1e54be3bfea704d8fd820a576f24c73e12d77cbc204ad23e420"
  },
  "scope": "Incident record, not an extra formal fit or training seed. Original failure artifacts remain immutable."
}
```

```json
{
  "failure_id": "round-accumulation-postprocess-path-type",
  "kind": "execution_incident_with_verified_recovery",
  "timestamp": "2026-09-10T15:44:40.685443+00:00",
  "stage": "postprocess_dispatch_before_child",
  "error": "AttributeError(\"'str' object has no attribute 'read_bytes'\")",
  "resolved": true,
  "scientific_hypothesis_failed": false,
  "formal_training_runs_failed": 0,
  "training_restarts": 0,
  "evidence_sha256": {
    "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/reports/measurement-pipeline/failure.json": "8f49831d5dbb15d74175059fcfaba37ccb7121e39b70be6f8ed259e4323ba8aa",
    "workspace/qformer-loop/taobao-transfer/recurrence-round-accumulation/reports/measurement-pipeline-recovery/completion.json": "cd0522ce92cc72ccd96536624b2ffad5bdbf20aee3bcd20936aef343663a5a59"
  },
  "resolution": "Fix dispatcher Path conversion before any child; same numerical and model sources, no measurement replay.",
  "scope": "Execution incident only; not an extra formal fit or seed."
}
```
