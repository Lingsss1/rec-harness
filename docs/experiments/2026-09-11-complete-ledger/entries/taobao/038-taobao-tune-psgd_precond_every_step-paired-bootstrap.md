## 38. taobao-tune-psgd_precond_every_step-paired-bootstrap

- 假设：Conditional optimizer gain
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "timestamp": "2026-09-07T16:14:08.327355+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "experiment_id": "taobao-tune-psgd_precond_every_step-paired-bootstrap",
  "family": "fixed_checkpoint_user_bootstrap",
  "hypothesis": "Conditional optimizer gain",
  "change": "500 paired user resamples",
  "budget": {
    "training_updates": 0,
    "new_training_seeds": []
  },
  "metrics": {
    "reference": "psgd_w384_single",
    "candidate": "psgd_precond_every_step",
    "users": 268201,
    "repetitions": 500,
    "bootstrap_seed": 20260906,
    "delta_auc": {
      "is_click": 0.0010580838164764295,
      "cart": 0.001761683024517624,
      "fav": 0.0013861377414385867,
      "buy": 0.00042368012181936976
    },
    "ci95_auc": {
      "is_click": [
        0.0004466133349234708,
        0.0016088516133680837
      ],
      "cart": [
        0.00023341649852549607,
        0.0034104389203389427
      ],
      "fav": [
        -4.787908942856519e-05,
        0.002860583318306281
      ],
      "buy": [
        -0.0015296894601300958,
        0.002513995762007355
      ]
    },
    "delta_macro_auc": 0.0011573961760629192,
    "ci95_macro_auc": [
      0.0003937883625616055,
      0.0019926658801359687
    ],
    "scope": "Conditional on fixed checkpoints and released validation protocol; does not capture training-seed variance or correct for candidate selection; correlated heads share bootstrap user weights"
  },
  "decision": "provisional",
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/optimizer-psgd-tune/reports/paired-bootstrap.json"
  ]
}
```

</details>
