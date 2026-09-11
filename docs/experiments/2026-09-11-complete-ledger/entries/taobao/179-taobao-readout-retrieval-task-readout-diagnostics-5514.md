## 179. taobao-readout-retrieval-task-readout-diagnostics-5514

- 假设：Learned task-specific early-final coefficients may contribute to predictions
- 判定：provisional
- 原因：见完整记录

<details><summary>完整实验记录（指标、预算、配置、有效性、证据）</summary>

```json
{
  "experiment_id": "taobao-readout-retrieval-task-readout-diagnostics-5514",
  "timestamp": "2026-09-09T08:03:06.279538+00:00",
  "dataset_id": "taobao-unirank-release-v1",
  "parent_experiment_ids": [
    "role_q4_read_task-s20262027-20260909T063952341877Z",
    "role_q8_read_task-s20262027-20260909T063952422184Z"
  ],
  "family": "fixed_checkpoint_task_readout_intervention",
  "code_id": "eb0cfbcf5181f4e2e826cfbfa2752dd36bf93330788afa0e58e00484c3ed93e2",
  "hypothesis": "Learned task-specific early-final coefficients may contribute to predictions",
  "change": "Native versus coefficient mean and zero; fixed two task checkpoints and fixed 5514 validation rows",
  "budget": {
    "training_updates": 0,
    "new_training_seeds": [],
    "bootstrap_repetitions": 10000
  },
  "metrics": {
    "effects": [
      {
        "index": 0,
        "family": "role_q4_read_task",
        "contrast": "mean_minus_native",
        "task": "is_click",
        "estimate": 9.272547789691856e-06,
        "exploratory_ci95": [
          7.079242937049017e-07,
          1.77657778956119e-05
        ]
      },
      {
        "index": 1,
        "family": "role_q4_read_task",
        "contrast": "mean_minus_native",
        "task": "cart",
        "estimate": -2.642426061001133e-07,
        "exploratory_ci95": [
          -7.823587678982792e-07,
          6.438157811288345e-08
        ]
      },
      {
        "index": 2,
        "family": "role_q4_read_task",
        "contrast": "mean_minus_native",
        "task": "fav",
        "estimate": -1.0472039413025877e-07,
        "exploratory_ci95": [
          -6.224345006872112e-07,
          3.6594628620867614e-07
        ]
      },
      {
        "index": 3,
        "family": "role_q4_read_task",
        "contrast": "mean_minus_native",
        "task": "buy",
        "estimate": 3.338265171470046e-07,
        "exploratory_ci95": [
          -5.1116982538029e-07,
          1.2188318881409335e-06
        ]
      },
      {
        "index": 4,
        "family": "role_q4_read_task",
        "contrast": "zero_minus_native",
        "task": "is_click",
        "estimate": -0.00010323224268424735,
        "exploratory_ci95": [
          -0.0002060328477919399,
          7.678263762828285e-07
        ]
      },
      {
        "index": 5,
        "family": "role_q4_read_task",
        "contrast": "zero_minus_native",
        "task": "cart",
        "estimate": 3.9761458129794484e-05,
        "exploratory_ci95": [
          -6.837551738833088e-06,
          0.00011302299007379865
        ]
      },
      {
        "index": 6,
        "family": "role_q4_read_task",
        "contrast": "zero_minus_native",
        "task": "fav",
        "estimate": 4.722955452612568e-06,
        "exploratory_ci95": [
          -8.991736576321023e-06,
          1.9885349760817633e-05
        ]
      },
      {
        "index": 7,
        "family": "role_q4_read_task",
        "contrast": "zero_minus_native",
        "task": "buy",
        "estimate": 4.797013640290029e-06,
        "exploratory_ci95": [
          -3.3910912374851757e-06,
          1.3418727330439994e-05
        ]
      },
      {
        "index": 8,
        "family": "role_q8_read_task",
        "contrast": "mean_minus_native",
        "task": "is_click",
        "estimate": -3.2895111980031698e-06,
        "exploratory_ci95": [
          -9.627870568186343e-06,
          3.118512368463815e-06
        ]
      },
      {
        "index": 9,
        "family": "role_q8_read_task",
        "contrast": "mean_minus_native",
        "task": "cart",
        "estimate": -6.447224242376651e-06,
        "exploratory_ci95": [
          -2.63509509878373e-05,
          4.542014457418477e-06
        ]
      },
      {
        "index": 10,
        "family": "role_q8_read_task",
        "contrast": "mean_minus_native",
        "task": "fav",
        "estimate": -7.51624536646707e-07,
        "exploratory_ci95": [
          -2.366304951511593e-06,
          1.6734494464580737e-06
        ]
      },
      {
        "index": 11,
        "family": "role_q8_read_task",
        "contrast": "mean_minus_native",
        "task": "buy",
        "estimate": -4.995025759392573e-08,
        "exploratory_ci95": [
          -4.361366025807976e-07,
          3.0517262223880697e-07
        ]
      },
      {
        "index": 12,
        "family": "role_q8_read_task",
        "contrast": "zero_minus_native",
        "task": "is_click",
        "estimate": -3.8235418371633286e-05,
        "exploratory_ci95": [
          -0.0001215971666196725,
          4.547905821225699e-05
        ]
      },
      {
        "index": 13,
        "family": "role_q8_read_task",
        "contrast": "zero_minus_native",
        "task": "cart",
        "estimate": 4.6800716954359534e-05,
        "exploratory_ci95": [
          -2.9398030607420494e-05,
          0.00018528581492702927
        ]
      },
      {
        "index": 14,
        "family": "role_q8_read_task",
        "contrast": "zero_minus_native",
        "task": "fav",
        "estimate": -9.228901644827389e-06,
        "exploratory_ci95": [
          -2.9265434446091327e-05,
          2.1450234972230498e-05
        ]
      },
      {
        "index": 15,
        "family": "role_q8_read_task",
        "contrast": "zero_minus_native",
        "task": "buy",
        "estimate": 2.8230040530210573e-06,
        "exploratory_ci95": [
          -5.915895837819814e-06,
          1.2351125960681868e-05
        ]
      }
    ],
    "coefficients": {
      "role_q4_read_task": {
        "native": {
          "dtype": "torch.float32",
          "shape": [
            4
          ],
          "values": [
            0.042576491832733154,
            0.0457356832921505,
            0.044561173766851425,
            0.05137879401445389
          ],
          "sha256": "23097e345a642cc54f31583ea0aabe72c5a329970a9db0874a8fd004dbcef0d9"
        },
        "mean": {
          "dtype": "torch.float32",
          "shape": [
            4
          ],
          "values": [
            0.04606303572654724,
            0.04606303572654724,
            0.04606303572654724,
            0.04606303572654724
          ],
          "sha256": "e4e305ef97b1a4cf3d7d47e561546dd70e6c9e332317225c13dab7956b5d7528"
        },
        "zero": {
          "dtype": "torch.float32",
          "shape": [
            4
          ],
          "values": [
            0.0,
            0.0,
            0.0,
            0.0
          ],
          "sha256": "45afc7e80dcd698424ef4bdc903b2ae7fcc1118eecfb4eacad49c12b3e13e6a5"
        }
      },
      "role_q8_read_task": {
        "native": {
          "dtype": "torch.float32",
          "shape": [
            4
          ],
          "values": [
            0.06769133359193802,
            0.05451339855790138,
            0.06781519204378128,
            0.0600215382874012
          ],
          "sha256": "b05ed226141c018518acd493ed6ffa5c1c167497b51606b9236d8e71bf2d3b43"
        },
        "mean": {
          "dtype": "torch.float32",
          "shape": [
            4
          ],
          "values": [
            0.06251037120819092,
            0.06251037120819092,
            0.06251037120819092,
            0.06251037120819092
          ],
          "sha256": "31bc79e8568b13a90cb600d6d92d8d960dc897889e7aa1a39cc832c9769b8794"
        },
        "zero": {
          "dtype": "torch.float32",
          "shape": [
            4
          ],
          "values": [
            0.0,
            0.0,
            0.0,
            0.0
          ],
          "sha256": "45afc7e80dcd698424ef4bdc903b2ae7fcc1118eecfb4eacad49c12b3e13e6a5"
        }
      }
    },
    "cost": {
      "wall_seconds": 91.852892100811,
      "cpu_seconds": 311.422337604,
      "peak_rss_bytes": 2759045120,
      "gpu_seconds": 0,
      "condition_row_forwards": 33084
    }
  },
  "validity": {
    "valid": true,
    "violations": [],
    "test_evaluated": false,
    "scope": "Exploratory uncorrected 95% intervals conditional on selected rows/checkpoints; no retraining gain"
  },
  "decision": "provisional",
  "confidence": "conditional_readout_dependency_only",
  "global_best_changed": false,
  "artifact_paths": [
    "workspace/qformer-loop/taobao-transfer/recurrence-readout-retrieval/reports/readout-diagnostics/aggregate/aggregate.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-readout-retrieval/reports/readout-result-review.json",
    "workspace/qformer-loop/taobao-transfer/recurrence-readout-retrieval/reports/interpretation.json"
  ],
  "next_question": "Task-dependent loop benefit does not establish improved early/final readout utility"
}
```

</details>
