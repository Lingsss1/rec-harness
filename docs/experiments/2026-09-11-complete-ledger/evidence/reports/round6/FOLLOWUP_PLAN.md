# Round 6 follow-up: four combinations, same existing seed

Added after seven initial runs completed. This is adaptive structural search,
not part of the eight-way initial matrix. Original plan/config hashes remain
in frozen-identity.json; the original whole YAML is archived as
phase1-model_config.yaml before appending these configurations.

Evidence: field-after AUC 0.870420 versus parent 0.870197; removing the field
prelude gives 7.852M parameters and AUC 0.869915; extra attention after history
gives AUC 0.869787 and NDCG 0.412305. All are seed 20262027 only.

| Family | Combination | Schedule |
|---|---|---|
| r6_small_after | terminal feedback plus remove prelude | N0 S0 S0 N0(context) |
| r6_after_all | terminal feedback reads all fields | N0 N1 S0 S0 N1(all) |
| r6_after_target | terminal feedback reads target fields | N0 N1 S0 S0 N1(target) |
| r6_small_attention_after | remove prelude plus attention after history | N0 S0 A N0(context) S0 A |

Use unchanged model implementation and training/data budget. Exactly four
new complete runs, seed 20262027; no new seed values or replications. CPU/GPU
combination checks cover actual call graph, source tensors, gradients and mask
invariance before launch. Do not choose from diagnostic-subset AUC.
