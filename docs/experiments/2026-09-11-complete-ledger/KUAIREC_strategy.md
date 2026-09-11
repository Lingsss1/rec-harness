# KuaiRec strategy.json 快照

此处保留历史判断，不把待执行计划当作已完成实验。

```json
{
  "schema_version": 1,
  "incumbent": {
    "experiment_id": "cross8_original",
    "primary_metric": 0.8693640263652379,
    "role": "primary baseline; no loop model promoted"
  },
  "phase": "explore",
  "plateau": {
    "consecutive_misses": 80,
    "recent_slope": null,
    "definition": "Rounds8-17: 76 screens did not exceed r7 AUC; ending curricula at2 partly recovers AUC, next isolate depth-dependent readout normalization",
    "scope": "rounds8-18 single-existing-seed candidates vs r7 AUC"
  },
  "active_hypothesis": "Optimize Taobao metrics: increased capacity plus shared recurrence, six runs active",
  "exhausted_families": [
    "r7 parent: tested mixer scales and norm positions at seed20262027/one epoch",
    "r7 parent: tested feedback permutations and pass-specific interface norms at seed20262027/one epoch",
    "r7 parent: tested selective repeated sublayer deletion with paired norm deletion at seed20262027/one epoch",
    "r7 parent: eight tested zero-initialized embeddings/Query-gate repeat-conditioning variants, seed20262027/one epoch",
    "r7 parent: eight complete sublayer reorder variants, existing seed20262027 and one epoch",
    "r7 parent: eight cloned-attention history writers across Query/self sources, placements and initial scales; existing seed20262027/one epoch",
    "r7 parent: budget-matched width680 SwiGLU/GEGLU at field/history/both and dense width1024 SiLU history/both, seed20262027 one epoch",
    "r7 parent: plain GELU FFN512/680 field/history/both and1536 history/both, existing seed20262027 one epoch",
    "deterministic alternate13, mixed1232 and half1-half3 on r7/field512, fixed eval2, seed20262027 one epoch",
    "equal depth-count phase132/312/123 curricula at r7 and field512, fixed eval2, existing seed20262027 one epoch",
    "Depth-specific final norm under phase132/mixed1232 at two capacities, seed20262027 epoch1 eval2"
  ],
  "next_queue": [
    {
      "task": "taobao_round2_capacity_loop",
      "state": "running",
      "seed": 20262027
    }
  ],
  "rejected_memory_edits": [],
  "last_consolidated_episode": "diagnostics-round18-r18_field512_norm_phase132",
  "candidate_policy": "Retain models close to Cross8 OR Cross11; candidate retention is separate from incumbent promotion",
  "tool_evolution_scan": {
    "ready_handoffs": 0,
    "note": "No handoff files exist; coevolve check requires a concrete task ID. Existing stable probes cover current questions."
  },
  "best_loop_quality_candidate": {
    "family": "attn_full",
    "auc_mean": 0.8704417624874181,
    "incumbent_promoted": false,
    "scope": "previous three-seed mean; retained reference, not compared directly with rounds6-18 single-seed scores"
  },
  "preferred_placement_candidate": {
    "family": "r6_field_after",
    "seed": 20262027,
    "auc": 0.8704204186010989,
    "incumbent_promoted": false,
    "scope": "single-seed structural screen; not comparable to old three-seed means"
  },
  "rounds_completed": [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18
  ],
  "total_training_runs": 167,
  "completed_round4": [
    {
      "hypothesis": "feedback_target",
      "status": "completed",
      "seeds": [
        20262027,
        20262028,
        20262029
      ]
    },
    {
      "hypothesis": "feedback_context",
      "status": "completed",
      "seeds": [
        20262027,
        20262028,
        20262029
      ]
    },
    {
      "hypothesis": "feedback_no_norm",
      "status": "completed",
      "seeds": [
        20262027,
        20262028,
        20262029
      ]
    },
    {
      "hypothesis": "field_norm_matched",
      "status": "completed",
      "seeds": [
        20262027,
        20262028,
        20262029
      ]
    }
  ],
  "completed_round5": [
    {
      "hypothesis": "compact_seq2",
      "status": "completed",
      "seeds": [
        20262027,
        20262028,
        20262029
      ]
    },
    {
      "hypothesis": "compact_seq1",
      "status": "completed",
      "seeds": [
        20262027,
        20262028,
        20262029
      ]
    },
    {
      "hypothesis": "attn_full",
      "status": "completed",
      "seeds": [
        20262027,
        20262028,
        20262029
      ]
    },
    {
      "hypothesis": "attn_seq2",
      "status": "completed",
      "seeds": [
        20262027,
        20262028,
        20262029
      ]
    },
    {
      "hypothesis": "attn_seq1",
      "status": "completed",
      "seeds": [
        20262027,
        20262028,
        20262029
      ]
    },
    {
      "hypothesis": "attn_small",
      "status": "completed",
      "seeds": [
        20262027,
        20262028,
        20262029
      ]
    }
  ],
  "seed_policy": "User requests no additional seeds; use existing 20262027 only, no seed extensions or replications",
  "completed_round6_initial": [
    {
      "hypothesis": "r6_shallow",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r6_small",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r6_field_before",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r6_field_after",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r6_history_only",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r6_loop3",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r6_attention_after",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r6_untied",
      "status": "completed",
      "seeds": [
        20262027
      ]
    }
  ],
  "user_constraints": {
    "no_additional_seeds": true,
    "single_existing_seed_for_new_structures": 20262027
  },
  "completed_round6": [
    {
      "hypothesis": "r6_shallow",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r6_small",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r6_field_before",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r6_field_after",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r6_history_only",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r6_loop3",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r6_attention_after",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r6_untied",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r6_small_after",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r6_after_all",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r6_after_target",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r6_small_attention_after",
      "status": "completed",
      "seeds": [
        20262027
      ]
    }
  ],
  "single_seed_candidates": {
    "r6_field_after": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.189043428007618,
        "AUC": 0.8704204186010989,
        "UAUC": 0.7905194700370222,
        "gAUC": 0.8030636194155752,
        "NDCG(10)": 0.40943247540043815
      },
      "role": "AUC-first structural candidate"
    },
    "r6_small_attention_after": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18856654845935533,
        "AUC": 0.8701526672193279,
        "UAUC": 0.7889198471985921,
        "gAUC": 0.8012094365187358,
        "NDCG(10)": 0.41003584900560863
      },
      "role": "compact attention candidate"
    },
    "r6_small": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18921211597398455,
        "AUC": 0.8699149407735861,
        "UAUC": 0.7902767594942809,
        "gAUC": 0.8029255244611984,
        "NDCG(10)": 0.4067549264699934
      },
      "role": "smaller faster recurrent candidate"
    },
    "r6_shallow": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18899200385492332,
        "AUC": 0.8693636981379977,
        "UAUC": 0.7895778032869896,
        "gAUC": 0.8015644239764229,
        "NDCG(10)": 0.40596422664368786
      },
      "role": "low-cost no-loop control"
    },
    "r7_small_attention_first": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.1880668584773958,
        "AUC": 0.8712340584580865,
        "UAUC": 0.7908370127304348,
        "gAUC": 0.803790662166209,
        "NDCG(10)": 0.40985615692370087
      },
      "role": "preferred quality/cost structural candidate"
    },
    "r7_small_midnorm": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18823720725206547,
        "AUC": 0.870780429478175,
        "UAUC": 0.7878941613308428,
        "gAUC": 0.8000851537980701,
        "NDCG(10)": 0.40647084742076683
      },
      "role": "secondary norm-placement candidate; UAUC/NDCG tradeoff"
    },
    "r7_terminal_query_memory": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18883086776515312,
        "AUC": 0.8698049388169445,
        "UAUC": 0.7906647151867764,
        "gAUC": 0.8025360883057836,
        "NDCG(10)": 0.41048321597712595
      },
      "role": "near-baseline ranking/logloss tradeoff"
    },
    "r8_first_midnorm": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.1887350927707119,
        "AUC": 0.8708634281514931,
        "UAUC": 0.7915098311294028,
        "gAUC": 0.8035485749908888,
        "NDCG(10)": 0.4103131965634268
      },
      "role": "higher UAUC/gAUC/NDCG than parent at similar cost; retained ranking tradeoff"
    },
    "r8_first_half": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18919575941522698,
        "AUC": 0.8691357179460785,
        "UAUC": 0.7902444191060852,
        "gAUC": 0.8024601965389518,
        "NDCG(10)": 0.40833956372049923
      },
      "role": "deprioritized post-norm half-residual control"
    },
    "r8_first_double": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18839468752466618,
        "AUC": 0.8707643231648986,
        "UAUC": 0.7903091189439009,
        "gAUC": 0.8021587066909625,
        "NDCG(10)": 0.4109657966141899
      },
      "role": "higher NDCG but lower AUC/UAUC than parent; retained alternative"
    },
    "r8_first_prenorm": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18840056543786562,
        "AUC": 0.8705906523287695,
        "UAUC": 0.790212587262865,
        "gAUC": 0.8020129357705934,
        "NDCG(10)": 0.41030129268821564
      },
      "role": "near-baseline alternative norm position; lower AUC than parent"
    },
    "r8_first_prenorm_half": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18851271862715885,
        "AUC": 0.8709844984229478,
        "UAUC": 0.7886379071450763,
        "gAUC": 0.8006169414610279,
        "NDCG(10)": 0.41101574321817647
      },
      "role": "near-parent AUC and better NDCG, but lower UAUC; retained alternative"
    },
    "r8_first_midnorm_half": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.1894293666911448,
        "AUC": 0.8690580688192202,
        "UAUC": 0.790112826082579,
        "gAUC": 0.8023195540834577,
        "NDCG(10)": 0.40846734064899204
      },
      "role": "deprioritized norm and half-residual combination"
    },
    "r8_first_loop1": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.188918735765551,
        "AUC": 0.8700957408328922,
        "UAUC": 0.792500634435548,
        "gAUC": 0.8040155681307842,
        "NDCG(10)": 0.4126499914423532
      },
      "role": "preferred low-cost ranking candidate; lower AUC than two cycles"
    },
    "r8_first_loop3": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18975302926662113,
        "AUC": 0.8680708028914691,
        "UAUC": 0.7897708624723025,
        "gAUC": 0.8018282584698708,
        "NDCG(10)": 0.40894145585404595
      },
      "role": "deprioritized depth control; lower quality and greater forward cost"
    },
    "r9_feedback_omit": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18893191037466942,
        "AUC": 0.8697117529770065,
        "UAUC": 0.7888708392230934,
        "gAUC": 0.8014473787762356,
        "NDCG(10)": 0.4056043491975313
      },
      "role": "low-cost mechanism control; dominated by earlier one-cycle candidate"
    },
    "r9_history_replace": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18867652247312922,
        "AUC": 0.8705609932560472,
        "UAUC": 0.7884863041466431,
        "gAUC": 0.8006438554942795,
        "NDCG(10)": 0.40484413070561776
      },
      "role": "mechanism candidate above Cross8/Cross11 AUC; lower ranking and greater forward cost than parent"
    },
    "r9_feedback_before": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18937419112054044,
        "AUC": 0.8692126804101246,
        "UAUC": 0.790496275904195,
        "gAUC": 0.8021018288082645,
        "NDCG(10)": 0.4071012078769103
      },
      "role": "near-Cross8 placement control; lower AUC than parent"
    },
    "r9_feedback_after": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18976953955524276,
        "AUC": 0.867980081856248,
        "UAUC": 0.7896835535830261,
        "gAUC": 0.8019457224197174,
        "NDCG(10)": 0.40713484743568085
      },
      "role": "deprioritized terminal-feedback control"
    },
    "r9_field_splitnorm": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18853778381928812,
        "AUC": 0.8708641336820779,
        "UAUC": 0.7906243197159787,
        "gAUC": 0.8029848361664242,
        "NDCG(10)": 0.40908682804682284
      },
      "role": "near-baseline alternative; lower AUC and guardrails than shared-norm parent"
    },
    "r9_history_splitnorm": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.1887317174308961,
        "AUC": 0.8699394412892679,
        "UAUC": 0.7905937721586386,
        "gAUC": 0.8031189561424581,
        "NDCG(10)": 0.40926366935476804
      },
      "role": "near-Cross11 norm-sharing control"
    },
    "r9_both_splitnorm": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18860957541203505,
        "AUC": 0.8707487106870296,
        "UAUC": 0.789850915232056,
        "gAUC": 0.8020032445318712,
        "NDCG(10)": 0.40691784711446677
      },
      "role": "near-baseline alternative; slightly better Logloss but worse AUC/ranking than matched midnorm parent"
    },
    "r9_untied_clone": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.1885894792226225,
        "AUC": 0.8707734305814254,
        "UAUC": 0.790631600027896,
        "gAUC": 0.8027914586205532,
        "NDCG(10)": 0.4082680589304552
      },
      "role": "higher-parameter alternative; lower AUC than shared history at matched initial function"
    },
    "r10_field_cross": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18934977093635563,
        "AUC": 0.8689517856044001,
        "UAUC": 0.7900916480299852,
        "gAUC": 0.8022476317279263,
        "NDCG(10)": 0.40817130369146726
      },
      "role": "deprioritized field-cross-only control"
    },
    "r10_field_cross_ffn": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.1891617516876806,
        "AUC": 0.869620924699128,
        "UAUC": 0.7886844059162511,
        "gAUC": 0.8005955372529251,
        "NDCG(10)": 0.40482603130622496
      },
      "role": "near-Cross8 mechanism candidate; lower AUC and ranking than parent"
    },
    "r10_field_self_cross": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.1887557245242615,
        "AUC": 0.8699165060928896,
        "UAUC": 0.7913099430822921,
        "gAUC": 0.8029510586966362,
        "NDCG(10)": 0.4089041084376512
      },
      "role": "UAUC/compute tradeoff versus parent; near-Cross11 AUC"
    },
    "r10_history_cross": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.1893201958069648,
        "AUC": 0.8692701758993641,
        "UAUC": 0.7870778326387385,
        "gAUC": 0.8000272766012883,
        "NDCG(10)": 0.4056678953821851
      },
      "role": "deprioritized history-cross-only control"
    },
    "r10_history_cross_ffn": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.188681692450572,
        "AUC": 0.8699899246037345,
        "UAUC": 0.7893531013177576,
        "gAUC": 0.802087955608948,
        "NDCG(10)": 0.4072611578723409
      },
      "role": "near-Cross11 quality/cost alternative; lower AUC than full-block parent"
    },
    "r10_history_self_cross": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18906726856094166,
        "AUC": 0.8699418038536793,
        "UAUC": 0.7895299283077258,
        "gAUC": 0.8011247478032673,
        "NDCG(10)": 0.4096305628882965
      },
      "role": "near-Cross11 alternative retaining self-attention and crossing"
    },
    "r10_both_cross_ffn": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18936981044132953,
        "AUC": 0.8689075953481885,
        "UAUC": 0.7895711783637501,
        "gAUC": 0.8015484649853287,
        "NDCG(10)": 0.4064480977191721
      },
      "role": "deprioritized joint self-attention deletion control"
    },
    "r10_both_cross": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.1892173269551231,
        "AUC": 0.8693125847498258,
        "UAUC": 0.7874705294028854,
        "gAUC": 0.7999993784492027,
        "NDCG(10)": 0.40495492632678626
      },
      "role": "deprioritized fastest selective-sublayer control"
    },
    "r11_field_embedding": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18943112023718625,
        "AUC": 0.8686968740853376,
        "UAUC": 0.7904875481767265,
        "gAUC": 0.8031070812251323,
        "NDCG(10)": 0.40725945458954393
      },
      "role": "保留完整负对照；同 seed AUC 未超过父模型"
    },
    "r11_history_embedding": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18829757927226762,
        "AUC": 0.8712172199871226,
        "UAUC": 0.7906890218647017,
        "gAUC": 0.803237653010303,
        "NDCG(10)": 0.4073778451760573
      },
      "role": "本轮 AUC 最佳且接近父模型；保留历史条件化候选，Logloss/排序尚无优势"
    },
    "r11_both_embedding": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18915149514187707,
        "AUC": 0.8695677332221532,
        "UAUC": 0.7901085027149738,
        "gAUC": 0.8027660863401285,
        "NDCG(10)": 0.4111215297483523
      },
      "role": "AUC 接近 Cross8、NDCG 高于父模型；保留排序取舍对照"
    },
    "r11_query_embedding": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.1888186673131899,
        "AUC": 0.8701003147826802,
        "UAUC": 0.7896109371407005,
        "gAUC": 0.8021287588377513,
        "NDCG(10)": 0.40893103845579304
      },
      "role": "保留完整负对照；同 seed AUC 未超过父模型"
    },
    "r11_field_gate": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18855963048560487,
        "AUC": 0.8705160359393742,
        "UAUC": 0.7900086657929454,
        "gAUC": 0.8025852671614417,
        "NDCG(10)": 0.40840645148119414
      },
      "role": "保留完整负对照；同 seed AUC 未超过父模型"
    },
    "r11_history_gate": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.1888709232990787,
        "AUC": 0.8706751443699479,
        "UAUC": 0.7908537164759459,
        "gAUC": 0.8036115575080156,
        "NDCG(10)": 0.40669273776965864
      },
      "role": "AUC 高于 Cross8/Cross11，UAUC 略高父模型；保留门控对照"
    },
    "r11_both_gate": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18877703153871678,
        "AUC": 0.8701367217913544,
        "UAUC": 0.78975750977543,
        "gAUC": 0.8021999568413861,
        "NDCG(10)": 0.40720129473182953
      },
      "role": "保留完整负对照；同 seed AUC 未超过父模型"
    },
    "r11_embedding_gate": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.1885572197426099,
        "AUC": 0.8706974194785089,
        "UAUC": 0.788224715457647,
        "gAUC": 0.8007698320798486,
        "NDCG(10)": 0.4078969849195367
      },
      "role": "保留完整负对照；同 seed AUC 未超过父模型"
    },
    "r12_field_caf": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18997244677412423,
        "AUC": 0.8673217249995929,
        "UAUC": 0.7895604967062085,
        "gAUC": 0.8014845650219046,
        "NDCG(10)": 0.40777287771073645
      },
      "role": "保留结构负对照；AUC 未超过父模型"
    },
    "r12_history_caf": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18910045891788455,
        "AUC": 0.869118350983261,
        "UAUC": 0.7885780919724925,
        "gAUC": 0.8004589922061912,
        "NDCG(10)": 0.40821722686190376
      },
      "role": "保留结构负对照；AUC 未超过父模型"
    },
    "r12_both_caf": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18863022572037302,
        "AUC": 0.8702124431415598,
        "UAUC": 0.7905580360614852,
        "gAUC": 0.8023162702744755,
        "NDCG(10)": 0.4087777962002001
      },
      "role": "重复双位置重排候选，AUC 高于 Cross11，Logloss/排序优于两轮均重排"
    },
    "r12_field_caf_all": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18968354246248031,
        "AUC": 0.8691755797940852,
        "UAUC": 0.79071904251194,
        "gAUC": 0.802690589355163,
        "NDCG(10)": 0.4095447468150799
      },
      "role": "保留结构负对照；AUC 未超过父模型"
    },
    "r12_history_caf_all": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18892439930046948,
        "AUC": 0.8697436035014909,
        "UAUC": 0.7894337239936854,
        "gAUC": 0.8015913383766728,
        "NDCG(10)": 0.40744049110506725
      },
      "role": "AUC 位于 Cross8/Cross11 之间，保留基线附近候选"
    },
    "r12_both_caf_all": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18907479974294594,
        "AUC": 0.8702325940364117,
        "UAUC": 0.7894961495925994,
        "gAUC": 0.8014486581830932,
        "NDCG(10)": 0.40568424591067986
      },
      "role": "AUC 高于 Cross11，保留全轮次重排对照"
    },
    "r12_field_afc": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.1896058373750363,
        "AUC": 0.8692775971111519,
        "UAUC": 0.7901035808285571,
        "gAUC": 0.8023413647846127,
        "NDCG(10)": 0.4048470883594941
      },
      "role": "保留结构负对照；AUC 未超过父模型"
    },
    "r12_history_afc": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18852657896259667,
        "AUC": 0.8703253165429728,
        "UAUC": 0.7887872661114976,
        "gAUC": 0.8010790376209026,
        "NDCG(10)": 0.40968084324100423
      },
      "role": "本轮 AUC 首选，高于 Cross8/Cross11，低于父模型；保留来源后读方向"
    },
    "r13_query_between_zero": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18866042420141285,
        "AUC": 0.8706945725149449,
        "UAUC": 0.7913528939465213,
        "gAUC": 0.8036817705552535,
        "NDCG(10)": 0.408685036501372
      },
      "role": "本轮 AUC 最佳，UAUC 高于父模型；保留 Query 反馈候选，参数及计算额外增加"
    },
    "r13_query_between_warm": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18886233324038054,
        "AUC": 0.8698287530061466,
        "UAUC": 0.7900109402294968,
        "gAUC": 0.8018196758146277,
        "NDCG(10)": 0.4091677275316061
      },
      "role": "保留历史源更新结构对照；AUC 未超过父模型"
    },
    "r13_query_after_field_zero": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18917638166786357,
        "AUC": 0.8695069241342309,
        "UAUC": 0.7908035301124489,
        "gAUC": 0.8031138548686829,
        "NDCG(10)": 0.4090719974052154
      },
      "role": "保留历史源更新结构对照；AUC 未超过父模型"
    },
    "r13_query_after_field_warm": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18863434060158704,
        "AUC": 0.8703108920242177,
        "UAUC": 0.7894676231178814,
        "gAUC": 0.8016902218940102,
        "NDCG(10)": 0.40568237301746374
      },
      "role": "AUC 高于 Cross8/Cross11；保留回读后写历史位置对照"
    },
    "r13_self_between_zero": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18912348497970413,
        "AUC": 0.8702136128250356,
        "UAUC": 0.790486819397606,
        "gAUC": 0.8031560883916246,
        "NDCG(10)": 0.4086269433217079
      },
      "role": "历史 self-attention 本轮最高 AUC，略高 warm；保留零初始对照"
    },
    "r13_self_between_warm": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18887012784605345,
        "AUC": 0.8701882533164638,
        "UAUC": 0.7904126622692615,
        "gAUC": 0.8028674448771849,
        "NDCG(10)": 0.41035215177423034
      },
      "role": "AUC 高于 Cross11，NDCG 高于父模型；保留历史 self-attention 排序取舍"
    },
    "r13_self_before_zero": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18906496754145133,
        "AUC": 0.8697203771123306,
        "UAUC": 0.7900190089885972,
        "gAUC": 0.8023852639446871,
        "NDCG(10)": 0.40919751128001053
      },
      "role": "保留历史源更新结构对照；AUC 未超过父模型"
    },
    "r13_self_before_warm": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18906237332804204,
        "AUC": 0.8691976039702708,
        "UAUC": 0.7892738605192696,
        "gAUC": 0.8016096220774392,
        "NDCG(10)": 0.4067192431000911
      },
      "role": "保留历史源更新结构对照；AUC 未超过父模型"
    },
    "r14_swiglu_field": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18925371914700853,
        "AUC": 0.8691484171555344,
        "UAUC": 0.7883152407765224,
        "gAUC": 0.8000928391667417,
        "NDCG(10)": 0.406227180185897
      },
      "role": "保留门控/激活结构负对照，AUC 未超过父模型"
    },
    "r14_swiglu_history": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.1888243300702403,
        "AUC": 0.8701001801597251,
        "UAUC": 0.7889325351321571,
        "gAUC": 0.8014697711343842,
        "NDCG(10)": 0.4064127416652562
      },
      "role": "AUC 高于 Cross11，保留单历史门控候选"
    },
    "r14_swiglu_both": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18846546930616379,
        "AUC": 0.870200706453784,
        "UAUC": 0.7916400039597632,
        "gAUC": 0.8029887194848476,
        "NDCG(10)": 0.4104361895551997
      },
      "role": "本轮 AUC 最佳；UAUC、NDCG 高于父模型，保留参数略少的排序候选"
    },
    "r14_geglu_field": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.1894591659500588,
        "AUC": 0.8685873259452879,
        "UAUC": 0.7902062655749541,
        "gAUC": 0.8027037259817863,
        "NDCG(10)": 0.40938893755839556
      },
      "role": "保留门控/激活结构负对照，AUC 未超过父模型"
    },
    "r14_geglu_history": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18936564711521722,
        "AUC": 0.868950584433582,
        "UAUC": 0.7883841886865016,
        "gAUC": 0.8007593739475868,
        "NDCG(10)": 0.40564125912824345
      },
      "role": "保留门控/激活结构负对照，AUC 未超过父模型"
    },
    "r14_geglu_both": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.1891547402282059,
        "AUC": 0.8688209032195592,
        "UAUC": 0.7889358579521009,
        "gAUC": 0.8008459203789507,
        "NDCG(10)": 0.4105988960364575
      },
      "role": "AUC 较低但 NDCG 本轮最高，保留负结果及排序对照"
    },
    "r14_silu_history": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18943758664316496,
        "AUC": 0.8694568160151103,
        "UAUC": 0.7892543491059268,
        "gAUC": 0.8009931646433784,
        "NDCG(10)": 0.4078900815748684
      },
      "role": "保留门控/激活结构负对照，AUC 未超过父模型"
    },
    "r14_silu_both": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.1894665515682686,
        "AUC": 0.8699905646110877,
        "UAUC": 0.7892554131981617,
        "gAUC": 0.800688777478991,
        "NDCG(10)": 0.4105455352634445
      },
      "role": "AUC 接近 Cross11，NDCG 高于父模型；保留纯激活对照"
    },
    "r15_w512_field": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18811097363891627,
        "AUC": 0.871053473618342,
        "UAUC": 0.7887096826330765,
        "gAUC": 0.8011888478372643,
        "NDCG(10)": 0.40850968280197725
      },
      "role": "本轮首选缩小候选：参数减少262656，AUC接近父模型，配对区间包含零；保留质量/成本取舍"
    },
    "r15_w512_history": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.1893389271794077,
        "AUC": 0.8695058058006561,
        "UAUC": 0.788255943604136,
        "gAUC": 0.8006527213639356,
        "NDCG(10)": 0.4062792474341235
      },
      "role": "保留完整宽度对照；AUC 未超过父模型"
    },
    "r15_w512_both": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18881680838002737,
        "AUC": 0.8704636769797732,
        "UAUC": 0.7907080725337244,
        "gAUC": 0.8025589305712282,
        "NDCG(10)": 0.4090798354063321
      },
      "role": "本轮最小模型，AUC高于Cross11，保留更小容量候选"
    },
    "r15_w680_field": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18892051604015586,
        "AUC": 0.8699750587032115,
        "UAUC": 0.7892989007427827,
        "gAUC": 0.8014458805172427,
        "NDCG(10)": 0.40898691657521763
      },
      "role": "AUC接近Cross11，保留中间字段宽度对照"
    },
    "r15_w680_history": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18922342941930084,
        "AUC": 0.8694319900433778,
        "UAUC": 0.7899754895747086,
        "gAUC": 0.8017257674266904,
        "NDCG(10)": 0.4082195650985395
      },
      "role": "保留完整宽度对照；AUC 未超过父模型"
    },
    "r15_w680_both": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.1890323946352457,
        "AUC": 0.8690157988645559,
        "UAUC": 0.789517646887768,
        "gAUC": 0.8017504941504203,
        "NDCG(10)": 0.406191537598606
      },
      "role": "保留完整宽度对照；AUC 未超过父模型"
    },
    "r15_w1536_history": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18832529545461402,
        "AUC": 0.8703771877590456,
        "UAUC": 0.7899211081115348,
        "gAUC": 0.8019346918356491,
        "NDCG(10)": 0.4076300629277791
      },
      "role": "最佳加宽对照；AUC低于字段缩减且参数更多"
    },
    "r15_w1536_both": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18866001971135787,
        "AUC": 0.8702826242590591,
        "UAUC": 0.7892138849718555,
        "gAUC": 0.8015619260117809,
        "NDCG(10)": 0.41003592696500707
      },
      "role": "NDCG略高父模型，保留容量/排序取舍"
    },
    "r16_base_alternate13": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.20097218743959347,
        "AUC": 0.8637244473279281,
        "UAUC": 0.7917133876233594,
        "gAUC": 0.8037468360512803,
        "NDCG(10)": 0.4086282682487054
      },
      "role": "保留可变深度负对照；固定两轮完整验证AUC低于对应父模型"
    },
    "r16_base_mixed1232": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18929151232504676,
        "AUC": 0.868667909411072,
        "UAUC": 0.7884124403363626,
        "gAUC": 0.8008448236622352,
        "NDCG(10)": 0.40267378515546376
      },
      "role": "原父模型可变深度中AUC最好，但低于固定2父模型和Cross8/Cross11"
    },
    "r16_base_curriculum13": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.19246249580109437,
        "AUC": 0.8648548849460762,
        "UAUC": 0.7954429211325335,
        "gAUC": 0.8058467151646105,
        "NDCG(10)": 0.42173087203170745
      },
      "role": "同seed完整验证NDCG最高的新排序候选，但AUC/Logloss明显变差；不替换AUC首选"
    },
    "r16_field512_alternate13": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.19141740435559754,
        "AUC": 0.8668230945392514,
        "UAUC": 0.7927064778416366,
        "gAUC": 0.8037829821740272,
        "NDCG(10)": 0.41143932410727785
      },
      "role": "保留可变深度负对照；固定两轮完整验证AUC低于对应父模型"
    },
    "r16_field512_mixed1232": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.19024235359683014,
        "AUC": 0.8669621998006798,
        "UAUC": 0.7893492905395305,
        "gAUC": 0.8016121503124702,
        "NDCG(10)": 0.4048700050898485
      },
      "role": "保留可变深度负对照；固定两轮完整验证AUC低于对应父模型"
    },
    "r16_field512_curriculum13": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.19079354726590925,
        "AUC": 0.8675540331212723,
        "UAUC": 0.7922428218148413,
        "gAUC": 0.8045735021122926,
        "NDCG(10)": 0.4094253581174222
      },
      "role": "小模型可变深度中AUC最好，UAUC略高但整体AUC明显下降"
    },
    "r17_base_phase132": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18919596072064104,
        "AUC": 0.8693122096729067,
        "UAUC": 0.7915434951656224,
        "gAUC": 0.8031111196610691,
        "NDCG(10)": 0.4125512943962935
      },
      "role": "原模型课程中AUC最高，接近Cross8，NDCG高于r7；未超过固定两轮父模型"
    },
    "r17_base_phase312": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18985858891050997,
        "AUC": 0.867415296238538,
        "UAUC": 0.789459965967518,
        "gAUC": 0.801595446556683,
        "NDCG(10)": 0.40706408636017666
      },
      "role": "保留阶段顺序对照；固定两轮AUC未超过对应父模型"
    },
    "r17_base_phase123": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.19455394429458747,
        "AUC": 0.8661672771531008,
        "UAUC": 0.7951428217590748,
        "gAUC": 0.8056752867278122,
        "NDCG(10)": 0.4181811540774068
      },
      "role": "本轮NDCG最高，但AUC明显下降；保留排序对照"
    },
    "r17_field512_phase132": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18853424734957908,
        "AUC": 0.8704386366506269,
        "UAUC": 0.7901645730728589,
        "gAUC": 0.8026829841808523,
        "NDCG(10)": 0.4100003176226837
      },
      "role": "本轮AUC最高；高于Cross8/Cross11，UAUC/NDCG高于字段512父模型，保留取舍候选"
    },
    "r17_field512_phase312": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18970431523174466,
        "AUC": 0.8684326152073885,
        "UAUC": 0.7880646981329759,
        "gAUC": 0.8007890426369534,
        "NDCG(10)": 0.4081175260046812
      },
      "role": "保留阶段顺序对照；固定两轮AUC未超过对应父模型"
    },
    "r17_field512_phase123": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18914920368994098,
        "AUC": 0.868986127063311,
        "UAUC": 0.7921526674256356,
        "gAUC": 0.8043222383961052,
        "NDCG(10)": 0.41163854327024363
      },
      "role": "保留阶段顺序对照；固定两轮AUC未超过对应父模型"
    },
    "r18_base_norm_phase132": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18866579149315915,
        "AUC": 0.8702125705973194,
        "UAUC": 0.7913000363925299,
        "gAUC": 0.803572725504752,
        "NDCG(10)": 0.41090968600201233
      },
      "role": "本轮AUC最佳；超过Cross8/Cross11且UAUC/NDCG略高于r7，保留取舍候选；未替换AUC首选"
    },
    "r18_base_norm_mixed1232": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.1895170639765351,
        "AUC": 0.8687300088263147,
        "UAUC": 0.7883841086660714,
        "gAUC": 0.8008722592714266,
        "NDCG(10)": 0.40211128094453363
      },
      "role": "相对共享norm AUC仅+0.000062，Logloss和排序未改善，保留近零对照"
    },
    "r18_field512_norm_phase132": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.18928623310239914,
        "AUC": 0.869173143298133,
        "UAUC": 0.7888230602266627,
        "gAUC": 0.8015915978797765,
        "NDCG(10)": 0.40576248260198594
      },
      "role": "保留负对照；未超过固定两轮父模型"
    },
    "r18_field512_norm_mixed1232": {
      "seed": 20262027,
      "metrics": {
        "logloss": 0.19020196299807227,
        "AUC": 0.8667476275273789,
        "UAUC": 0.7898832619494086,
        "gAUC": 0.8020390493886456,
        "NDCG(10)": 0.40457900275300274
      },
      "role": "保留负对照；未超过固定两轮父模型"
    }
  },
  "preferred_compact_candidate": {
    "family": "r7_small_attention_first",
    "seed": 20262027,
    "auc": 0.8712340584580865,
    "incumbent_promoted": false
  },
  "completed_round7": [
    {
      "hypothesis": "r7_terminal_prenorm",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r7_terminal_query_memory",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r7_terminal_field_memory",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r7_small_query_memory",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r7_small_field_memory",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r7_small_midnorm",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r7_small_attention_last",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r7_small_attention_first",
      "status": "completed",
      "seeds": [
        20262027
      ]
    }
  ],
  "highest_auc_candidate_current_fixed_seed": {
    "family": "r7_small_attention_first",
    "seed": 20262027,
    "auc": 0.8712340584580865,
    "incumbent_promoted": false
  },
  "completed_round8": [
    {
      "hypothesis": "r8_first_midnorm",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r8_first_half",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r8_first_double",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r8_first_prenorm",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r8_first_prenorm_half",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r8_first_midnorm_half",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r8_first_loop1",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r8_first_loop3",
      "status": "completed",
      "seeds": [
        20262027
      ]
    }
  ],
  "workspace_authorization": {
    "root": "external/home/lhr",
    "autonomous_continuation": true,
    "outside_project_folder_access": false
  },
  "completed_round9": [
    {
      "hypothesis": "r9_feedback_omit",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r9_history_replace",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r9_feedback_before",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r9_feedback_after",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r9_field_splitnorm",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r9_history_splitnorm",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r9_both_splitnorm",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r9_untied_clone",
      "status": "completed",
      "seeds": [
        20262027
      ]
    }
  ],
  "completed_round10": [
    {
      "hypothesis": "r10_field_cross",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r10_field_cross_ffn",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r10_field_self_cross",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r10_history_cross",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r10_history_cross_ffn",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r10_history_self_cross",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r10_both_cross_ffn",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r10_both_cross",
      "status": "completed",
      "seeds": [
        20262027
      ]
    }
  ],
  "completed_round11": [
    {
      "hypothesis": "r11_field_embedding",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r11_history_embedding",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r11_both_embedding",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r11_query_embedding",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r11_field_gate",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r11_history_gate",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r11_both_gate",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r11_embedding_gate",
      "status": "completed",
      "seeds": [
        20262027
      ]
    }
  ],
  "completed_round12": [
    {
      "hypothesis": "r12_field_caf",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r12_history_caf",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r12_both_caf",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r12_field_caf_all",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r12_history_caf_all",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r12_both_caf_all",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r12_field_afc",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r12_history_afc",
      "status": "completed",
      "seeds": [
        20262027
      ]
    }
  ],
  "completed_round13": [
    {
      "hypothesis": "r13_query_between_zero",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r13_query_between_warm",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r13_query_after_field_zero",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r13_query_after_field_warm",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r13_self_between_zero",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r13_self_between_warm",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r13_self_before_zero",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r13_self_before_warm",
      "status": "completed",
      "seeds": [
        20262027
      ]
    }
  ],
  "completed_round14": [
    {
      "hypothesis": "r14_swiglu_field",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r14_swiglu_history",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r14_swiglu_both",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r14_geglu_field",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r14_geglu_history",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r14_geglu_both",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r14_silu_history",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r14_silu_both",
      "status": "completed",
      "seeds": [
        20262027
      ]
    }
  ],
  "completed_round15": [
    {
      "hypothesis": "r15_w512_field",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r15_w512_history",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r15_w512_both",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r15_w680_field",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r15_w680_history",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r15_w680_both",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r15_w1536_history",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r15_w1536_both",
      "status": "completed",
      "seeds": [
        20262027
      ]
    }
  ],
  "smaller_near_auc_candidate": {
    "family": "r15_w512_field",
    "seed": 20262027,
    "auc": 0.871053473618342,
    "parameters": 7852817,
    "role": "fewer parameters; near parent AUC; lower UAUC/NDCG; no clear latency gain",
    "promoted": false
  },
  "completed_round16": [
    {
      "hypothesis": "r16_base_alternate13",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r16_base_mixed1232",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r16_base_curriculum13",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r16_field512_alternate13",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r16_field512_mixed1232",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r16_field512_curriculum13",
      "status": "completed",
      "seeds": [
        20262027
      ]
    }
  ],
  "highest_ndcg_candidate_current_fixed_seed": {
    "family": "r16_base_curriculum13",
    "seed": 20262027,
    "ndcg": 0.42173087203170745,
    "auc": 0.8648548849460762,
    "role": "ranking tradeoff only; substantially lower global AUC and worse logloss",
    "promoted_as_auc_candidate": false
  },
  "completed_round17": [
    {
      "hypothesis": "r17_base_phase132",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r17_base_phase312",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r17_base_phase123",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r17_field512_phase132",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r17_field512_phase312",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r17_field512_phase123",
      "status": "completed",
      "seeds": [
        20262027
      ]
    }
  ],
  "completed_round18": [
    {
      "hypothesis": "r18_base_norm_phase132",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r18_base_norm_mixed1232",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r18_field512_norm_phase132",
      "status": "completed",
      "seeds": [
        20262027
      ]
    },
    {
      "hypothesis": "r18_field512_norm_mixed1232",
      "status": "completed",
      "seeds": [
        20262027
      ]
    }
  ],
  "user_scope_updates": [
    {
      "request": "不一定一定要self-attention,或者qformer,可以尝试很多其他结构",
      "effect": "Architecture search unrestricted; no requirement to retain self-attention or QFormer"
    },
    {
      "request": "尝试把目前最好的迭代模型应用到淘宝任务",
      "effect": "Prioritize r7_small_attention_first architecture transfer to Taobao_Action"
    }
  ],
  "deferred_internal_norm_trial": {
    "round": 19,
    "training_started": false,
    "cpu_tests": 5,
    "reason": "User broadened structure search and prioritized Taobao transfer",
    "artifacts": [
      "model_zoo/QFormerCross11PassNorm.py",
      "tests/test_pass_norm.py"
    ]
  },
  "taobao_memory": "taobao-transfer/memory"
}

```
