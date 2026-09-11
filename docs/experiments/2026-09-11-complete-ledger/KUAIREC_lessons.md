# KuaiRec lessons.json 快照

此处保留历史判断，不把待执行计划当作已完成实验。

```json
{
  "schema_version": 1,
  "last_consolidated_episode": "diagnostics-round18-r18_field512_norm_phase132",
  "lessons": [
    {
      "lesson_id": "round2-query-route-mismatch",
      "statement": "在本 KuaiRec 协议和已训练的 Cross11/middle2 checkpoint 上，恢复训练时的全部 Query 改善三个 seed 的平均 AUC 和所有 seed 的 Logloss；单 seed AUC 不保证改善。",
      "scope": {
        "dataset": "kuairec-loop-v1",
        "component": "query_fusion_fixed_scales",
        "budget": "one epoch",
        "models": [
          "Cross11",
          "middle2"
        ]
      },
      "status": "supported",
      "confidence": 0.85,
      "supporting_episode_ids": [
        "routes-baseline-s20262027",
        "routes-baseline-s20262028",
        "routes-baseline-s20262029",
        "routes-middle2-s20262027",
        "routes-middle2-s20262028",
        "routes-middle2-s20262029"
      ],
      "counterevidence_episode_ids": [
        "routes-middle2-s20262029"
      ],
      "falsification_condition": "在相同标签/切分的独立 seed 或更长训练预算下，统一路由不再改善平均 AUC/Logloss。",
      "use_count": 1,
      "successful_use_count": 0,
      "note": "支持本轮统一路由对照，不证明循环独有收益或跨数据集收益。"
    },
    {
      "lesson_id": "round2-zero-gradient-routing",
      "statement": "被固定路由置零的 Query 的零梯度与零 Shuffle 效应不能作为删除模块的证据。",
      "scope": {
        "component": "query diagnostic",
        "route": "Q3 fixed zero"
      },
      "status": "supported",
      "confidence": 0.95,
      "supporting_episode_ids": [
        "diagnostics-round2-cross11",
        "diagnostics-round2-middle2-v2"
      ],
      "counterevidence_episode_ids": [],
      "falsification_condition": "如果在未屏蔽且存在有效监督的路由下仍无敏感性，应重新检验。",
      "use_count": 1,
      "successful_use_count": 1
    },
    {
      "lesson_id": "round3-field-feedback-candidate",
      "statement": "在固定 KuaiRec 一轮训练中，历史后字段回读的三 seed 平均 AUC 接近 Cross8 和原 middle2；相对原 middle2 少一次历史块调用，固定批量前向更快，可作为结构优化候选。",
      "scope": {
        "dataset": "kuairec-loop-v1",
        "model": "place_feedback2",
        "epochs": 1,
        "batch_size": 8192,
        "inference_batch": 1024
      },
      "status": "supported",
      "confidence": 0.72,
      "supporting_episode_ids": [
        "place_feedback2-s20262027-20260906T135252383734Z",
        "place_feedback2-s20262028-20260906T140336021412Z",
        "place_feedback2-s20262029-20260906T140336014811Z"
      ],
      "counterevidence_episode_ids": [
        "place_feedback2-s20262027-20260906T135252383734Z",
        "place_feedback2-s20262028-20260906T140336021412Z",
        "place_feedback2-s20262029-20260906T140336014811Z"
      ],
      "counterevidence_note": "AUC 均低于同 seed Cross8；相对 Cross11 全 Query 的条件用户 bootstrap 区间为负。支持候选保留，不支持胜出。",
      "falsification_condition": "扩展 seed/预算后明显低于基线，或端到端测量不保留成本优势。",
      "use_count": 1,
      "successful_use_count": 0
    },
    {
      "lesson_id": "round3-field-recurrence-variance",
      "statement": "字段末层循环在本次三个 seed 中 AUC 波动大，不能只看 seed=20262028 的较高结果。",
      "scope": {
        "model": "place_field2",
        "dataset": "kuairec-loop-v1"
      },
      "status": "provisional",
      "confidence": 0.65,
      "supporting_episode_ids": [
        "place_field2-s20262027-20260906T135252567700Z",
        "place_field2-s20262028-20260906T140336061987Z",
        "place_field2-s20262029-20260906T140336018898Z"
      ],
      "counterevidence_episode_ids": [],
      "falsification_condition": "增加 seed 或控制用户/目标接口后波动消失。",
      "use_count": 1,
      "successful_use_count": 0
    },
    {
      "lesson_id": "round4-context-feedback",
      "statement": "同协议三 seed 中，历史后只回读用户/上下文较全部字段反馈改善平均 AUC 和 Logloss；初始字段编码仍读取全部特征。",
      "scope": {
        "dataset": "kuairec-loop-v1",
        "epochs": 1,
        "batch_size": 8192,
        "seeds": [
          20262027,
          20262028,
          20262029
        ]
      },
      "status": "provisional",
      "confidence": 0.7,
      "supporting_episode_ids": [
        "feedback_context-s20262027-20260906T143634603091Z",
        "feedback_context-s20262028-20260906T143634525420Z",
        "feedback_context-s20262029-20260906T143634581333Z"
      ],
      "counterevidence_episode_ids": [
        "feedback_context-s20262027-20260906T143634603091Z",
        "feedback_context-s20262028-20260906T143634525420Z",
        "feedback_context-s20262029-20260906T143634581333Z"
      ],
      "counterevidence_note": "seed 29 AUC 仅提高 0.0000035；平均 NDCG 降低且 seed 波动增加；相对优化基线的条件区间跨零。",
      "falsification_condition": "独立 seed、固定计算预算或更长训练预算下平均 AUC 收益消失。",
      "use_count": 1,
      "successful_use_count": 0
    },
    {
      "lesson_id": "round5-single-history-recurrence",
      "statement": "本协议一轮训练下，保留两字段块和一个共享历史块的 compact_seq1 在三个 seed 均超过 Cross8 AUC，参数略少且固定 batch 延迟接近。",
      "scope": {
        "dataset": "kuairec-loop-v1",
        "epochs": 1,
        "batch_size": 8192,
        "inference_batch": 1024,
        "seeds": [
          20262027,
          20262028,
          20262029
        ]
      },
      "status": "supported",
      "confidence": 0.8,
      "supporting_episode_ids": [
        "compact_seq1-s20262027-20260906T150445272533Z",
        "compact_seq1-s20262028-20260906T150445070621Z",
        "compact_seq1-s20262029-20260906T150444924864Z"
      ],
      "counterevidence_episode_ids": [
        "compact_seq1-s20262027-20260906T150445272533Z",
        "compact_seq1-s20262028-20260906T150445070621Z",
        "compact_seq1-s20262029-20260906T150444924864Z"
      ],
      "counterevidence_note": "相对 Cross11/父模型不是每 seed 均改善；UAUC 略低于父模型。未训练浅层无循环对照，尚不能分离减层与循环收益。",
      "falsification_condition": "独立 seed、更长预算或固定 GPU 时间比较时收益消失。",
      "use_count": 0,
      "successful_use_count": 0
    },
    {
      "lesson_id": "round5-extra-attention-depth-interaction",
      "statement": "本轮额外共享 Query self-attention 在三历史块和两历史块结构提高平均 AUC，在单历史块结构降低平均 AUC，收益依赖保留的结构。",
      "scope": {
        "dataset": "kuairec-loop-v1",
        "epochs": 1,
        "placement": "before each recurrent field read"
      },
      "status": "provisional",
      "confidence": 0.7,
      "supporting_episode_ids": [
        "attn_full-s20262027-20260906T150445276311Z",
        "attn_full-s20262028-20260906T150445272581Z",
        "attn_full-s20262029-20260906T151301200510Z",
        "attn_seq2-s20262027-20260906T151301214301Z",
        "attn_seq2-s20262028-20260906T151301220984Z",
        "attn_seq2-s20262029-20260906T151412118962Z",
        "attn_seq1-s20262027-20260906T151417256474Z",
        "attn_seq1-s20262028-20260906T151417316199Z",
        "attn_seq1-s20262029-20260906T151447655718Z"
      ],
      "counterevidence_episode_ids": [
        "attn_seq1-s20262027-20260906T151417256474Z",
        "attn_seq1-s20262028-20260906T151417316199Z",
        "attn_seq1-s20262029-20260906T151447655718Z"
      ],
      "falsification_condition": "独立 seed 或更多深度/位置对照不再出现这种交互差异。",
      "use_count": 0,
      "successful_use_count": 0
    },
    {
      "lesson_id": "round6-terminal-context-feedback",
      "statement": "固定 seed 20262027 下，紧凑历史循环完成后回读上下文较中途回读提高 AUC，但 Logloss/UAUC/NDCG 略降；两种位置均应保留指标取舍。",
      "scope": {
        "dataset": "kuairec-loop-v1",
        "epochs": 1,
        "seed": 20262027,
        "model": "QFormerCross11Schedule"
      },
      "status": "provisional",
      "confidence": 0.6,
      "supporting_episode_ids": [
        "r6_field_after-s20262027-20260906T154408812865Z"
      ],
      "counterevidence_episode_ids": [
        "r6_field_after-s20262027-20260906T154408812865Z",
        "r6_small_after-s20262027-20260906T160133981082Z"
      ],
      "counterevidence_note": "收益限单 seed，且与去掉字段前置块组合后反而退化；不能外推位置独立有效。",
      "falsification_condition": "固定 seed 的结构控制不再保留收益，或更改预算后收益消失。",
      "use_count": 0,
      "successful_use_count": 0
    },
    {
      "lesson_id": "round6-compact-post-history-attention",
      "statement": "固定 seed 下，1 字段/1 历史块循环加历史后共享 attention 可在 AUC 接近父模型时减少参数/延迟，并改善 Logloss。",
      "scope": {
        "dataset": "kuairec-loop-v1",
        "epochs": 1,
        "seed": 20262027,
        "inference_batch": 1024
      },
      "status": "provisional",
      "confidence": 0.65,
      "supporting_episode_ids": [
        "r6_small_attention_after-s20262027-20260906T160133918683Z"
      ],
      "counterevidence_episode_ids": [
        "r6_small_attention_after-s20262027-20260906T160133918683Z"
      ],
      "counterevidence_note": "相对父模型 AUC 略低、UAUC 下降；只验证已有一个 seed。",
      "falsification_condition": "固定预算的进一步结构控制下质量或实际成本优势消失。",
      "use_count": 0,
      "successful_use_count": 0
    },
    {
      "lesson_id": "round6-sharing-and-cycle-controls",
      "statement": "同一个已有 seed 中，去掉额外循环或改为独立历史权重均未超过父模型；从头训练三次循环也低于两次。",
      "scope": {
        "dataset": "kuairec-loop-v1",
        "epochs": 1,
        "seed": 20262027
      },
      "status": "provisional",
      "confidence": 0.6,
      "supporting_episode_ids": [
        "r6_shallow-s20262027-20260906T154408839146Z",
        "r6_untied-s20262027-20260906T154409087381Z",
        "r6_loop3-s20262027-20260906T154408977323Z"
      ],
      "counterevidence_episode_ids": [
        "r6_shallow-s20262027-20260906T154408839146Z"
      ],
      "counterevidence_note": "浅层方案仍以明显更低前向成本接近 Cross8，不能仅因 AUC 较低删除。",
      "falsification_condition": "后续固定 seed 的独立结构或预算对照改变这些关系。",
      "use_count": 0,
      "successful_use_count": 0
    },
    {
      "lesson_id": "round7-first-pass-attention",
      "statement": "固定已有 seed 下，1 字段/1 历史块两次循环只保留第一次额外 attention，较两次 attention 提高 AUC/Logloss 并减少前向耗时；只保留最后一次则降低 AUC。",
      "scope": {
        "dataset": "kuairec-loop-v1",
        "epochs": 1,
        "seed": 20262027
      },
      "status": "provisional",
      "confidence": 0.6,
      "supporting_episode_ids": [
        "r7_small_attention_first-s20262027-20260906T163655782605Z",
        "r7_small_attention_last-s20262027-20260906T163655854191Z"
      ],
      "counterevidence_episode_ids": [
        "r7_small_attention_first-s20262027-20260906T163655782605Z",
        "r7_small_attention_last-s20262027-20260906T163655854191Z"
      ],
      "counterevidence_note": "最佳方案 NDCG 较父模型低 0.000180；只完成单 seed 固定预算结构筛选。",
      "falsification_condition": "固定 seed 的进一步结构控制或训练预算变化改变该关系。",
      "use_count": 0,
      "successful_use_count": 0
    },
    {
      "lesson_id": "round7-norm-location",
      "statement": "每次历史读取后、额外 attention 前应用序列 norm 改善小模型 AUC/Logloss；将 norm 移到末尾字段反馈之前未提高 terminal 父模型 AUC。",
      "scope": {
        "dataset": "kuairec-loop-v1",
        "epochs": 1,
        "seed": 20262027
      },
      "status": "provisional",
      "confidence": 0.6,
      "supporting_episode_ids": [
        "r7_small_midnorm-s20262027-20260906T163655828274Z",
        "r7_terminal_prenorm-s20262027-20260906T163655868809Z"
      ],
      "counterevidence_episode_ids": [
        "r7_small_midnorm-s20262027-20260906T163655828274Z",
        "r7_terminal_prenorm-s20262027-20260906T163655868809Z"
      ],
      "counterevidence_note": "小模型 UAUC/NDCG 下降，不能将 norm 移位视为跨位置通用收益。",
      "falsification_condition": "固定 seed 的进一步结构控制或训练预算变化改变该关系。",
      "use_count": 0,
      "successful_use_count": 0
    },
    {
      "lesson_id": "round7-memory-kv",
      "statement": "在本轮固定 seed 的四个对照中，额外字段 K/V 拼接初始 Query 或首次字段编码均未提高直接父模型 AUC，且增加少量前向计算。",
      "scope": {
        "dataset": "kuairec-loop-v1",
        "epochs": 1,
        "seed": 20262027
      },
      "status": "provisional",
      "confidence": 0.6,
      "supporting_episode_ids": [
        "r7_terminal_query_memory-s20262027-20260906T163655724136Z",
        "r7_terminal_field_memory-s20262027-20260906T163655831635Z",
        "r7_small_query_memory-s20262027-20260906T163655785402Z",
        "r7_small_field_memory-s20262027-20260906T163655842646Z"
      ],
      "counterevidence_episode_ids": [
        "r7_terminal_query_memory-s20262027-20260906T163655724136Z",
        "r7_terminal_field_memory-s20262027-20260906T163655831635Z",
        "r7_small_query_memory-s20262027-20260906T163655785402Z",
        "r7_small_field_memory-s20262027-20260906T163655842646Z"
      ],
      "counterevidence_note": "terminal 初始 Query 记忆改善 Logloss/NDCG，应保留指标取舍；本结果不否定其他记忆融合方式。",
      "falsification_condition": "固定 seed 的进一步结构控制或训练预算变化改变该关系。",
      "use_count": 0,
      "successful_use_count": 0
    },
    {
      "lesson_id": "round8-trained-cycle-depth",
      "statement": "同参数、同 seed、同更新预算从头训练时，当前首次 attention 小模型两次共享循环 AUC 高于一次和三次；增加到三次退化。",
      "scope": {
        "dataset": "kuairec-loop-v1",
        "epochs": 1,
        "seed": 20262027,
        "parent": "r7_small_attention_first"
      },
      "status": "provisional",
      "confidence": 0.6,
      "supporting_episode_ids": [
        "r8_first_loop1-s20262027-20260906T170409366470Z",
        "r8_first_loop3-s20262027-20260906T170444520001Z"
      ],
      "counterevidence_episode_ids": [
        "r8_first_loop1-s20262027-20260906T170409366470Z",
        "r8_first_loop3-s20262027-20260906T170444520001Z"
      ],
      "counterevidence_note": "一次循环 UAUC/NDCG 更高且前向更快；对照同时改变计算量与上下文回读，不能单独证明权重共享本身的收益。",
      "falsification_condition": "固定 seed 下更改回读位置、独立权重或训练预算后关系改变。",
      "use_count": 0,
      "successful_use_count": 0
    },
    {
      "lesson_id": "round8-norm-scale-interaction",
      "statement": "仅首次 attention 的父模型上，post-norm 残差减半在原接口和每轮序列 norm 两种结构均降低 AUC；pre-norm 后减半优于全强度 pre-norm，但仍未超过父模型。",
      "scope": {
        "dataset": "kuairec-loop-v1",
        "epochs": 1,
        "seed": 20262027,
        "parent": "r7_small_attention_first"
      },
      "status": "provisional",
      "confidence": 0.6,
      "supporting_episode_ids": [
        "r8_first_half-s20262027-20260906T170409452426Z",
        "r8_first_midnorm_half-s20262027-20260906T170409455169Z",
        "r8_first_prenorm-s20262027-20260906T170409169559Z",
        "r8_first_prenorm_half-s20262027-20260906T170409454737Z"
      ],
      "counterevidence_episode_ids": [
        "r8_first_half-s20262027-20260906T170409452426Z",
        "r8_first_midnorm_half-s20262027-20260906T170409455169Z",
        "r8_first_prenorm-s20262027-20260906T170409169559Z",
        "r8_first_prenorm_half-s20262027-20260906T170409454737Z"
      ],
      "counterevidence_note": "收益依赖归一化位置；不能从单一缩放实验推断所有残差强度控制无效。",
      "falsification_condition": "固定 seed 下更改回读位置、独立权重或训练预算后关系改变。",
      "use_count": 0,
      "successful_use_count": 0
    },
    {
      "lesson_id": "round8-ranking-cost-frontier",
      "statement": "一次 N S A 前向在本固定 seed 达到接近 Cross8/Cross11 的 AUC，UAUC/NDCG 高于两次循环父模型，前向耗时减少约三分之一，可作为低成本排序候选。",
      "scope": {
        "dataset": "kuairec-loop-v1",
        "epochs": 1,
        "seed": 20262027,
        "parent": "r7_small_attention_first"
      },
      "status": "provisional",
      "confidence": 0.6,
      "supporting_episode_ids": [
        "r8_first_loop1-s20262027-20260906T170409366470Z"
      ],
      "counterevidence_episode_ids": [
        "r8_first_loop1-s20262027-20260906T170409366470Z"
      ],
      "counterevidence_note": "AUC 比两次循环低约 0.001138，Logloss 更差；只有固定 batch 前向成本，未验证端到端部署。",
      "falsification_condition": "固定 seed 下更改回读位置、独立权重或训练预算后关系改变。",
      "use_count": 0,
      "successful_use_count": 0
    },
    {
      "lesson_id": "round9-context-feedback-position",
      "statement": "当前首次 attention 小模型中，去掉第二次上下文回读、替换为历史回读，或将它提前/后移，均未超过原交替结构完整训练后的 AUC。",
      "scope": {
        "dataset": "kuairec-loop-v1",
        "epochs": 1,
        "seed": 20262027,
        "parent": "r7_small_attention_first"
      },
      "status": "provisional",
      "confidence": 0.6,
      "supporting_episode_ids": [
        "r9_feedback_omit-s20262027-20260906T173725470939Z",
        "r9_history_replace-s20262027-20260906T173725574888Z",
        "r9_feedback_before-s20262027-20260906T173725569782Z",
        "r9_feedback_after-s20262027-20260906T173725584656Z"
      ],
      "counterevidence_episode_ids": [
        "r9_feedback_omit-s20262027-20260906T173725470939Z",
        "r9_history_replace-s20262027-20260906T173725574888Z",
        "r9_feedback_before-s20262027-20260906T173725569782Z",
        "r9_feedback_after-s20262027-20260906T173725584656Z"
      ],
      "counterevidence_note": "去掉回读减少计算，但也改变读入信息；同 block 数不等于同 FLOPs。此前其他结构曾受益于末尾回读，不能跨结构泛化。",
      "falsification_condition": "固定 seed 下修改回读子层或预算后关系改变。",
      "use_count": 0,
      "successful_use_count": 0
    },
    {
      "lesson_id": "round9-function-matched-history-sharing",
      "statement": "从相同初始历史块复制并独立训练第二次读取，AUC 比共享版低 0.000461，且增加 1,185,536 参数；本固定预算下继续保留共享。",
      "scope": {
        "dataset": "kuairec-loop-v1",
        "epochs": 1,
        "seed": 20262027,
        "parent": "r7_small_attention_first"
      },
      "status": "provisional",
      "confidence": 0.6,
      "supporting_episode_ids": [
        "r9_untied_clone-s20262027-20260906T174430682740Z"
      ],
      "counterevidence_episode_ids": [
        "r9_untied_clone-s20262027-20260906T174430682740Z"
      ],
      "counterevidence_note": "只是一个已有 seed、一个初始函数的独立权重对照，不证明共享在所有预算和模型下更好。",
      "falsification_condition": "固定 seed 下修改回读子层或预算后关系改变。",
      "use_count": 0,
      "successful_use_count": 0
    },
    {
      "lesson_id": "round9-interface-sharing",
      "statement": "两次字段 norm 独立未超过共享父模型；历史/双接口分轮 norm 均未超过归一化位置匹配的 r8_first_midnorm AUC。",
      "scope": {
        "dataset": "kuairec-loop-v1",
        "epochs": 1,
        "seed": 20262027,
        "parent": "r7_small_attention_first"
      },
      "status": "provisional",
      "confidence": 0.6,
      "supporting_episode_ids": [
        "r9_field_splitnorm-s20262027-20260906T173725574982Z",
        "r9_history_splitnorm-s20262027-20260906T173725520543Z",
        "r9_both_splitnorm-s20262027-20260906T173725417308Z"
      ],
      "counterevidence_episode_ids": [
        "r9_field_splitnorm-s20262027-20260906T173725574982Z",
        "r9_history_splitnorm-s20262027-20260906T173725520543Z",
        "r9_both_splitnorm-s20262027-20260906T173725417308Z"
      ],
      "counterevidence_note": "双接口独立稍微改善匹配父模型 Logloss，但用户排序指标退化；保留条件取舍。",
      "falsification_condition": "固定 seed 下修改回读子层或预算后关系改变。",
      "use_count": 0,
      "successful_use_count": 0
    },
    {
      "lesson_id": "round10-selective-repeated-sublayers",
      "statement": "首次字段/历史读取保持完整、第二次精简 self-attention 或 FFN 的八种结构，均未超过完整共享 block 父模型 AUC。",
      "scope": {
        "dataset": "kuairec-loop-v1",
        "epochs": 1,
        "seed": 20262027,
        "parent": "r7_small_attention_first"
      },
      "status": "provisional",
      "confidence": 0.6,
      "supporting_episode_ids": [
        "r10_field_cross-s20262027-20260906T181435551595Z",
        "r10_field_cross_ffn-s20262027-20260906T181435546200Z",
        "r10_field_self_cross-s20262027-20260906T181435546256Z",
        "r10_history_cross-s20262027-20260906T181435546214Z",
        "r10_history_cross_ffn-s20262027-20260906T181435553962Z",
        "r10_history_self_cross-s20262027-20260906T181435552759Z",
        "r10_both_cross_ffn-s20262027-20260906T181435276632Z",
        "r10_both_cross-s20262027-20260906T181435276810Z"
      ],
      "counterevidence_episode_ids": [
        "r10_field_cross-s20262027-20260906T181435551595Z",
        "r10_field_cross_ffn-s20262027-20260906T181435546200Z",
        "r10_field_self_cross-s20262027-20260906T181435546256Z",
        "r10_history_cross-s20262027-20260906T181435546214Z",
        "r10_history_cross_ffn-s20262027-20260906T181435553962Z",
        "r10_history_self_cross-s20262027-20260906T181435552759Z",
        "r10_both_cross_ffn-s20262027-20260906T181435276632Z",
        "r10_both_cross-s20262027-20260906T181435276810Z"
      ],
      "counterevidence_note": "每次删除也移除了对应 residual LayerNorm，不能将下降单独归因于 attention 或 FFN；字段去 FFN 的 UAUC 略高。",
      "falsification_condition": "保留被跳过的 norm 或引入条件化的共享更新后，同预算关系改变。",
      "use_count": 0,
      "successful_use_count": 0
    },
    {
      "lesson_id": "round10-selective-cost-quality",
      "statement": "选择性执行重复子层可降低固定 batch 前向耗时，历史 CrossValue+FFN 方案 AUC 接近 Cross11，保留为质量/成本候选。",
      "scope": {
        "dataset": "kuairec-loop-v1",
        "epochs": 1,
        "seed": 20262027,
        "parent": "r7_small_attention_first"
      },
      "status": "provisional",
      "confidence": 0.6,
      "supporting_episode_ids": [
        "r10_history_cross_ffn-s20262027-20260906T181435553962Z",
        "r10_field_self_cross-s20262027-20260906T181435546256Z",
        "r10_both_cross-s20262027-20260906T181435276810Z"
      ],
      "counterevidence_episode_ids": [
        "r10_history_cross_ffn-s20262027-20260906T181435553962Z",
        "r10_field_self_cross-s20262027-20260906T181435546256Z",
        "r10_both_cross-s20262027-20260906T181435276810Z"
      ],
      "counterevidence_note": "本轮仍未超过当前 AUC 首选；最快精简模型的 AUC/UAUC/NDCG 和速度均不如已有一次循环候选。",
      "falsification_condition": "保留被跳过的 norm 或引入条件化的共享更新后，同预算关系改变。",
      "use_count": 0,
      "successful_use_count": 0
    },
    {
      "lesson_id": "round11-neutral-repeat-conditioning",
      "statement": "八个初始函数兼容的轮次向量/Query门控变体未超过父模型；历史向量接近父模型但无主要指标优势。",
      "scope": {
        "dataset": "kuairec-loop-v1",
        "epochs": 1,
        "seed": 20262027,
        "parent": "r7_small_attention_first"
      },
      "status": "provisional",
      "confidence": 0.6,
      "supporting_episode_ids": [
        "r11_field_embedding-s20262027-20260907T032608534452Z",
        "r11_history_embedding-s20262027-20260907T032608681208Z",
        "r11_both_embedding-s20262027-20260907T032608534442Z",
        "r11_query_embedding-s20262027-20260907T032608496547Z",
        "r11_field_gate-s20262027-20260907T032608540320Z",
        "r11_history_gate-s20262027-20260907T032608594282Z",
        "r11_both_gate-s20262027-20260907T032608561082Z",
        "r11_embedding_gate-s20262027-20260907T032659791882Z"
      ],
      "counterevidence_episode_ids": [
        "r11_history_embedding-s20262027-20260907T032608681208Z"
      ],
      "counterevidence_note": "历史向量与父模型 AUC 差仅 0.00001684；不能将单 seed 的排序推广为普遍无效。",
      "falsification_condition": "改变 block 内信息流、条件作用点或训练预算后关系改变。",
      "use_count": 0,
      "successful_use_count": 0
    },
    {
      "lesson_id": "round12-full-sublayer-order",
      "statement": "完整循环 block 的八种子层重排均未超过标准 Self/Cross/FFN 父模型；两轮均 CAF 的 AUC 高于仅重复 CAF，但不能推出全部 guardrail 更优。",
      "scope": {
        "dataset": "kuairec-loop-v1",
        "epochs": 1,
        "seed": 20262027,
        "parent": "r7_small_attention_first"
      },
      "status": "provisional",
      "confidence": 0.6,
      "supporting_episode_ids": [
        "r12_field_caf-s20262027-20260907T035559099466Z",
        "r12_history_caf-s20262027-20260907T035559393353Z",
        "r12_both_caf-s20262027-20260907T035559396151Z",
        "r12_field_caf_all-s20262027-20260907T035559123398Z",
        "r12_history_caf_all-s20262027-20260907T035559379528Z",
        "r12_both_caf_all-s20262027-20260907T035559385613Z",
        "r12_field_afc-s20262027-20260907T035559388666Z",
        "r12_history_afc-s20262027-20260907T035559125014Z"
      ],
      "counterevidence_episode_ids": [
        "r12_both_caf-s20262027-20260907T035559396151Z"
      ],
      "counterevidence_note": "双位置只改重复调用的 Logloss/排序优于两轮都改；历史 AFC 仍超过 Cross8/Cross11。",
      "falsification_condition": "改变历史源编码、循环更新接口或预算后关系改变。",
      "use_count": 0,
      "successful_use_count": 0
    },
    {
      "lesson_id": "round13-history-source-update",
      "statement": "Query反馈/历史self-attention的八种历史更新结构未刷新AUC；两轮之间零初始Query反馈有UAUC取舍，历史self-attention warm有NDCG取舍。",
      "scope": {
        "dataset": "kuairec-loop-v1",
        "epochs": 1,
        "seed": 20262027,
        "parent": "r7_small_attention_first"
      },
      "status": "provisional",
      "confidence": 0.6,
      "supporting_episode_ids": [
        "r13_query_between_zero-s20262027-20260907T042433943622Z",
        "r13_query_between_warm-s20262027-20260907T042433862402Z",
        "r13_query_after_field_zero-s20262027-20260907T042433963604Z",
        "r13_query_after_field_warm-s20262027-20260907T042433869217Z",
        "r13_self_between_zero-s20262027-20260907T042433943647Z",
        "r13_self_between_warm-s20262027-20260907T042433953319Z",
        "r13_self_before_zero-s20262027-20260907T042433845232Z",
        "r13_self_before_warm-s20262027-20260907T043325955638Z"
      ],
      "counterevidence_episode_ids": [
        "r13_query_between_zero-s20262027-20260907T042433943622Z",
        "r13_self_between_warm-s20262027-20260907T042433953319Z"
      ],
      "counterevidence_note": "UAUC或NDCG局部改善；不可将单seed阴性AUC结果推广为历史反馈普遍无效。",
      "falsification_condition": "改变共享FFN、历史更新方式或训练预算后关系改变。",
      "use_count": 0,
      "successful_use_count": 0
    },
    {
      "lesson_id": "round14-gated-activation-budget",
      "statement": "同初始化及宽度的门控对照中，SwiGLU AUC在三个位置均高于GEGLU；八组门控/SiLU对照未刷新r7，双位置SwiGLU有UAUC/NDCG取舍。",
      "scope": {
        "dataset": "kuairec-loop-v1",
        "epochs": 1,
        "seed": 20262027,
        "parent": "r7_small_attention_first"
      },
      "status": "provisional",
      "confidence": 0.6,
      "supporting_episode_ids": [
        "r14_swiglu_field-s20262027-20260907T045344463996Z",
        "r14_swiglu_history-s20262027-20260907T045344433620Z",
        "r14_swiglu_both-s20262027-20260907T045344417858Z",
        "r14_geglu_field-s20262027-20260907T045344405903Z",
        "r14_geglu_history-s20262027-20260907T045344436148Z",
        "r14_geglu_both-s20262027-20260907T045344165025Z",
        "r14_silu_history-s20262027-20260907T045344217569Z",
        "r14_silu_both-s20262027-20260907T050135045692Z"
      ],
      "counterevidence_episode_ids": [
        "r14_geglu_both-s20262027-20260907T045344165025Z"
      ],
      "counterevidence_note": "双位置GEGLU NDCG略高于SwiGLU；激活AUC排序不能推广为全指标或多seed稳定收益。",
      "falsification_condition": "只改普通FFN宽度、初始化或训练预算后关系改变。",
      "use_count": 0,
      "successful_use_count": 0
    },
    {
      "lesson_id": "round15-plain-width-capacity",
      "statement": "字段GELU FFN512以更少参数达到接近r7的AUC；同预算加宽未超过它，宽度与AUC关系不单调。",
      "scope": {
        "dataset": "kuairec-loop-v1",
        "epochs": 1,
        "seed": 20262027,
        "parent": "r7_small_attention_first"
      },
      "status": "provisional",
      "confidence": 0.6,
      "supporting_episode_ids": [
        "r15_w512_field-s20262027-20260907T052153858349Z",
        "r15_w512_history-s20262027-20260907T052153719493Z",
        "r15_w512_both-s20262027-20260907T052153886924Z",
        "r15_w680_field-s20262027-20260907T052153748368Z",
        "r15_w680_history-s20262027-20260907T052153885943Z",
        "r15_w680_both-s20262027-20260907T052153679180Z",
        "r15_w1536_history-s20262027-20260907T052153898429Z",
        "r15_w1536_both-s20262027-20260907T052939555012Z"
      ],
      "counterevidence_episode_ids": [
        "r15_w512_field-s20262027-20260907T052153858349Z",
        "r15_w1536_both-s20262027-20260907T052939555012Z"
      ],
      "counterevidence_note": "字段缩减UAUC/NDCG低于r7，加宽双位置NDCG略高；配对AUC区间包含零不证明等效或多seed稳定。",
      "falsification_condition": "可变训练深度、初始化方式或训练预算变化后关系改变。",
      "use_count": 0,
      "successful_use_count": 0
    },
    {
      "lesson_id": "round16-variable-depth-ranking-calibration",
      "statement": "可变训练深度固定两轮验证未提升AUC；先1后3课程创造较高用户排序指标但全局AUC/Logloss恶化，后续需检验结束深度与验证深度对齐。",
      "scope": {
        "dataset": "kuairec-loop-v1",
        "epochs": 1,
        "seed": 20262027,
        "evaluation_depth": 2
      },
      "status": "provisional",
      "confidence": 0.6,
      "supporting_episode_ids": [
        "r16_base_alternate13-s20262027-20260907T071543287067Z",
        "r16_base_mixed1232-s20262027-20260907T071543487684Z",
        "r16_base_curriculum13-s20262027-20260907T071543460834Z",
        "r16_field512_alternate13-s20262027-20260907T071543464488Z",
        "r16_field512_mixed1232-s20262027-20260907T071543357571Z",
        "r16_field512_curriculum13-s20262027-20260907T071543495742Z"
      ],
      "counterevidence_episode_ids": [
        "r16_base_curriculum13-s20262027-20260907T071543460834Z"
      ],
      "counterevidence_note": "该候选全局AUC较差，子集改为推理3轮仅为机制线索；原始seq_len不是截断后的计算量，已只读重放纠正。",
      "falsification_condition": "课程最终在两轮训练结束后，AUC和排序的关系改变。",
      "use_count": 0,
      "successful_use_count": 0
    },
    {
      "lesson_id": "round17-phase-order-ending-depth",
      "statement": "相同深度更新次数下，phase132在两类父模型上均比phase123/312的AUC高，但都未超过固定两轮父模型；阶段顺序不能简化为仅结束深度效应。",
      "scope": {
        "dataset": "kuairec-loop-v1",
        "epochs": 1,
        "seed": 20262027,
        "evaluation_depth": 2
      },
      "status": "provisional",
      "confidence": 0.6,
      "supporting_episode_ids": [
        "r17_base_phase132-s20262027-20260907T073714696072Z",
        "r17_base_phase312-s20262027-20260907T073714772329Z",
        "r17_base_phase123-s20262027-20260907T073714793487Z",
        "r17_field512_phase132-s20262027-20260907T073714767736Z",
        "r17_field512_phase312-s20262027-20260907T073714677736Z",
        "r17_field512_phase123-s20262027-20260907T073714793481Z"
      ],
      "counterevidence_episode_ids": [
        "r17_base_phase123-s20262027-20260907T073714793487Z"
      ],
      "counterevidence_note": "phase123具有更高NDCG/UAUC而全局AUC较低，指标取舍仍存在。",
      "falsification_condition": "按执行深度区分最终归一化后，同预算关系改变。",
      "use_count": 0,
      "successful_use_count": 0
    },
    {
      "lesson_id": "round18-depth-final-norm-not-general",
      "statement": "按执行深度独立最终LayerNorm，仅原容量phase132出现较明显AUC恢复，field512两种安排均下降，四组均低于固定两轮父模型。",
      "scope": {
        "dataset": "kuairec-loop-v1",
        "epochs": 1,
        "seed": 20262027,
        "evaluation_depth": 2
      },
      "status": "provisional",
      "confidence": 0.6,
      "supporting_episode_ids": [
        "r18_base_norm_phase132-s20262027-20260907T080802807230Z",
        "r18_base_norm_mixed1232-s20262027-20260907T080802773837Z",
        "r18_field512_norm_phase132-s20262027-20260907T080802844563Z",
        "r18_field512_norm_mixed1232-s20262027-20260907T080802738804Z"
      ],
      "counterevidence_episode_ids": [
        "r18_base_norm_phase132-s20262027-20260907T080802807230Z"
      ],
      "counterevidence_note": "原容量phase132较完全匹配安排共享norm AUC+0.000900，但低于r7；不同容量方向不一致。",
      "falsification_condition": "固定两轮内部轮次norm或预先限定调参改变当前关系；不能外推当前单seed结果。",
      "use_count": 0,
      "successful_use_count": 0
    }
  ]
}

```
