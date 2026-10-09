# KuaiRec：阶段10–18及补充记录

快照时间：2026-09-09T07:36:23.974768+00:00。返回[总览](README.md)。

[阶段1–9](KUAIREC_EXPERIMENTS.md) · [阶段10–18、跨实验归因与工程补充](KUAIREC_ROUNDS10_18.md)

<a id="phase-10"></a>
## 阶段 10：第二轮子层选择

**方法**：第二次字段/历史读取只保留Cross、Cross+FFN或Self+Cross。

**归因与边界**：8组均未超过完整父模型；部分方案省计算。删除子层同时删除其norm，不能独立识别算子作用。

**历史后续假设**：保留完整block，以中性初始化的轮次向量/门控条件化；见K11。

<a id="k138"></a>
### K138 · `r10_field_cross-s20262027-20260906T181435551595Z`

- **方法**：{"field_repeat_mode":"cross","history_repeat_mode":"full"}；设置 {"model":"QFormerCross11Selective","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Select shared sublayers only in repeated field/history block calls
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.189349771；AUC=0.868951786；UAUC=0.790091648；gAUC=0.802247632；NDCG(10)=0.408171304。
- **开销**：参数 8115473；训练含验证 421.3702529668808 秒；峰值CUDA 15918333440 bytes；更新 1240。
- **原记录差值**：-0.002282273；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.0012829124589598395,"AUC":-0.002282272853686429,"UAUC":-0.0007453647004496133,"gAUC":-0.0015430304382826332,"NDCG(10)":-0.0016848532322336074}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.069200038909912,"mean_ms":12.086395263671875,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.002282273，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 10](#phase-10)。
- **后续假设**：保留完整block，以中性初始化的轮次向量/门控条件化；见K11。
- **原判读**：deprioritized field-cross-only control
- **证据**：`qformer-loop/runs/r10_field_cross-s20262027-20260906T181435551595Z/manifest.json`（SHA256前缀 `7bce00111e37`）；`qformer-loop/runs/r10_field_cross-s20262027-20260906T181435551595Z/result.json`（SHA256前缀 `967f2e25a1a4`）；`qformer-loop/runs/r10_field_cross-s20262027-20260906T181435551595Z/inference-all_queries.json`（SHA256前缀 `efecc214bcc2`）

<a id="k139"></a>
### K139 · `r10_field_cross_ffn-s20262027-20260906T181435546200Z`

- **方法**：{"field_repeat_mode":"cross_ffn","history_repeat_mode":"full"}；设置 {"model":"QFormerCross11Selective","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Select shared sublayers only in repeated field/history block calls
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.189161752；AUC=0.869620925；UAUC=0.788684406；gAUC=0.800595537；NDCG(10)=0.404826031。
- **开销**：参数 8115473；训练含验证 425.8074773326516 秒；峰值CUDA 16287956480 bytes；更新 1240。
- **原记录差值**：-0.001613134；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.001094893210284803,"AUC":-0.0016131337589585515,"UAUC":-0.002152606814183722,"gAUC":-0.0031951249132838866,"NDCG(10)":-0.0050301256174759}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.257696151733398,"mean_ms":12.260565638542175,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.001613134，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 10](#phase-10)。
- **后续假设**：保留完整block，以中性初始化的轮次向量/门控条件化；见K11。
- **原判读**：near-Cross8 mechanism candidate; lower AUC and ranking than parent
- **证据**：`qformer-loop/runs/r10_field_cross_ffn-s20262027-20260906T181435546200Z/manifest.json`（SHA256前缀 `a3e329939b89`）；`qformer-loop/runs/r10_field_cross_ffn-s20262027-20260906T181435546200Z/result.json`（SHA256前缀 `94fdec4c1a12`）；`qformer-loop/runs/r10_field_cross_ffn-s20262027-20260906T181435546200Z/inference-all_queries.json`（SHA256前缀 `63602779080c`）

<a id="k140"></a>
### K140 · `r10_field_self_cross-s20262027-20260906T181435546256Z`

- **方法**：{"field_repeat_mode":"self_cross","history_repeat_mode":"full"}；设置 {"model":"QFormerCross11Selective","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Select shared sublayers only in repeated field/history block calls
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.188755725；AUC=0.869916506；UAUC=0.791309943；gAUC=0.802951059；NDCG(10)=0.408904108。
- **开销**：参数 8115473；训练含验证 433.1369151175022 秒；峰值CUDA 16393732608 bytes；更新 1240。
- **原记录差值**：-0.001317552；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.000688866046865716,"AUC":-0.0013175523651969279,"UAUC":0.0004729303518572481,"gAUC":-0.0008396034695727916,"NDCG(10)":-0.0009520484860496836}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.54918384552002,"mean_ms":12.571839928627014,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.001317552，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 10](#phase-10)。
- **后续假设**：保留完整block，以中性初始化的轮次向量/门控条件化；见K11。
- **原判读**：UAUC/compute tradeoff versus parent; near-Cross11 AUC
- **证据**：`qformer-loop/runs/r10_field_self_cross-s20262027-20260906T181435546256Z/manifest.json`（SHA256前缀 `34298bdd0251`）；`qformer-loop/runs/r10_field_self_cross-s20262027-20260906T181435546256Z/result.json`（SHA256前缀 `aad9e0928748`）；`qformer-loop/runs/r10_field_self_cross-s20262027-20260906T181435546256Z/inference-all_queries.json`（SHA256前缀 `8e4106d8e6e2`）

<a id="k141"></a>
### K141 · `r10_history_cross-s20262027-20260906T181435546214Z`

- **方法**：{"field_repeat_mode":"full","history_repeat_mode":"cross"}；设置 {"model":"QFormerCross11Selective","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Select shared sublayers only in repeated field/history block calls
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.189320196；AUC=0.869270176；UAUC=0.787077833；gAUC=0.800027277；NDCG(10)=0.405667895。
- **开销**：参数 8115473；训练含验证 425.59011823683977 秒；峰值CUDA 16187686400 bytes；更新 1240。
- **原记录差值**：-0.001963883；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.0012533373295690042,"AUC":-0.001963882558722485,"UAUC":-0.0037591800916962903,"gAUC":-0.0037633855649207293,"NDCG(10)":-0.004188261541515748}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.07153606414795,"mean_ms":12.083022403717042,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.001963883，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 10](#phase-10)。
- **后续假设**：保留完整block，以中性初始化的轮次向量/门控条件化；见K11。
- **原判读**：deprioritized history-cross-only control
- **证据**：`qformer-loop/runs/r10_history_cross-s20262027-20260906T181435546214Z/manifest.json`（SHA256前缀 `47840c428faa`）；`qformer-loop/runs/r10_history_cross-s20262027-20260906T181435546214Z/result.json`（SHA256前缀 `9fdc299655e9`）；`qformer-loop/runs/r10_history_cross-s20262027-20260906T181435546214Z/inference-all_queries.json`（SHA256前缀 `37064a4e7601`）

<a id="k142"></a>
### K142 · `r10_history_cross_ffn-s20262027-20260906T181435553962Z`

- **方法**：{"field_repeat_mode":"full","history_repeat_mode":"cross_ffn"}；设置 {"model":"QFormerCross11Selective","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Select shared sublayers only in repeated field/history block calls
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.188681692；AUC=0.869989925；UAUC=0.789353101；gAUC=0.802087956；NDCG(10)=0.407261158。
- **开销**：参数 8115473；训练含验证 422.79869351536036 秒；峰值CUDA 16187686400 bytes；更新 1240。
- **原记录差值**：-0.001244134；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.0006148339731762154,"AUC":-0.0012441338543520697,"UAUC":-0.0014839114126772301,"gAUC":-0.0017027065572610134,"NDCG(10)":-0.0025949990513599586}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.211071968078613,"mean_ms":12.312717604637147,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.001244134，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 10](#phase-10)。
- **后续假设**：保留完整block，以中性初始化的轮次向量/门控条件化；见K11。
- **原判读**：near-Cross11 quality/cost alternative; lower AUC than full-block parent
- **证据**：`qformer-loop/runs/r10_history_cross_ffn-s20262027-20260906T181435553962Z/manifest.json`（SHA256前缀 `0535e4beed2e`）；`qformer-loop/runs/r10_history_cross_ffn-s20262027-20260906T181435553962Z/result.json`（SHA256前缀 `ebc05682a706`）；`qformer-loop/runs/r10_history_cross_ffn-s20262027-20260906T181435553962Z/inference-all_queries.json`（SHA256前缀 `7d1ff2f6aa6e`）

<a id="k143"></a>
### K143 · `r10_history_self_cross-s20262027-20260906T181435552759Z`

- **方法**：{"field_repeat_mode":"full","history_repeat_mode":"self_cross"}；设置 {"model":"QFormerCross11Selective","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Select shared sublayers only in repeated field/history block calls
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.189067269；AUC=0.869941804；UAUC=0.789529928；gAUC=0.801124748；NDCG(10)=0.409630563。
- **开销**：参数 8115473；训练含验证 437.1414013504982 秒；峰值CUDA 16763355648 bytes；更新 1240。
- **原记录差值**：-0.001292255；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.0010004100835458718,"AUC":-0.0012922546044072947,"UAUC":-0.0013070844227089928,"gAUC":-0.0026659143629417237,"NDCG(10)":-0.00022559403540434175}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.541200160980225,"mean_ms":12.541192746162414,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.001292255，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 10](#phase-10)。
- **后续假设**：保留完整block，以中性初始化的轮次向量/门控条件化；见K11。
- **原判读**：near-Cross11 alternative retaining self-attention and crossing
- **证据**：`qformer-loop/runs/r10_history_self_cross-s20262027-20260906T181435552759Z/manifest.json`（SHA256前缀 `4b0aedbe9cc0`）；`qformer-loop/runs/r10_history_self_cross-s20262027-20260906T181435552759Z/result.json`（SHA256前缀 `f89d64fcdd60`）；`qformer-loop/runs/r10_history_self_cross-s20262027-20260906T181435552759Z/inference-all_queries.json`（SHA256前缀 `a039ab6db53d`）

<a id="k144"></a>
### K144 · `r10_both_cross_ffn-s20262027-20260906T181435276632Z`

- **方法**：{"field_repeat_mode":"cross_ffn","history_repeat_mode":"cross_ffn"}；设置 {"model":"QFormerCross11Selective","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Select shared sublayers only in repeated field/history block calls
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.189369810；AUC=0.868907595；UAUC=0.789571178；gAUC=0.801548465；NDCG(10)=0.406448098。
- **开销**：参数 8115473；训练含验证 406.95897080004215 秒；峰值CUDA 15711762944 bytes；更新 1240。
- **原记录差值**：-0.002326463；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.001302951963933735,"AUC":-0.002326463109898036,"UAUC":-0.0012658343666847038,"gAUC":-0.0022421971808802565,"NDCG(10)":-0.003408059204528746}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":11.773856163024902,"mean_ms":11.790099143981934,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.002326463，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 10](#phase-10)。
- **后续假设**：保留完整block，以中性初始化的轮次向量/门控条件化；见K11。
- **原判读**：deprioritized joint self-attention deletion control
- **证据**：`qformer-loop/runs/r10_both_cross_ffn-s20262027-20260906T181435276632Z/manifest.json`（SHA256前缀 `424b376bd29d`）；`qformer-loop/runs/r10_both_cross_ffn-s20262027-20260906T181435276632Z/result.json`（SHA256前缀 `0e3d8a9ff295`）；`qformer-loop/runs/r10_both_cross_ffn-s20262027-20260906T181435276632Z/inference-all_queries.json`（SHA256前缀 `8ddac8b09480`）

<a id="k145"></a>
### K145 · `r10_both_cross-s20262027-20260906T181435276810Z`

- **方法**：{"field_repeat_mode":"cross","history_repeat_mode":"cross"}；设置 {"model":"QFormerCross11Selective","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Select shared sublayers only in repeated field/history block calls
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.189217327；AUC=0.869312585；UAUC=0.787470529；gAUC=0.799999378；NDCG(10)=0.404954926。
- **开销**：参数 8115473；训练含验证 400.01009085029364 秒；峰值CUDA 15342533120 bytes；更新 1240。
- **原记录差值**：-0.001921474；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.0011504684777273178,"AUC":-0.0019214737082607902,"UAUC":-0.0033664833275494077,"gAUC":-0.003791283717006322,"NDCG(10)":-0.0049012305969146075}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":11.467343807220459,"mean_ms":11.484782409667968,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.001921474，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 10](#phase-10)。
- **后续假设**：保留完整block，以中性初始化的轮次向量/门控条件化；见K11。
- **原判读**：deprioritized fastest selective-sublayer control
- **证据**：`qformer-loop/runs/r10_both_cross-s20262027-20260906T181435276810Z/manifest.json`（SHA256前缀 `1cac34229815`）；`qformer-loop/runs/r10_both_cross-s20262027-20260906T181435276810Z/result.json`（SHA256前缀 `1fc8435faa1d`）；`qformer-loop/runs/r10_both_cross-s20262027-20260906T181435276810Z/inference-all_queries.json`（SHA256前缀 `4b5a37797b2a`）

<a id="k146"></a>
### K146 · `diagnostics-round10-r10_field_self_cross`

- **方法**：Query Zero/Shuffle, extra attention off and inference-depth interventions
- **原假设**：Measure actual Query output reliance when repeated blocks execute selected sublayers
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"training_seed":20262027,"new_training_seed_values":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "rows": 57408,
  "users": 7176,
  "positives": 4472,
  "precision": "bf16",
  "scores": {
    "original": {
      "AUC": 0.8607152009832374,
      "logloss": 0.20276500284671783
    },
    "all_queries": {
      "AUC": 0.8607152009832374,
      "logloss": 0.20276500284671783
    },
    "q3_half": {
      "AUC": 0.8607178622452387,
      "logloss": 0.20291344821453094
    },
    "q0_zero": {
      "AUC": 0.86012527523363,
      "logloss": 0.20358015596866608
    },
    "q0_shuffle": {
      "AUC": 0.8549051401185703,
      "logloss": 0.20675282180309296
    },
    "q1_zero": {
      "AUC": 0.8602119331055721,
      "logloss": 0.2043738067150116
    },
    "q1_shuffle": {
      "AUC": 0.8565148593549222,
      "logloss": 0.20659694075584412
    },
    "q2_zero": {
      "AUC": 0.8594522082797251,
      "logloss": 0.2047424018383026
    },
    "q2_shuffle": {
      "AUC": 0.8564621199008192,
      "logloss": 0.20612920820713043
    },
    "q3_zero": {
      "AUC": 0.8604994043166311,
      "logloss": 0.2037874013185501
    },
    "q3_shuffle": {
      "AUC": 0.8586352937783175,
      "logloss": 0.20521384477615356
    },
    "q4_zero": {
      "AUC": 0.8610180737201003,
      "logloss": 0.20560941100120544
    },
    "q4_shuffle": {
      "AUC": 0.8591095813576348,
      "logloss": 0.20437081158161163
    },
    "q5_zero": {
      "AUC": 0.8599344395149049,
      "logloss": 0.2049524337053299
    },
    "q5_shuffle": {
      "AUC": 0.8544383315303212,
      "logloss": 0.20769628882408142
    },
    "q6_zero": {
      "AUC": 0.8598761409801771,
      "logloss": 0.20767709612846375
    },
    "q6_shuffle": {
      "AUC": 0.8567550995017982,
      "logloss": 0.2057032287120819
    },
    "q7_zero": {
      "AUC": 0.8601442335572195,
      "logloss": 0.20387373864650726
    },
    "q7_shuffle": {
      "AUC": 0.858235061094465,
      "logloss": 0.20488488674163818
    },
    "inference_placement_steps1": {
      "AUC": 0.8016959563754442,
      "logloss": 0.23441998660564423
    },
    "inference_placement_steps3": {
      "AUC": 0.844351035462406,
      "logloss": 0.21246148645877838
    },
    "extra_attention_off": {
      "AUC": 0.8596480940599146,
      "logloss": 0.2036614716053009
    }
  },
  "checkpoint_unchanged": true,
  "note": "subset diagnostics; within-block shuffle; blockwise loss means in toolkit differ from pooled full-validation metrics"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 10](#phase-10)。
- **后续假设**：保留完整block，以中性初始化的轮次向量/门控条件化；见K11。
- **证据**：`qformer-loop/reports/round10/r10_field_self_cross/summary.json`（SHA256前缀 `0e156fcee70b`）；`qformer-loop/reports/round10/r10_field_self_cross/toolkit-hooks/hook_report.json`（SHA256前缀 `a0ac8ea2aa6f`）；`qformer-loop/reports/round10/r10_field_self_cross/toolkit-perturbations/perturbation_report.json`（SHA256前缀 `de12a71330ec`）

<a id="k147"></a>
### K147 · `diagnostics-round10-r10_history_cross_ffn`

- **方法**：Query Zero/Shuffle, extra attention off and inference-depth interventions
- **原假设**：Measure actual Query output reliance when repeated blocks execute selected sublayers
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"training_seed":20262027,"new_training_seed_values":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "rows": 57408,
  "users": 7176,
  "positives": 4472,
  "precision": "bf16",
  "scores": {
    "original": {
      "AUC": 0.8608562119633847,
      "logloss": 0.20277981460094452
    },
    "all_queries": {
      "AUC": 0.8608562119633847,
      "logloss": 0.20277981460094452
    },
    "q3_half": {
      "AUC": 0.8605554048727421,
      "logloss": 0.20273548364639282
    },
    "q0_zero": {
      "AUC": 0.8604414078140193,
      "logloss": 0.2044495940208435
    },
    "q0_shuffle": {
      "AUC": 0.8507344229829763,
      "logloss": 0.2110581398010254
    },
    "q1_zero": {
      "AUC": 0.8601575229703238,
      "logloss": 0.2051168829202652
    },
    "q1_shuffle": {
      "AUC": 0.8577536514711254,
      "logloss": 0.20620910823345184
    },
    "q2_zero": {
      "AUC": 0.8603072675364831,
      "logloss": 0.20559191703796387
    },
    "q2_shuffle": {
      "AUC": 0.8570926869229878,
      "logloss": 0.20720762014389038
    },
    "q3_zero": {
      "AUC": 0.8598830919430708,
      "logloss": 0.20332789421081543
    },
    "q3_shuffle": {
      "AUC": 0.8572260140371348,
      "logloss": 0.20637021958827972
    },
    "q4_zero": {
      "AUC": 0.8585485957762343,
      "logloss": 0.20446400344371796
    },
    "q4_shuffle": {
      "AUC": 0.8566247525786699,
      "logloss": 0.20612290501594543
    },
    "q5_zero": {
      "AUC": 0.8605628902001485,
      "logloss": 0.20364996790885925
    },
    "q5_shuffle": {
      "AUC": 0.8580854432550679,
      "logloss": 0.2063411921262741
    },
    "q6_zero": {
      "AUC": 0.8600400747194507,
      "logloss": 0.2049277126789093
    },
    "q6_shuffle": {
      "AUC": 0.8598701383558855,
      "logloss": 0.20481908321380615
    },
    "q7_zero": {
      "AUC": 0.859979723211179,
      "logloss": 0.204807311296463
    },
    "q7_shuffle": {
      "AUC": 0.8585060345932294,
      "logloss": 0.20567989349365234
    },
    "inference_placement_steps1": {
      "AUC": 0.832164383855835,
      "logloss": 0.25836342573165894
    },
    "inference_placement_steps3": {
      "AUC": 0.8456732644786846,
      "logloss": 0.21572312712669373
    },
    "extra_attention_off": {
      "AUC": 0.8578514401769929,
      "logloss": 0.2050691545009613
    }
  },
  "checkpoint_unchanged": true,
  "note": "subset diagnostics; within-block shuffle; blockwise loss means in toolkit differ from pooled full-validation metrics"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 10](#phase-10)。
- **后续假设**：保留完整block，以中性初始化的轮次向量/门控条件化；见K11。
- **证据**：`qformer-loop/reports/round10/r10_history_cross_ffn/summary.json`（SHA256前缀 `7bd5f5b5e412`）；`qformer-loop/reports/round10/r10_history_cross_ffn/toolkit-hooks/hook_report.json`（SHA256前缀 `ff906ebbeef0`）；`qformer-loop/reports/round10/r10_history_cross_ffn/toolkit-perturbations/perturbation_report.json`（SHA256前缀 `59f871e65385`）

<a id="phase-11"></a>
## 阶段 11：轮次向量与Query门控

**方法**：字段/历史/双位置零初始化轮次embedding与Query条件门控。

**归因与边界**：8组均未刷新r7；history_embedding非常接近父模型，但校准/排序存在代价。非零参数或敏感性不等于重训收益。

**历史后续假设**：完整保留子层与norm，仅重排信息流；见K12。

<a id="k148"></a>
### K148 · `round11-diagnostic-launch-failure-1`

- **方法**：initialize project dependency paths by importing diagnostic_round2 first
- **原假设**：Run fixed-checkpoint conditioning probe
- **状态／预算**：原记录 `crashed`；{"training_updates":0,"new_training_seed_values":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "decision": "crashed",
  "validity": {
    "valid": false,
    "violations": [
      "sklearn imported before diagnostic_round2 configured project runtime/deps"
    ],
    "notes": [
      "failed before checkpoint loading; corrected probe subsequently completed"
    ]
  },
  "falsification_result": null,
  "note": "本条未保存独立性能指标；保留运行状态与证据位置，不补造数值。"
}
```

</details>

- **归因**：这是执行或有效性问题；本条不能用作结构性能反证。修复后完整训练需用独立运行ID核对。 本阶段证据解释见[阶段 11](#phase-11)。
- **后续假设**：完整保留子层与norm，仅重排信息流；见K12。
- **失败后续门槛**：先复核同配置/初始函数/有效梯度及完整预算，再判断科学假设；不把同一次修复重复算成独立支持。
- **证据**：`qformer-loop/reports/round11/failures.json`（SHA256前缀 `8b8ecd7613a8`）；`qformer-loop/reports/round11/conditioning-probe-import-failure.log`

<a id="k149"></a>
### K149 · `r11_field_embedding-s20262027-20260907T032608534452Z`

- **方法**：{"repeat_condition":"embedding","condition_scope":"field"}；设置 {"model":"QFormerCross11Conditioned","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Condition repeated full shared blocks with initially neutral embeddings or Query gates
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.189431120；AUC=0.868696874；UAUC=0.790487548；gAUC=0.803107081；NDCG(10)=0.407259455。
- **开销**：参数 8115729；训练含验证 436.7830817848444 秒；峰值CUDA 16763358720 bytes；更新 1240。
- **原记录差值**：-0.002537184；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.0013642617597904583,"AUC":-0.0025371843727489685,"UAUC":-0.00034946455370832297,"gAUC":-0.000683580941076678,"NDCG(10)":-0.0025967023341569306}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.572896003723145,"mean_ms":12.57896318435669,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.002537184，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 11](#phase-11)。
- **后续假设**：完整保留子层与norm，仅重排信息流；见K12。
- **原判读**：保留完整负对照；同 seed AUC 未超过父模型
- **证据**：`qformer-loop/runs/r11_field_embedding-s20262027-20260907T032608534452Z/manifest.json`（SHA256前缀 `55aa8f751a15`）；`qformer-loop/runs/r11_field_embedding-s20262027-20260907T032608534452Z/result.json`（SHA256前缀 `e44a19a43359`）；`qformer-loop/runs/r11_field_embedding-s20262027-20260907T032608534452Z/inference-all_queries.json`（SHA256前缀 `bfde3424e2d1`）

<a id="k150"></a>
### K150 · `r11_history_embedding-s20262027-20260907T032608681208Z`

- **方法**：{"repeat_condition":"embedding","condition_scope":"history"}；设置 {"model":"QFormerCross11Conditioned","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Condition repeated full shared blocks with initially neutral embeddings or Query gates
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.188297579；AUC=0.871217220；UAUC=0.790689022；gAUC=0.803237653；NDCG(10)=0.407377845。
- **开销**：参数 8115729；训练含验证 438.6887850910425 秒；峰值CUDA 16830467584 bytes；更新 1240。
- **原记录差值**：-0.000016838；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.00023072079487182484,"AUC":-1.683847096389801e-05,"UAUC":-0.00014799086573313858,"gAUC":-0.000553009155905948,"NDCG(10)":-0.002478311747643569}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.584047794342041,"mean_ms":12.58670802116394,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000016838，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 11](#phase-11)。
- **后续假设**：完整保留子层与norm，仅重排信息流；见K12。
- **原判读**：本轮 AUC 最佳且接近父模型；保留历史条件化候选，Logloss/排序尚无优势
- **证据**：`qformer-loop/runs/r11_history_embedding-s20262027-20260907T032608681208Z/manifest.json`（SHA256前缀 `eb3da013b505`）；`qformer-loop/runs/r11_history_embedding-s20262027-20260907T032608681208Z/result.json`（SHA256前缀 `b83529c7529a`）；`qformer-loop/runs/r11_history_embedding-s20262027-20260907T032608681208Z/inference-all_queries.json`（SHA256前缀 `536cf6febb4a`）

<a id="k151"></a>
### K151 · `r11_both_embedding-s20262027-20260907T032608534442Z`

- **方法**：{"repeat_condition":"embedding","condition_scope":"both"}；设置 {"model":"QFormerCross11Conditioned","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Condition repeated full shared blocks with initially neutral embeddings or Query gates
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.189151495；AUC=0.869567733；UAUC=0.790108503；gAUC=0.802766086；NDCG(10)=0.411121530。
- **开销**：参数 8115985；训练含验证 437.4337214231491 秒；峰值CUDA 16830470656 bytes；更新 1240。
- **原记录差值**：-0.001666325；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.0010846366644812755,"AUC":-0.0016663252359333525,"UAUC":-0.0007285100154610369,"gAUC":-0.0010245758260805093,"NDCG(10)":0.001265372824651445}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.580863952636719,"mean_ms":12.584906435012817,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.001666325，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 11](#phase-11)。
- **后续假设**：完整保留子层与norm，仅重排信息流；见K12。
- **原判读**：AUC 接近 Cross8、NDCG 高于父模型；保留排序取舍对照
- **证据**：`qformer-loop/runs/r11_both_embedding-s20262027-20260907T032608534442Z/manifest.json`（SHA256前缀 `ba3f5500cb83`）；`qformer-loop/runs/r11_both_embedding-s20262027-20260907T032608534442Z/result.json`（SHA256前缀 `988c0736758c`）；`qformer-loop/runs/r11_both_embedding-s20262027-20260907T032608534442Z/inference-all_queries.json`（SHA256前缀 `5f99b38c423f`）

<a id="k152"></a>
### K152 · `r11_query_embedding-s20262027-20260907T032608496547Z`

- **方法**：{"repeat_condition":"query_embedding","condition_scope":"both"}；设置 {"model":"QFormerCross11Conditioned","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Condition repeated full shared blocks with initially neutral embeddings or Query gates
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.188818667；AUC=0.870100315；UAUC=0.789610937；gAUC=0.802128759；NDCG(10)=0.408931038。
- **开销**：参数 8119569；训练含验证 437.7164953276515 秒；峰值CUDA 16830513664 bytes；更新 1240。
- **原记录差值**：-0.001133744；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.0007518088357940933,"AUC":-0.0011337436754063068,"UAUC":-0.00122607558973431,"gAUC":-0.0016619033284577256,"NDCG(10)":-0.0009251184679078261}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.57524824142456,"mean_ms":12.579497623443604,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.001133744，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 11](#phase-11)。
- **后续假设**：完整保留子层与norm，仅重排信息流；见K12。
- **原判读**：保留完整负对照；同 seed AUC 未超过父模型
- **证据**：`qformer-loop/runs/r11_query_embedding-s20262027-20260907T032608496547Z/manifest.json`（SHA256前缀 `fe2dd60f863b`）；`qformer-loop/runs/r11_query_embedding-s20262027-20260907T032608496547Z/result.json`（SHA256前缀 `f7a9bf36b022`）；`qformer-loop/runs/r11_query_embedding-s20262027-20260907T032608496547Z/inference-all_queries.json`（SHA256前缀 `bcce1c5abcb0`）

<a id="k153"></a>
### K153 · `r11_field_gate-s20262027-20260907T032608540320Z`

- **方法**：{"repeat_condition":"gate","condition_scope":"field"}；设置 {"model":"QFormerCross11Conditioned","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Condition repeated full shared blocks with initially neutral embeddings or Query gates
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.188559630；AUC=0.870516036；UAUC=0.790008666；gAUC=0.802585267；NDCG(10)=0.408406451。
- **开销**：参数 8115730；训练含验证 438.52260944992304 秒；峰值CUDA 16864942080 bytes；更新 1240。
- **原记录差值**：-0.000718023；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.0004927720082090747,"AUC":-0.0007180225187123357,"UAUC":-0.0008283469374894548,"gAUC":-0.0012053950047672402,"NDCG(10)":-0.0014497054425067213}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.626271724700928,"mean_ms":12.674772834777832,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000718023，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 11](#phase-11)。
- **后续假设**：完整保留子层与norm，仅重排信息流；见K12。
- **原判读**：保留完整负对照；同 seed AUC 未超过父模型
- **证据**：`qformer-loop/runs/r11_field_gate-s20262027-20260907T032608540320Z/manifest.json`（SHA256前缀 `307abd6e7654`）；`qformer-loop/runs/r11_field_gate-s20262027-20260907T032608540320Z/result.json`（SHA256前缀 `fd8d09332b71`）；`qformer-loop/runs/r11_field_gate-s20262027-20260907T032608540320Z/inference-all_queries.json`（SHA256前缀 `7d88f7135ef4`）

<a id="k154"></a>
### K154 · `r11_history_gate-s20262027-20260907T032608594282Z`

- **方法**：{"repeat_condition":"gate","condition_scope":"history"}；设置 {"model":"QFormerCross11Conditioned","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Condition repeated full shared blocks with initially neutral embeddings or Query gates
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.188870923；AUC=0.870675144；UAUC=0.790853716；gAUC=0.803611558；NDCG(10)=0.406692738。
- **开销**：参数 8115730；训练含验证 448.4191065579653 秒；峰值CUDA 16763360256 bytes；更新 1240。
- **原记录差值**：-0.000558914；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.0008040648216829094,"AUC":-0.0005589140881386845,"UAUC":1.6703745511104096e-05,"gAUC":-0.00017910465819337418,"NDCG(10)":-0.003163419154042224}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.615615844726562,"mean_ms":12.628308868408203,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000558914，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 11](#phase-11)。
- **后续假设**：完整保留子层与norm，仅重排信息流；见K12。
- **原判读**：AUC 高于 Cross8/Cross11，UAUC 略高父模型；保留门控对照
- **证据**：`qformer-loop/runs/r11_history_gate-s20262027-20260907T032608594282Z/manifest.json`（SHA256前缀 `663f5d67dc07`）；`qformer-loop/runs/r11_history_gate-s20262027-20260907T032608594282Z/result.json`（SHA256前缀 `9cbf874216b6`）；`qformer-loop/runs/r11_history_gate-s20262027-20260907T032608594282Z/inference-all_queries.json`（SHA256前缀 `6d79b5da54c5`）

<a id="k155"></a>
### K155 · `r11_both_gate-s20262027-20260907T032608561082Z`

- **方法**：{"repeat_condition":"gate","condition_scope":"both"}；设置 {"model":"QFormerCross11Conditioned","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Condition repeated full shared blocks with initially neutral embeddings or Query gates
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.188777032；AUC=0.870136722；UAUC=0.789757510；gAUC=0.802199957；NDCG(10)=0.407201295。
- **开销**：参数 8115987；训练含验证 439.6968698129058 秒；峰值CUDA 16864946688 bytes；更新 1240。
- **原记录差值**：-0.001097337；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.0007101730613209845,"AUC":-0.0010973366667321738,"UAUC":-0.0010795029550048074,"gAUC":-0.0015907053248228609,"NDCG(10)":-0.002654862191871332}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.674544334411621,"mean_ms":12.685861611366272,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.001097337，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 11](#phase-11)。
- **后续假设**：完整保留子层与norm，仅重排信息流；见K12。
- **原判读**：保留完整负对照；同 seed AUC 未超过父模型
- **证据**：`qformer-loop/runs/r11_both_gate-s20262027-20260907T032608561082Z/manifest.json`（SHA256前缀 `c3dc86aee5bf`）；`qformer-loop/runs/r11_both_gate-s20262027-20260907T032608561082Z/result.json`（SHA256前缀 `eddb5b012819`）；`qformer-loop/runs/r11_both_gate-s20262027-20260907T032608561082Z/inference-all_queries.json`（SHA256前缀 `5298b63b88b2`）

<a id="k156"></a>
### K156 · `r11_embedding_gate-s20262027-20260907T032659791882Z`

- **方法**：{"repeat_condition":"embedding_gate","condition_scope":"both"}；设置 {"model":"QFormerCross11Conditioned","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Condition repeated full shared blocks with initially neutral embeddings or Query gates
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.188557220；AUC=0.870697419；UAUC=0.788224715；gAUC=0.800769832；NDCG(10)=0.407896985。
- **开销**：参数 8116499；训练含验证 444.3694644421339 秒；峰值CUDA 16931537408 bytes；更新 1240。
- **原记录差值**：-0.000536639；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.0004903612652140954,"AUC":-0.0005366389795776749,"UAUC":-0.002612297272787778,"gAUC":-0.0030208300863603377,"NDCG(10)":-0.001959172004164178}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.710240364074707,"mean_ms":12.749687242507935,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000536639，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 11](#phase-11)。
- **后续假设**：完整保留子层与norm，仅重排信息流；见K12。
- **原判读**：保留完整负对照；同 seed AUC 未超过父模型
- **证据**：`qformer-loop/runs/r11_embedding_gate-s20262027-20260907T032659791882Z/manifest.json`（SHA256前缀 `16970a642448`）；`qformer-loop/runs/r11_embedding_gate-s20262027-20260907T032659791882Z/result.json`（SHA256前缀 `9577bf6cdaff`）；`qformer-loop/runs/r11_embedding_gate-s20262027-20260907T032659791882Z/inference-all_queries.json`（SHA256前缀 `25a01ab0ded3`）

<a id="k157"></a>
### K157 · `diagnostics-round11-r11_history_embedding`

- **方法**：Query Zero/Shuffle, extra attention off and inference-depth interventions
- **原假设**：Measure Query and recurrence reliance in initially neutral conditioning models
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"training_seed":20262027,"new_training_seed_values":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "rows": 57408,
  "users": 7176,
  "positives": 4472,
  "precision": "bf16",
  "scores": {
    "original": {
      "AUC": 0.8620953420176197,
      "logloss": 0.2023247927427292
    },
    "all_queries": {
      "AUC": 0.8620953420176197,
      "logloss": 0.2023247927427292
    },
    "q3_half": {
      "AUC": 0.8622352230174731,
      "logloss": 0.20258347690105438
    },
    "q0_zero": {
      "AUC": 0.8613078576945652,
      "logloss": 0.204599529504776
    },
    "q0_shuffle": {
      "AUC": 0.857271192127774,
      "logloss": 0.20727603137493134
    },
    "q1_zero": {
      "AUC": 0.8610681392395259,
      "logloss": 0.20341946184635162
    },
    "q1_shuffle": {
      "AUC": 0.8606796815839723,
      "logloss": 0.20354467630386353
    },
    "q2_zero": {
      "AUC": 0.8605446373222008,
      "logloss": 0.2087322175502777
    },
    "q2_shuffle": {
      "AUC": 0.8570404839455105,
      "logloss": 0.20882387459278107
    },
    "q3_zero": {
      "AUC": 0.8622093496369059,
      "logloss": 0.2061953991651535
    },
    "q3_shuffle": {
      "AUC": 0.8600147335912838,
      "logloss": 0.20452436804771423
    },
    "q4_zero": {
      "AUC": 0.8613745708862871,
      "logloss": 0.20310045778751373
    },
    "q4_shuffle": {
      "AUC": 0.8587529912584893,
      "logloss": 0.20655031502246857
    },
    "q5_zero": {
      "AUC": 0.8611379318915634,
      "logloss": 0.20427782833576202
    },
    "q5_shuffle": {
      "AUC": 0.8555936423920821,
      "logloss": 0.20811434090137482
    },
    "q6_zero": {
      "AUC": 0.8612763280761891,
      "logloss": 0.20375263690948486
    },
    "q6_shuffle": {
      "AUC": 0.8567319528587258,
      "logloss": 0.20714080333709717
    },
    "q7_zero": {
      "AUC": 0.8611550315559776,
      "logloss": 0.20454740524291992
    },
    "q7_shuffle": {
      "AUC": 0.8578168057529489,
      "logloss": 0.2061692178249359
    },
    "inference_placement_steps1": {
      "AUC": 0.7762735224301638,
      "logloss": 0.25892677903175354
    },
    "inference_placement_steps3": {
      "AUC": 0.8311601629760229,
      "logloss": 0.2202664613723755
    },
    "extra_attention_off": {
      "AUC": 0.8587848017878544,
      "logloss": 0.2060232162475586
    }
  },
  "checkpoint_unchanged": true,
  "note": "subset diagnostics; within-block shuffle; blockwise loss means in toolkit differ from pooled full-validation metrics"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 11](#phase-11)。
- **后续假设**：完整保留子层与norm，仅重排信息流；见K12。
- **证据**：`qformer-loop/reports/round11/r11_history_embedding/summary.json`（SHA256前缀 `c4f4dd97181f`）；`qformer-loop/reports/round11/r11_history_embedding/toolkit-hooks/hook_report.json`（SHA256前缀 `78aa162ca33b`）；`qformer-loop/reports/round11/r11_history_embedding/toolkit-perturbations/perturbation_report.json`（SHA256前缀 `a3b32048ea66`）

<a id="k158"></a>
### K158 · `diagnostics-round11-r11_history_gate`

- **方法**：Query Zero/Shuffle, extra attention off and inference-depth interventions
- **原假设**：Measure Query and recurrence reliance in initially neutral conditioning models
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"training_seed":20262027,"new_training_seed_values":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "rows": 57408,
  "users": 7176,
  "positives": 4472,
  "precision": "bf16",
  "scores": {
    "original": {
      "AUC": 0.8622876118608679,
      "logloss": 0.2026587873697281
    },
    "all_queries": {
      "AUC": 0.8622876118608679,
      "logloss": 0.2026587873697281
    },
    "q3_half": {
      "AUC": 0.8623996404305546,
      "logloss": 0.2027561217546463
    },
    "q0_zero": {
      "AUC": 0.8618066077631665,
      "logloss": 0.2045663446187973
    },
    "q0_shuffle": {
      "AUC": 0.8575655023597537,
      "logloss": 0.20743662118911743
    },
    "q1_zero": {
      "AUC": 0.8619619156341758,
      "logloss": 0.2039002627134323
    },
    "q1_shuffle": {
      "AUC": 0.8584925360809678,
      "logloss": 0.20634907484054565
    },
    "q2_zero": {
      "AUC": 0.8614451914865029,
      "logloss": 0.20560117065906525
    },
    "q2_shuffle": {
      "AUC": 0.8563034474342798,
      "logloss": 0.20953713357448578
    },
    "q3_zero": {
      "AUC": 0.862203581457124,
      "logloss": 0.20531578361988068
    },
    "q3_shuffle": {
      "AUC": 0.8599804265447081,
      "logloss": 0.2056674063205719
    },
    "q4_zero": {
      "AUC": 0.8619209089661177,
      "logloss": 0.20298001170158386
    },
    "q4_shuffle": {
      "AUC": 0.8594905198919787,
      "logloss": 0.20609210431575775
    },
    "q5_zero": {
      "AUC": 0.8619080398634407,
      "logloss": 0.20433452725410461
    },
    "q5_shuffle": {
      "AUC": 0.8579060277297079,
      "logloss": 0.20713600516319275
    },
    "q6_zero": {
      "AUC": 0.8617943976396516,
      "logloss": 0.21293756365776062
    },
    "q6_shuffle": {
      "AUC": 0.856436563337157,
      "logloss": 0.2092454582452774
    },
    "q7_zero": {
      "AUC": 0.8611221903156152,
      "logloss": 0.20368094742298126
    },
    "q7_shuffle": {
      "AUC": 0.8601087141579544,
      "logloss": 0.2044895887374878
    },
    "inference_placement_steps1": {
      "AUC": 0.8203831163759905,
      "logloss": 0.23958566784858704
    },
    "inference_placement_steps3": {
      "AUC": 0.8238012539629993,
      "logloss": 0.22397945821285248
    },
    "extra_attention_off": {
      "AUC": 0.8593364307099969,
      "logloss": 0.2058333456516266
    }
  },
  "checkpoint_unchanged": true,
  "note": "subset diagnostics; within-block shuffle; blockwise loss means in toolkit differ from pooled full-validation metrics"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 11](#phase-11)。
- **后续假设**：完整保留子层与norm，仅重排信息流；见K12。
- **证据**：`qformer-loop/reports/round11/r11_history_gate/summary.json`（SHA256前缀 `1dc186acba1a`）；`qformer-loop/reports/round11/r11_history_gate/toolkit-hooks/hook_report.json`（SHA256前缀 `f175ede399d7`）；`qformer-loop/reports/round11/r11_history_gate/toolkit-perturbations/perturbation_report.json`（SHA256前缀 `e46dc995276e`）

<a id="k159"></a>
### K159 · `probe-round11-r11_history_embedding-conditioning`

- **方法**：Neutralize learned field/history conditioning separately and jointly
- **原假设**：Measure learned repeat conditioning reliance without retraining
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"training_seed":20262027,"new_training_seed_values":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "rows": 57408,
  "scores": {
    "original": {
      "AUC": 0.8620953420176197,
      "logloss": 0.2023247852803083
    },
    "history_off": {
      "AUC": 0.862080109460832,
      "logloss": 0.20232856837559776
    },
    "all_off": {
      "AUC": 0.862080109460832,
      "logloss": 0.20232856837559776
    }
  },
  "checkpoint_unchanged": true,
  "note": "fixed-checkpoint subset intervention; neutralizing the branch is not retraining evidence"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 11](#phase-11)。
- **后续假设**：完整保留子层与norm，仅重排信息流；见K12。
- **证据**：`qformer-loop/reports/round11/r11_history_embedding-conditioning/result.json`（SHA256前缀 `0065831d9fa1`）

<a id="k160"></a>
### K160 · `probe-round11-r11_history_gate-conditioning`

- **方法**：Neutralize learned field/history conditioning separately and jointly
- **原假设**：Measure learned repeat conditioning reliance without retraining
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"training_seed":20262027,"new_training_seed_values":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "rows": 57408,
  "scores": {
    "original": {
      "AUC": 0.8622876118608679,
      "logloss": 0.20265878728740766
    },
    "history_off": {
      "AUC": 0.8623886595566306,
      "logloss": 0.20245838163333188
    },
    "all_off": {
      "AUC": 0.8623886595566306,
      "logloss": 0.20245838163333188
    }
  },
  "checkpoint_unchanged": true,
  "note": "fixed-checkpoint subset intervention; neutralizing the branch is not retraining evidence"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 11](#phase-11)。
- **后续假设**：完整保留子层与norm，仅重排信息流；见K12。
- **证据**：`qformer-loop/reports/round11/r11_history_gate-conditioning/result.json`（SHA256前缀 `31d0481ecb4e`）

<a id="phase-12"></a>
## 阶段 12：完整子层顺序

**方法**：Self/Cross/FFN顺序改为CAF/AFC，比较仅第二轮改变与两轮均改变。

**归因与边界**：8组均未超过标准顺序；全轮CAF常优于只改第二轮，但各指标不一致。

**历史后续假设**：在两轮之间更新历史源，分开Query写入与历史self-attention；见K13。

<a id="k161"></a>
### K161 · `r12_field_caf-s20262027-20260907T035559099466Z`

- **方法**：{"field_order":"caf","history_order":"acf","order_passes":"repeat"}；设置 {"model":"QFormerCross11Ordered","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Change source-read and Query-mixing order while preserving full shared sublayers
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.189972447；AUC=0.867321725；UAUC=0.789560497；gAUC=0.801484565；NDCG(10)=0.407772878。
- **开销**：参数 8115473；训练含验证 436.32649986445904 秒；峰值CUDA 16762962432 bytes；更新 1240。
- **原记录差值**：-0.003912333；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.0019055882967284332,"AUC":-0.003912333458493622,"UAUC":-0.001276516024226293,"gAUC":-0.0023060971443044087,"NDCG(10)":-0.0020832792129644173}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.667216300964355,"mean_ms":12.671361637115478,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.003912333，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 12](#phase-12)。
- **后续假设**：在两轮之间更新历史源，分开Query写入与历史self-attention；见K13。
- **原判读**：保留结构负对照；AUC 未超过父模型
- **证据**：`qformer-loop/runs/r12_field_caf-s20262027-20260907T035559099466Z/manifest.json`（SHA256前缀 `e6fdbad8f6b4`）；`qformer-loop/runs/r12_field_caf-s20262027-20260907T035559099466Z/result.json`（SHA256前缀 `5af992718aba`）；`qformer-loop/runs/r12_field_caf-s20262027-20260907T035559099466Z/inference-all_queries.json`（SHA256前缀 `aecbc60a3863`）

<a id="k162"></a>
### K162 · `r12_history_caf-s20262027-20260907T035559393353Z`

- **方法**：{"field_order":"acf","history_order":"caf","order_passes":"repeat"}；设置 {"model":"QFormerCross11Ordered","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Change source-read and Query-mixing order while preserving full shared sublayers
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.189100459；AUC=0.869118351；UAUC=0.788578092；gAUC=0.800458992；NDCG(10)=0.408217227。
- **开销**：参数 8115473；训练含验证 438.5931016281247 秒；峰值CUDA 16187686400 bytes；更新 1240。
- **原记录差值**：-0.002115707；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.0010336004404887522,"AUC":-0.00211570747482559,"UAUC":-0.0022589207579423176,"gAUC":-0.003331669960017769,"NDCG(10)":-0.0016389300617971059}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.670383930206299,"mean_ms":12.682019209861755,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.002115707，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 12](#phase-12)。
- **后续假设**：在两轮之间更新历史源，分开Query写入与历史self-attention；见K13。
- **原判读**：保留结构负对照；AUC 未超过父模型
- **证据**：`qformer-loop/runs/r12_history_caf-s20262027-20260907T035559393353Z/manifest.json`（SHA256前缀 `f526da49b089`）；`qformer-loop/runs/r12_history_caf-s20262027-20260907T035559393353Z/result.json`（SHA256前缀 `2eb98bba9b27`）；`qformer-loop/runs/r12_history_caf-s20262027-20260907T035559393353Z/inference-all_queries.json`（SHA256前缀 `40e90d3aef0d`）

<a id="k163"></a>
### K163 · `r12_both_caf-s20262027-20260907T035559396151Z`

- **方法**：{"field_order":"caf","history_order":"caf","order_passes":"repeat"}；设置 {"model":"QFormerCross11Ordered","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Change source-read and Query-mixing order while preserving full shared sublayers
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.188630226；AUC=0.870212443；UAUC=0.790558036；gAUC=0.802316270；NDCG(10)=0.408777796。
- **开销**：参数 8115473；训练含验证 439.3573826998472 秒；峰值CUDA 16186768896 bytes；更新 1240。
- **原记录差值**：-0.001021615；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.0005633672429772307,"AUC":-0.0010216153165267805,"UAUC":-0.00027897666894960427,"gAUC":-0.0014743918917334842,"NDCG(10)":-0.0010783607235007597}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.787951946258545,"mean_ms":13.129608726501464,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.001021615，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 12](#phase-12)。
- **后续假设**：在两轮之间更新历史源，分开Query写入与历史self-attention；见K13。
- **原判读**：重复双位置重排候选，AUC 高于 Cross11，Logloss/排序优于两轮均重排
- **证据**：`qformer-loop/runs/r12_both_caf-s20262027-20260907T035559396151Z/manifest.json`（SHA256前缀 `65c54a77bf66`）；`qformer-loop/runs/r12_both_caf-s20262027-20260907T035559396151Z/result.json`（SHA256前缀 `072224821c09`）；`qformer-loop/runs/r12_both_caf-s20262027-20260907T035559396151Z/inference-all_queries.json`（SHA256前缀 `8407cb6a835c`）

<a id="k164"></a>
### K164 · `r12_field_caf_all-s20262027-20260907T035559123398Z`

- **方法**：{"field_order":"caf","history_order":"acf","order_passes":"both"}；设置 {"model":"QFormerCross11Ordered","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Change source-read and Query-mixing order while preserving full shared sublayers
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.189683542；AUC=0.869175580；UAUC=0.790719043；gAUC=0.802690589；NDCG(10)=0.409544747。
- **开销**：参数 8115473；训练含验证 441.2975614592433 秒；峰值CUDA 16763027968 bytes；更新 1240。
- **原记录差值**：-0.002058479；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.0016166839850845216,"AUC":-0.0020584786640013286,"UAUC":-0.0001179702184948006,"gAUC":-0.00110007281104596,"NDCG(10)":-0.0003114101086209575}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.668047904968262,"mean_ms":12.68697440624237,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.002058479，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 12](#phase-12)。
- **后续假设**：在两轮之间更新历史源，分开Query写入与历史self-attention；见K13。
- **原判读**：保留结构负对照；AUC 未超过父模型
- **证据**：`qformer-loop/runs/r12_field_caf_all-s20262027-20260907T035559123398Z/manifest.json`（SHA256前缀 `a7637f9cea76`）；`qformer-loop/runs/r12_field_caf_all-s20262027-20260907T035559123398Z/result.json`（SHA256前缀 `948d007740ee`）；`qformer-loop/runs/r12_field_caf_all-s20262027-20260907T035559123398Z/inference-all_queries.json`（SHA256前缀 `afc2a4b45517`）

<a id="k165"></a>
### K165 · `r12_history_caf_all-s20262027-20260907T035559379528Z`

- **方法**：{"field_order":"acf","history_order":"caf","order_passes":"both"}；设置 {"model":"QFormerCross11Ordered","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Change source-read and Query-mixing order while preserving full shared sublayers
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.188924399；AUC=0.869743604；UAUC=0.789433724；gAUC=0.801591338；NDCG(10)=0.407440491。
- **开销**：参数 8115473；训练含验证 440.95589496940374 秒；峰值CUDA 16186965504 bytes；更新 1240。
- **原记录差值**：-0.001490455；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.0008575408230736825,"AUC":-0.0014904549565956104,"UAUC":-0.0014032887367494418,"gAUC":-0.002199323789536134,"NDCG(10)":-0.0024156658186336166}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.66427230834961,"mean_ms":12.674885654449463,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.001490455，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 12](#phase-12)。
- **后续假设**：在两轮之间更新历史源，分开Query写入与历史self-attention；见K13。
- **原判读**：AUC 位于 Cross8/Cross11 之间，保留基线附近候选
- **证据**：`qformer-loop/runs/r12_history_caf_all-s20262027-20260907T035559379528Z/manifest.json`（SHA256前缀 `fe3d21369537`）；`qformer-loop/runs/r12_history_caf_all-s20262027-20260907T035559379528Z/result.json`（SHA256前缀 `feed4560300f`）；`qformer-loop/runs/r12_history_caf_all-s20262027-20260907T035559379528Z/inference-all_queries.json`（SHA256前缀 `9d182d6f07d8`）

<a id="k166"></a>
### K166 · `r12_both_caf_all-s20262027-20260907T035559385613Z`

- **方法**：{"field_order":"caf","history_order":"caf","order_passes":"both"}；设置 {"model":"QFormerCross11Ordered","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Change source-read and Query-mixing order while preserving full shared sublayers
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.189074800；AUC=0.870232594；UAUC=0.789496150；gAUC=0.801448658；NDCG(10)=0.405684246。
- **开销**：参数 8115473；训练含验证 445.3329227119684 秒；峰值CUDA 16186768896 bytes；更新 1240。
- **原记录差值**：-0.001001464；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.0010079412655501496,"AUC":-0.0010014644216748136,"UAUC":-0.0013408631378354352,"gAUC":-0.0023420039831157924,"NDCG(10)":-0.004171911013021001}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.673184394836426,"mean_ms":12.74129843711853,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.001001464，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 12](#phase-12)。
- **后续假设**：在两轮之间更新历史源，分开Query写入与历史self-attention；见K13。
- **原判读**：AUC 高于 Cross11，保留全轮次重排对照
- **证据**：`qformer-loop/runs/r12_both_caf_all-s20262027-20260907T035559385613Z/manifest.json`（SHA256前缀 `7710588258e2`）；`qformer-loop/runs/r12_both_caf_all-s20262027-20260907T035559385613Z/result.json`（SHA256前缀 `c0c069e954f3`）；`qformer-loop/runs/r12_both_caf_all-s20262027-20260907T035559385613Z/inference-all_queries.json`（SHA256前缀 `ad1136726723`）

<a id="k167"></a>
### K167 · `r12_field_afc-s20262027-20260907T035559388666Z`

- **方法**：{"field_order":"afc","history_order":"acf","order_passes":"repeat"}；设置 {"model":"QFormerCross11Ordered","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Change source-read and Query-mixing order while preserving full shared sublayers
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.189605837；AUC=0.869277597；UAUC=0.790103581；gAUC=0.802341365；NDCG(10)=0.404847088。
- **开销**：参数 8115473；训练含验证 436.16274362802505 秒；峰值CUDA 16763486720 bytes；更新 1240。
- **原记录差值**：-0.001956461；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.0015389788976405139,"AUC":-0.0019564613469346126,"UAUC":-0.0007334319018776903,"gAUC":-0.0014492973815962706,"NDCG(10)":-0.005009068564206742}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.65828800201416,"mean_ms":12.788086414337158,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.001956461，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 12](#phase-12)。
- **后续假设**：在两轮之间更新历史源，分开Query写入与历史self-attention；见K13。
- **原判读**：保留结构负对照；AUC 未超过父模型
- **证据**：`qformer-loop/runs/r12_field_afc-s20262027-20260907T035559388666Z/manifest.json`（SHA256前缀 `e4d3f0339595`）；`qformer-loop/runs/r12_field_afc-s20262027-20260907T035559388666Z/result.json`（SHA256前缀 `43a452595da8`）；`qformer-loop/runs/r12_field_afc-s20262027-20260907T035559388666Z/inference-all_queries.json`（SHA256前缀 `5fea2715c653`）

<a id="k168"></a>
### K168 · `r12_history_afc-s20262027-20260907T035559125014Z`

- **方法**：{"field_order":"acf","history_order":"afc","order_passes":"repeat"}；设置 {"model":"QFormerCross11Ordered","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Change source-read and Query-mixing order while preserving full shared sublayers
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.188526579；AUC=0.870325317；UAUC=0.788787266；gAUC=0.801079038；NDCG(10)=0.409680843。
- **开销**：参数 8115473；训练含验证 437.3415684700012 秒；峰值CUDA 17132978688 bytes；更新 1240。
- **原记录差值**：-0.000908742；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.00045972048520087627,"AUC":-0.0009087419151136977,"UAUC":-0.0020497466189371982,"gAUC":-0.002711624545306379,"NDCG(10)":-0.00017531368269663172}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.66542387008667,"mean_ms":12.689560723304748,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000908742，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 12](#phase-12)。
- **后续假设**：在两轮之间更新历史源，分开Query写入与历史self-attention；见K13。
- **原判读**：本轮 AUC 首选，高于 Cross8/Cross11，低于父模型；保留来源后读方向
- **证据**：`qformer-loop/runs/r12_history_afc-s20262027-20260907T035559125014Z/manifest.json`（SHA256前缀 `2a8b9a45845c`）；`qformer-loop/runs/r12_history_afc-s20262027-20260907T035559125014Z/result.json`（SHA256前缀 `777851a542b8`）；`qformer-loop/runs/r12_history_afc-s20262027-20260907T035559125014Z/inference-all_queries.json`（SHA256前缀 `0ef975797de0`）

<a id="k169"></a>
### K169 · `diagnostics-round12-r12_both_caf`

- **方法**：Query Zero/Shuffle, extra attention off and inference-depth interventions
- **原假设**：Measure Query output and depth reliance with reordered complete shared blocks
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"training_seed":20262027,"new_training_seed_values":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "rows": 57408,
  "users": 7176,
  "positives": 4472,
  "precision": "bf16",
  "scores": {
    "original": {
      "AUC": 0.8605587504592578,
      "logloss": 0.20308107137680054
    },
    "all_queries": {
      "AUC": 0.8605587504592578,
      "logloss": 0.20308107137680054
    },
    "q3_half": {
      "AUC": 0.8605998796298524,
      "logloss": 0.20371922850608826
    },
    "q0_zero": {
      "AUC": 0.8605649917522844,
      "logloss": 0.20514312386512756
    },
    "q0_shuffle": {
      "AUC": 0.8585968807846543,
      "logloss": 0.2056359350681305
    },
    "q1_zero": {
      "AUC": 0.8601878571329121,
      "logloss": 0.20747138559818268
    },
    "q1_shuffle": {
      "AUC": 0.8578497652716225,
      "logloss": 0.20537109673023224
    },
    "q2_zero": {
      "AUC": 0.8600046609258205,
      "logloss": 0.2047995775938034
    },
    "q2_shuffle": {
      "AUC": 0.8579754739952629,
      "logloss": 0.20541414618492126
    },
    "q3_zero": {
      "AUC": 0.8601893947509572,
      "logloss": 0.20555417239665985
    },
    "q3_shuffle": {
      "AUC": 0.8541818534610126,
      "logloss": 0.2092607021331787
    },
    "q4_zero": {
      "AUC": 0.8597290302185538,
      "logloss": 0.2051766961812973
    },
    "q4_shuffle": {
      "AUC": 0.8578516344913614,
      "logloss": 0.20610471069812775
    },
    "q5_zero": {
      "AUC": 0.8599133331727,
      "logloss": 0.20530641078948975
    },
    "q5_shuffle": {
      "AUC": 0.8559713958604753,
      "logloss": 0.20833425223827362
    },
    "q6_zero": {
      "AUC": 0.8598455660367411,
      "logloss": 0.20385202765464783
    },
    "q6_shuffle": {
      "AUC": 0.8589481504719101,
      "logloss": 0.2048431634902954
    },
    "q7_zero": {
      "AUC": 0.8606013200907133,
      "logloss": 0.20507125556468964
    },
    "q7_shuffle": {
      "AUC": 0.8554210912329954,
      "logloss": 0.2084410935640335
    },
    "inference_placement_steps1": {
      "AUC": 0.8220335613694115,
      "logloss": 0.24324816465377808
    },
    "inference_placement_steps3": {
      "AUC": 0.8525547388644688,
      "logloss": 0.2136959284543991
    },
    "extra_attention_off": {
      "AUC": 0.8571424145043816,
      "logloss": 0.20798076689243317
    }
  },
  "checkpoint_unchanged": true,
  "note": "subset diagnostics; within-block shuffle; blockwise loss means in toolkit differ from pooled full-validation metrics"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 12](#phase-12)。
- **后续假设**：在两轮之间更新历史源，分开Query写入与历史self-attention；见K13。
- **证据**：`qformer-loop/reports/round12/r12_both_caf/summary.json`（SHA256前缀 `01a6097dcc9e`）；`qformer-loop/reports/round12/r12_both_caf/toolkit-hooks/hook_report.json`（SHA256前缀 `4abb5e018677`）；`qformer-loop/reports/round12/r12_both_caf/toolkit-perturbations/perturbation_report.json`（SHA256前缀 `0d96eddff661`）

<a id="k170"></a>
### K170 · `diagnostics-round12-r12_history_afc`

- **方法**：Query Zero/Shuffle, extra attention off and inference-depth interventions
- **原假设**：Measure Query output and depth reliance with reordered complete shared blocks
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"training_seed":20262027,"new_training_seed_values":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "rows": 57408,
  "users": 7176,
  "positives": 4472,
  "precision": "bf16",
  "scores": {
    "original": {
      "AUC": 0.8608843600893291,
      "logloss": 0.20276586711406708
    },
    "all_queries": {
      "AUC": 0.8608843600893291,
      "logloss": 0.20276586711406708
    },
    "q3_half": {
      "AUC": 0.8605762387524084,
      "logloss": 0.20293626189231873
    },
    "q0_zero": {
      "AUC": 0.8601457162603345,
      "logloss": 0.2034449726343155
    },
    "q0_shuffle": {
      "AUC": 0.8589637040698284,
      "logloss": 0.20496496558189392
    },
    "q1_zero": {
      "AUC": 0.8603168459675747,
      "logloss": 0.205058291554451
    },
    "q1_shuffle": {
      "AUC": 0.857439774627099,
      "logloss": 0.2066207230091095
    },
    "q2_zero": {
      "AUC": 0.859660511170474,
      "logloss": 0.20474675297737122
    },
    "q2_shuffle": {
      "AUC": 0.8580201156092767,
      "logloss": 0.20572419464588165
    },
    "q3_zero": {
      "AUC": 0.8597572607168936,
      "logloss": 0.2046816051006317
    },
    "q3_shuffle": {
      "AUC": 0.8590871950751345,
      "logloss": 0.20486269891262054
    },
    "q4_zero": {
      "AUC": 0.8598106675986096,
      "logloss": 0.20488913357257843
    },
    "q4_shuffle": {
      "AUC": 0.8557038524327347,
      "logloss": 0.20676633715629578
    },
    "q5_zero": {
      "AUC": 0.8604189159258839,
      "logloss": 0.20340555906295776
    },
    "q5_shuffle": {
      "AUC": 0.8585799648740451,
      "logloss": 0.20513010025024414
    },
    "q6_zero": {
      "AUC": 0.8602157961596992,
      "logloss": 0.2058844119310379
    },
    "q6_shuffle": {
      "AUC": 0.8533509504371972,
      "logloss": 0.20955641567707062
    },
    "q7_zero": {
      "AUC": 0.8607931358297312,
      "logloss": 0.2074599713087082
    },
    "q7_shuffle": {
      "AUC": 0.8570560100859634,
      "logloss": 0.20688827335834503
    },
    "inference_placement_steps1": {
      "AUC": 0.7950288191863912,
      "logloss": 0.2302149087190628
    },
    "inference_placement_steps3": {
      "AUC": 0.8447034583631958,
      "logloss": 0.21072815358638763
    },
    "extra_attention_off": {
      "AUC": 0.8595770003464541,
      "logloss": 0.20380772650241852
    }
  },
  "checkpoint_unchanged": true,
  "note": "subset diagnostics; within-block shuffle; blockwise loss means in toolkit differ from pooled full-validation metrics"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 12](#phase-12)。
- **后续假设**：在两轮之间更新历史源，分开Query写入与历史self-attention；见K13。
- **证据**：`qformer-loop/reports/round12/r12_history_afc/summary.json`（SHA256前缀 `45fa76cd1db1`）；`qformer-loop/reports/round12/r12_history_afc/toolkit-hooks/hook_report.json`（SHA256前缀 `5b28ce7f17cd`）；`qformer-loop/reports/round12/r12_history_afc/toolkit-perturbations/perturbation_report.json`（SHA256前缀 `ecef586703ee`）

<a id="phase-13"></a>
## 阶段 13：历史源写入

**方法**：Query或历史self-attention更新历史；不同位置；zero/warm初始化。

**归因与边界**：8组未提高父模型AUC，部分UAUC/NDCG有取舍，新增writer提高开销。

**历史后续假设**：在固定循环图上比较参数预算接近的门控FFN；见K14。

<a id="k171"></a>
### K171 · `r13_query_between_zero-s20262027-20260907T042433943622Z`

- **方法**：{"history_write":"query_between","writer_initial_scale":0.0}；设置 {"model":"QFormerCross11SourceUpdate","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Update history between shared Query cycles using Query feedback or source self-attention
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.188660424；AUC=0.870694573；UAUC=0.791352894；gAUC=0.803681771；NDCG(10)=0.408685037。
- **开销**：参数 8379410；训练含验证 499.36246854811907 秒；峰值CUDA 20862465536 bytes；更新 1240。
- **原记录差值**：-0.000539486；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.0005935657240170589,"AUC":-0.0005394859431416821,"UAUC":0.0005158812160864779,"gAUC":-0.00010889161095550826,"NDCG(10)":-0.001171120422328864}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":15.159631729125977,"mean_ms":15.15866723060608,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000539486，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 13](#phase-13)。
- **后续假设**：在固定循环图上比较参数预算接近的门控FFN；见K14。
- **原判读**：本轮 AUC 最佳，UAUC 高于父模型；保留 Query 反馈候选，参数及计算额外增加
- **证据**：`qformer-loop/runs/r13_query_between_zero-s20262027-20260907T042433943622Z/manifest.json`（SHA256前缀 `195e00fc8ee1`）；`qformer-loop/runs/r13_query_between_zero-s20262027-20260907T042433943622Z/result.json`（SHA256前缀 `9b5d7ba3c9ba`）；`qformer-loop/runs/r13_query_between_zero-s20262027-20260907T042433943622Z/inference-all_queries.json`（SHA256前缀 `749ab2e60535`）

<a id="k172"></a>
### K172 · `r13_query_between_warm-s20262027-20260907T042433862402Z`

- **方法**：{"history_write":"query_between","writer_initial_scale":0.1}；设置 {"model":"QFormerCross11SourceUpdate","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Update history between shared Query cycles using Query feedback or source self-attention
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.188862333；AUC=0.869828753；UAUC=0.790010940；gAUC=0.801819676；NDCG(10)=0.409167728。
- **开销**：参数 8379410；训练含验证 506.4821131154895 秒；峰值CUDA 20862465536 bytes；更新 1240。
- **原记录差值**：-0.001405305；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.0007954747629847492,"AUC":-0.0014053054519399533,"UAUC":-0.0008260725009380243,"gAUC":-0.0019709863515813275,"NDCG(10)":-0.0006884293920947915}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":15.163536071777344,"mean_ms":15.16804325580597,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.001405305，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 13](#phase-13)。
- **后续假设**：在固定循环图上比较参数预算接近的门控FFN；见K14。
- **原判读**：保留历史源更新结构对照；AUC 未超过父模型
- **证据**：`qformer-loop/runs/r13_query_between_warm-s20262027-20260907T042433862402Z/manifest.json`（SHA256前缀 `34ad53c70e15`）；`qformer-loop/runs/r13_query_between_warm-s20262027-20260907T042433862402Z/result.json`（SHA256前缀 `07305ab3fb81`）；`qformer-loop/runs/r13_query_between_warm-s20262027-20260907T042433862402Z/inference-all_queries.json`（SHA256前缀 `9df2363f7c45`）

<a id="k173"></a>
### K173 · `r13_query_after_field_zero-s20262027-20260907T042433963604Z`

- **方法**：{"history_write":"query_after_field","writer_initial_scale":0.0}；设置 {"model":"QFormerCross11SourceUpdate","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Update history between shared Query cycles using Query feedback or source self-attention
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.189176382；AUC=0.869506924；UAUC=0.790803530；gAUC=0.803113855；NDCG(10)=0.409071997。
- **开销**：参数 8379410；训练含验证 499.53887467086315 秒；峰值CUDA 20795356672 bytes；更新 1240。
- **原记录差值**：-0.001727134；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.0011095231904677816,"AUC":-0.001727134323855628,"UAUC":-3.34826179859693e-05,"gAUC":-0.0006768072975260608,"NDCG(10)":-0.0007841595184854833}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":15.152624130249023,"mean_ms":15.15570638179779,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.001727134，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 13](#phase-13)。
- **后续假设**：在固定循环图上比较参数预算接近的门控FFN；见K14。
- **原判读**：保留历史源更新结构对照；AUC 未超过父模型
- **证据**：`qformer-loop/runs/r13_query_after_field_zero-s20262027-20260907T042433963604Z/manifest.json`（SHA256前缀 `aa251c192c1a`）；`qformer-loop/runs/r13_query_after_field_zero-s20262027-20260907T042433963604Z/result.json`（SHA256前缀 `a4f700bb27b6`）；`qformer-loop/runs/r13_query_after_field_zero-s20262027-20260907T042433963604Z/inference-all_queries.json`（SHA256前缀 `c34efdcb99b4`）

<a id="k174"></a>
### K174 · `r13_query_after_field_warm-s20262027-20260907T042433869217Z`

- **方法**：{"history_write":"query_after_field","writer_initial_scale":0.1}；设置 {"model":"QFormerCross11SourceUpdate","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Update history between shared Query cycles using Query feedback or source self-attention
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.188634341；AUC=0.870310892；UAUC=0.789467623；gAUC=0.801690222；NDCG(10)=0.405682373。
- **开销**：参数 8379410；训练含验证 501.4302778169513 秒；峰值CUDA 20795356672 bytes；更新 1240。
- **原记录差值**：-0.000923166；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.0005674821241912487,"AUC":-0.000923166433868805,"UAUC":-0.0013693896125533955,"gAUC":-0.0021004402721988003,"NDCG(10)":-0.004173783906237127}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":15.175136089324951,"mean_ms":15.256152009963989,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000923166，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 13](#phase-13)。
- **后续假设**：在固定循环图上比较参数预算接近的门控FFN；见K14。
- **原判读**：AUC 高于 Cross8/Cross11；保留回读后写历史位置对照
- **证据**：`qformer-loop/runs/r13_query_after_field_warm-s20262027-20260907T042433869217Z/manifest.json`（SHA256前缀 `c1744690f562`）；`qformer-loop/runs/r13_query_after_field_warm-s20262027-20260907T042433869217Z/result.json`（SHA256前缀 `559f4fedeaf3`）；`qformer-loop/runs/r13_query_after_field_warm-s20262027-20260907T042433869217Z/inference-all_queries.json`（SHA256前缀 `91d8160df430`）

<a id="k175"></a>
### K175 · `r13_self_between_zero-s20262027-20260907T042433943647Z`

- **方法**：{"history_write":"self_between","writer_initial_scale":0.0}；设置 {"model":"QFormerCross11SourceUpdate","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Update history between shared Query cycles using Query feedback or source self-attention
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.189123485；AUC=0.870213613；UAUC=0.790486819；gAUC=0.803156088；NDCG(10)=0.408626943。
- **开销**：参数 8379410；训练含验证 541.0143840461969 秒；峰值CUDA 22719771136 bytes；更新 1240。
- **原记录差值**：-0.001020446；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.0010566265023083388,"AUC":-0.001020445633050926,"UAUC":-0.0003501933328288187,"gAUC":-0.0006345737745844238,"NDCG(10)":-0.0012292136019929623}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":16.776592254638672,"mean_ms":16.806350421905517,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.001020446，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 13](#phase-13)。
- **后续假设**：在固定循环图上比较参数预算接近的门控FFN；见K14。
- **原判读**：历史 self-attention 本轮最高 AUC，略高 warm；保留零初始对照
- **证据**：`qformer-loop/runs/r13_self_between_zero-s20262027-20260907T042433943647Z/manifest.json`（SHA256前缀 `3e28753f9459`）；`qformer-loop/runs/r13_self_between_zero-s20262027-20260907T042433943647Z/result.json`（SHA256前缀 `f73910d3e46a`）；`qformer-loop/runs/r13_self_between_zero-s20262027-20260907T042433943647Z/inference-all_queries.json`（SHA256前缀 `1466c61ae3bf`）

<a id="k176"></a>
### K176 · `r13_self_between_warm-s20262027-20260907T042433953319Z`

- **方法**：{"history_write":"self_between","writer_initial_scale":0.1}；设置 {"model":"QFormerCross11SourceUpdate","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Update history between shared Query cycles using Query feedback or source self-attention
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.188870128；AUC=0.870188253；UAUC=0.790412662；gAUC=0.802867445；NDCG(10)=0.410352152。
- **开销**：参数 8379410；训练含验证 546.8407229185104 秒；峰值CUDA 22719771136 bytes；更新 1240。
- **原记录差值**：-0.001045805；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.0008032693686576553,"AUC":-0.0010458051416227798,"UAUC":-0.000424350461173284,"gAUC":-0.0009232172890241142,"NDCG(10)":0.0004959948505294798}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":16.772480010986328,"mean_ms":16.785963106155396,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.001045805，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 13](#phase-13)。
- **后续假设**：在固定循环图上比较参数预算接近的门控FFN；见K14。
- **原判读**：AUC 高于 Cross11，NDCG 高于父模型；保留历史 self-attention 排序取舍
- **证据**：`qformer-loop/runs/r13_self_between_warm-s20262027-20260907T042433953319Z/manifest.json`（SHA256前缀 `80007d184376`）；`qformer-loop/runs/r13_self_between_warm-s20262027-20260907T042433953319Z/result.json`（SHA256前缀 `6ef7adcd6a4e`）；`qformer-loop/runs/r13_self_between_warm-s20262027-20260907T042433953319Z/inference-all_queries.json`（SHA256前缀 `973716fd54e2`）

<a id="k177"></a>
### K177 · `r13_self_before_zero-s20262027-20260907T042433845232Z`

- **方法**：{"history_write":"self_before","writer_initial_scale":0.0}；设置 {"model":"QFormerCross11SourceUpdate","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Update history between shared Query cycles using Query feedback or source self-attention
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.189064968；AUC=0.869720377；UAUC=0.790019009；gAUC=0.802385264；NDCG(10)=0.409197511。
- **开销**：参数 8379410；训练含验证 537.7256479263306 秒；峰值CUDA 22718788096 bytes；更新 1240。
- **原记录差值**：-0.001513681；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.0009981090640555357,"AUC":-0.0015136813457559128,"UAUC":-0.0008180037418376029,"gAUC":-0.0014053982215218586,"NDCG(10)":-0.0006586456436903365}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":16.75881576538086,"mean_ms":16.765546464920043,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.001513681，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 13](#phase-13)。
- **后续假设**：在固定循环图上比较参数预算接近的门控FFN；见K14。
- **原判读**：保留历史源更新结构对照；AUC 未超过父模型
- **证据**：`qformer-loop/runs/r13_self_before_zero-s20262027-20260907T042433845232Z/manifest.json`（SHA256前缀 `c203c49a6de1`）；`qformer-loop/runs/r13_self_before_zero-s20262027-20260907T042433845232Z/result.json`（SHA256前缀 `239827b820d3`）；`qformer-loop/runs/r13_self_before_zero-s20262027-20260907T042433845232Z/inference-all_queries.json`（SHA256前缀 `920facdf9e4f`）

<a id="k178"></a>
### K178 · `r13_self_before_warm-s20262027-20260907T043325955638Z`

- **方法**：{"history_write":"self_before","writer_initial_scale":0.1}；设置 {"model":"QFormerCross11SourceUpdate","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Update history between shared Query cycles using Query feedback or source self-attention
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.189062373；AUC=0.869197604；UAUC=0.789273861；gAUC=0.801609622；NDCG(10)=0.406719243。
- **开销**：参数 8379410；训练含验证 535.527639657259 秒；峰值CUDA 22718788096 bytes；更新 1240。
- **原记录差值**：-0.002036454；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.0009955148506462508,"AUC":-0.002036454487815753,"UAUC":-0.0015631522111652085,"gAUC":-0.0021810400887697368,"NDCG(10)":-0.003136913823609755}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":16.77067279815674,"mean_ms":16.808514404296876,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.002036454，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 13](#phase-13)。
- **后续假设**：在固定循环图上比较参数预算接近的门控FFN；见K14。
- **原判读**：保留历史源更新结构对照；AUC 未超过父模型
- **证据**：`qformer-loop/runs/r13_self_before_warm-s20262027-20260907T043325955638Z/manifest.json`（SHA256前缀 `8f650bca5689`）；`qformer-loop/runs/r13_self_before_warm-s20262027-20260907T043325955638Z/result.json`（SHA256前缀 `b2aa25a8ddeb`）；`qformer-loop/runs/r13_self_before_warm-s20262027-20260907T043325955638Z/inference-all_queries.json`（SHA256前缀 `8ddc983a3c10`）

<a id="k179"></a>
### K179 · `diagnostics-round13-r13_query_between_zero`

- **方法**：Query Zero/Shuffle, extra attention off and inference-depth interventions
- **原假设**：Measure Query output and depth reliance after history source updates
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"training_seed":20262027,"new_training_seed_values":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "rows": 57408,
  "users": 7176,
  "positives": 4472,
  "precision": "bf16",
  "scores": {
    "original": {
      "AUC": 0.8618545041428499,
      "logloss": 0.2026730477809906
    },
    "all_queries": {
      "AUC": 0.8618545041428499,
      "logloss": 0.2026730477809906
    },
    "q3_half": {
      "AUC": 0.8619405452778839,
      "logloss": 0.20362365245819092
    },
    "q0_zero": {
      "AUC": 0.8610204688559013,
      "logloss": 0.20565569400787354
    },
    "q0_shuffle": {
      "AUC": 0.8561044779695492,
      "logloss": 0.208650141954422
    },
    "q1_zero": {
      "AUC": 0.8608367889750015,
      "logloss": 0.2041623741388321
    },
    "q1_shuffle": {
      "AUC": 0.8596773003543213,
      "logloss": 0.20512118935585022
    },
    "q2_zero": {
      "AUC": 0.8592762481707414,
      "logloss": 0.20763452351093292
    },
    "q2_shuffle": {
      "AUC": 0.8556016789809032,
      "logloss": 0.20881754159927368
    },
    "q3_zero": {
      "AUC": 0.8615425936757466,
      "logloss": 0.2095019668340683
    },
    "q3_shuffle": {
      "AUC": 0.8585674569426395,
      "logloss": 0.2071591019630432
    },
    "q4_zero": {
      "AUC": 0.8595965373889232,
      "logloss": 0.20552773773670197
    },
    "q4_shuffle": {
      "AUC": 0.8589253671122222,
      "logloss": 0.20579314231872559
    },
    "q5_zero": {
      "AUC": 0.8615558155012446,
      "logloss": 0.2045442909002304
    },
    "q5_shuffle": {
      "AUC": 0.8576945313245575,
      "logloss": 0.20655511319637299
    },
    "q6_zero": {
      "AUC": 0.8612620818760319,
      "logloss": 0.20520880818367004
    },
    "q6_shuffle": {
      "AUC": 0.8533339774995452,
      "logloss": 0.20948658883571625
    },
    "q7_zero": {
      "AUC": 0.8614384982013586,
      "logloss": 0.20414645969867706
    },
    "q7_shuffle": {
      "AUC": 0.8591945136335016,
      "logloss": 0.20627178251743317
    },
    "inference_placement_steps1": {
      "AUC": 0.8182778532581146,
      "logloss": 0.2504047453403473
    },
    "inference_placement_steps3": {
      "AUC": 0.819705261262596,
      "logloss": 0.2308981716632843
    },
    "extra_attention_off": {
      "AUC": 0.8581254973602984,
      "logloss": 0.20574747025966644
    }
  },
  "checkpoint_unchanged": true,
  "note": "subset diagnostics; within-block shuffle; blockwise loss means in toolkit differ from pooled full-validation metrics"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 13](#phase-13)。
- **后续假设**：在固定循环图上比较参数预算接近的门控FFN；见K14。
- **证据**：`qformer-loop/reports/round13/r13_query_between_zero/summary.json`（SHA256前缀 `fc005eb5559c`）；`qformer-loop/reports/round13/r13_query_between_zero/toolkit-hooks/hook_report.json`（SHA256前缀 `3a50f0616466`）；`qformer-loop/reports/round13/r13_query_between_zero/toolkit-perturbations/perturbation_report.json`（SHA256前缀 `86b75ada5e4b`）

<a id="k180"></a>
### K180 · `diagnostics-round13-r13_self_between_zero`

- **方法**：Query Zero/Shuffle, extra attention off and inference-depth interventions
- **原假设**：Measure Query output and depth reliance after history source updates
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"training_seed":20262027,"new_training_seed_values":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "rows": 57408,
  "users": 7176,
  "positives": 4472,
  "precision": "bf16",
  "scores": {
    "original": {
      "AUC": 0.862037922121775,
      "logloss": 0.20293764770030975
    },
    "all_queries": {
      "AUC": 0.862037922121775,
      "logloss": 0.20293764770030975
    },
    "q3_half": {
      "AUC": 0.8618285209324226,
      "logloss": 0.20335787534713745
    },
    "q0_zero": {
      "AUC": 0.861132294662769,
      "logloss": 0.20598571002483368
    },
    "q0_shuffle": {
      "AUC": 0.8580003356738471,
      "logloss": 0.2072789967060089
    },
    "q1_zero": {
      "AUC": 0.860680927730465,
      "logloss": 0.20377136766910553
    },
    "q1_shuffle": {
      "AUC": 0.8601522279037866,
      "logloss": 0.20459569990634918
    },
    "q2_zero": {
      "AUC": 0.8615662620106556,
      "logloss": 0.20738103985786438
    },
    "q2_shuffle": {
      "AUC": 0.8569349733556137,
      "logloss": 0.2098129838705063
    },
    "q3_zero": {
      "AUC": 0.8614617863559819,
      "logloss": 0.20570366084575653
    },
    "q3_shuffle": {
      "AUC": 0.858588164095544,
      "logloss": 0.20734238624572754
    },
    "q4_zero": {
      "AUC": 0.8616361285866377,
      "logloss": 0.20529763400554657
    },
    "q4_shuffle": {
      "AUC": 0.8586504524111608,
      "logloss": 0.20780137181282043
    },
    "q5_zero": {
      "AUC": 0.8619239060540382,
      "logloss": 0.20422467589378357
    },
    "q5_shuffle": {
      "AUC": 0.8586025623678156,
      "logloss": 0.20679527521133423
    },
    "q6_zero": {
      "AUC": 0.8609754618463906,
      "logloss": 0.20848512649536133
    },
    "q6_shuffle": {
      "AUC": 0.8585287820470014,
      "logloss": 0.2071087658405304
    },
    "q7_zero": {
      "AUC": 0.8617335645696845,
      "logloss": 0.20447267591953278
    },
    "q7_shuffle": {
      "AUC": 0.860100595196738,
      "logloss": 0.2047157883644104
    },
    "inference_placement_steps1": {
      "AUC": 0.8197849745079825,
      "logloss": 0.2523525357246399
    },
    "inference_placement_steps3": {
      "AUC": 0.8075147888441518,
      "logloss": 0.23194953799247742
    },
    "extra_attention_off": {
      "AUC": 0.8605150318384939,
      "logloss": 0.20434311032295227
    }
  },
  "checkpoint_unchanged": true,
  "note": "subset diagnostics; within-block shuffle; blockwise loss means in toolkit differ from pooled full-validation metrics"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 13](#phase-13)。
- **后续假设**：在固定循环图上比较参数预算接近的门控FFN；见K14。
- **证据**：`qformer-loop/reports/round13/r13_self_between_zero/summary.json`（SHA256前缀 `f909f5581d76`）；`qformer-loop/reports/round13/r13_self_between_zero/toolkit-hooks/hook_report.json`（SHA256前缀 `26972ed9a646`）；`qformer-loop/reports/round13/r13_self_between_zero/toolkit-perturbations/perturbation_report.json`（SHA256前缀 `909bc809262c`）

<a id="k181"></a>
### K181 · `probe-round13-r13_query_between_zero-writer-probe`

- **方法**：Bypass writer; deterministically roll feedback Query within a batch when applicable
- **原假设**：Measure reliance on learned history updates and matching Query feedback
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"training_seed":20262027,"new_training_seed_values":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "rows": 57408,
  "scores": {
    "original": {
      "AUC": 0.8618545041428499,
      "logloss": 0.2026730654581627
    },
    "writer_off": {
      "AUC": 0.8618688559486422,
      "logloss": 0.2026655711245802
    },
    "query_roll": {
      "AUC": 0.8618534692076272,
      "logloss": 0.2026687526995918
    }
  },
  "checkpoint_unchanged": true,
  "note": "fixed-checkpoint subset only; Query roll is deterministic within batch; no retraining or parameter edits"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 13](#phase-13)。
- **后续假设**：在固定循环图上比较参数预算接近的门控FFN；见K14。
- **证据**：`qformer-loop/reports/round13/r13_query_between_zero-writer-probe/result.json`（SHA256前缀 `4a1e25f73718`）

<a id="k182"></a>
### K182 · `probe-round13-r13_self_between_zero-writer-probe`

- **方法**：Bypass writer; deterministically roll feedback Query within a batch when applicable
- **原假设**：Measure reliance on learned history updates and matching Query feedback
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"training_seed":20262027,"new_training_seed_values":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "rows": 57408,
  "scores": {
    "original": {
      "AUC": 0.862037922121775,
      "logloss": 0.20293763862381028
    },
    "writer_off": {
      "AUC": 0.8620517480115051,
      "logloss": 0.20293325915825533
    }
  },
  "checkpoint_unchanged": true,
  "note": "fixed-checkpoint subset only; Query roll is deterministic within batch; no retraining or parameter edits"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 13](#phase-13)。
- **后续假设**：在固定循环图上比较参数预算接近的门控FFN；见K14。
- **证据**：`qformer-loop/reports/round13/r13_self_between_zero-writer-probe/result.json`（SHA256前缀 `13b444770d85`）

<a id="phase-14"></a>
## 阶段 14：门控FFN与激活

**方法**：字段/历史/双位置SwiGLU、GEGLU和密集SiLU。

**归因与边界**：8组均未刷新r7；同初值SwiGLU优于GEGLU不等于优于原GELU。更高有效秩没有转化为更高AUC。

**历史后续假设**：保留GELU，单独改变FFN宽度以分离容量因素；见K15。

<a id="k183"></a>
### K183 · `r14_swiglu_field-s20262027-20260907T045344463996Z`

- **方法**：{"gated_ffn_kind":"swiglu","gated_ffn_scope":"field","parameter_delta":-1712}；设置 {"model":"QFormerCross11GatedFFN","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Compare budget-matched shared gated FFNs and dense activation controls
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.189253719；AUC=0.869148417；UAUC=0.788315241；gAUC=0.800092839；NDCG(10)=0.406227180。
- **开销**：参数 8113761；训练含验证 440.0315157994628 秒；峰值CUDA 17144627712 bytes；更新 1240。
- **原记录差值**：-0.002085641；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.0011868606696127382,"AUC":-0.002085641302552177,"UAUC":-0.002521771953912455,"gAUC":-0.003697822999467304,"NDCG(10)":-0.0036289767378038795}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.633488178253174,"mean_ms":12.651758432388306,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.002085641，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 14](#phase-14)。
- **后续假设**：保留GELU，单独改变FFN宽度以分离容量因素；见K15。
- **原判读**：保留门控/激活结构负对照，AUC 未超过父模型
- **证据**：`qformer-loop/runs/r14_swiglu_field-s20262027-20260907T045344463996Z/manifest.json`（SHA256前缀 `f607e135c069`）；`qformer-loop/runs/r14_swiglu_field-s20262027-20260907T045344463996Z/result.json`（SHA256前缀 `ce1771103e3a`）；`qformer-loop/runs/r14_swiglu_field-s20262027-20260907T045344463996Z/inference-all_queries.json`（SHA256前缀 `38c42476e5d0`）

<a id="k184"></a>
### K184 · `r14_swiglu_history-s20262027-20260907T045344433620Z`

- **方法**：{"gated_ffn_kind":"swiglu","gated_ffn_scope":"history","parameter_delta":-1712}；设置 {"model":"QFormerCross11GatedFFN","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Compare budget-matched shared gated FFNs and dense activation controls
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.188824330；AUC=0.870100180；UAUC=0.788932535；gAUC=0.801469771；NDCG(10)=0.406412742。
- **开销**：参数 8113761；训练含验证 439.36368860304356 秒；峰值CUDA 17020633600 bytes；更新 1240。
- **原记录差值**：-0.001133878；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.0007574715928445053,"AUC":-0.0011338782983614237,"UAUC":-0.0019044775982777207,"gAUC":-0.0023208910318247566,"NDCG(10)":-0.00344341525844466}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.628479957580566,"mean_ms":12.634674453735352,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.001133878，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 14](#phase-14)。
- **后续假设**：保留GELU，单独改变FFN宽度以分离容量因素；见K15。
- **原判读**：AUC 高于 Cross11，保留单历史门控候选
- **证据**：`qformer-loop/runs/r14_swiglu_history-s20262027-20260907T045344433620Z/manifest.json`（SHA256前缀 `b5d7261b154d`）；`qformer-loop/runs/r14_swiglu_history-s20262027-20260907T045344433620Z/result.json`（SHA256前缀 `2d30df2e91c0`）；`qformer-loop/runs/r14_swiglu_history-s20262027-20260907T045344433620Z/inference-all_queries.json`（SHA256前缀 `fa36286ca620`）

<a id="k185"></a>
### K185 · `r14_swiglu_both-s20262027-20260907T045344417858Z`

- **方法**：{"gated_ffn_kind":"swiglu","gated_ffn_scope":"both","parameter_delta":-3424}；设置 {"model":"QFormerCross11GatedFFN","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Compare budget-matched shared gated FFNs and dense activation controls
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.188465469；AUC=0.870200706；UAUC=0.791640004；gAUC=0.802988719；NDCG(10)=0.410436190。
- **开销**：参数 8112049；训练含验证 440.77670258283615 秒；峰值CUDA 17268081152 bytes；更新 1240。
- **原记录差值**：-0.001033352；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.00039861082876799236,"AUC":-0.0010333520043025457,"UAUC":0.0008029912293283648,"gAUC":-0.0008019426813613562,"NDCG(10)":0.0005800326314988613}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.698464393615723,"mean_ms":12.720164012908935,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.001033352，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 14](#phase-14)。
- **后续假设**：保留GELU，单独改变FFN宽度以分离容量因素；见K15。
- **原判读**：本轮 AUC 最佳；UAUC、NDCG 高于父模型，保留参数略少的排序候选
- **证据**：`qformer-loop/runs/r14_swiglu_both-s20262027-20260907T045344417858Z/manifest.json`（SHA256前缀 `0239013389b2`）；`qformer-loop/runs/r14_swiglu_both-s20262027-20260907T045344417858Z/result.json`（SHA256前缀 `5b0efad8c0cd`）；`qformer-loop/runs/r14_swiglu_both-s20262027-20260907T045344417858Z/inference-all_queries.json`（SHA256前缀 `c5a0ff661097`）

<a id="k186"></a>
### K186 · `r14_geglu_field-s20262027-20260907T045344405903Z`

- **方法**：{"gated_ffn_kind":"geglu","gated_ffn_scope":"field","parameter_delta":-1712}；设置 {"model":"QFormerCross11GatedFFN","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Compare budget-matched shared gated FFNs and dense activation controls
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.189459166；AUC=0.868587326；UAUC=0.790206266；gAUC=0.802703726；NDCG(10)=0.409388938。
- **开销**：参数 8113761；训练含验证 440.70128882676363 秒；峰值CUDA 17144627712 bytes；更新 1240。
- **原记录差值**：-0.002646733；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.0013923074726630014,"AUC":-0.0026467325127986774,"UAUC":-0.0006307471554807265,"gAUC":-0.0010869361844226821,"NDCG(10)":-0.0004672193653053047}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.640223979949951,"mean_ms":12.648621606826783,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.002646733，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 14](#phase-14)。
- **后续假设**：保留GELU，单独改变FFN宽度以分离容量因素；见K15。
- **原判读**：保留门控/激活结构负对照，AUC 未超过父模型
- **证据**：`qformer-loop/runs/r14_geglu_field-s20262027-20260907T045344405903Z/manifest.json`（SHA256前缀 `bb191d88acef`）；`qformer-loop/runs/r14_geglu_field-s20262027-20260907T045344405903Z/result.json`（SHA256前缀 `cd24dfa5c92c`）；`qformer-loop/runs/r14_geglu_field-s20262027-20260907T045344405903Z/inference-all_queries.json`（SHA256前缀 `316bf891bd99`）

<a id="k187"></a>
### K187 · `r14_geglu_history-s20262027-20260907T045344436148Z`

- **方法**：{"gated_ffn_kind":"geglu","gated_ffn_scope":"history","parameter_delta":-1712}；设置 {"model":"QFormerCross11GatedFFN","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Compare budget-matched shared gated FFNs and dense activation controls
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.189365647；AUC=0.868950584；UAUC=0.788384189；gAUC=0.800759374；NDCG(10)=0.405641259。
- **开销**：参数 8113761；训练含验证 442.60458716005087 秒；峰值CUDA 17020633600 bytes；更新 1240。
- **原记录差值**：-0.002283474；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.0012987886378214297,"AUC":-0.0022834740245045726,"UAUC":-0.0024528240439332105,"gAUC":-0.0030312882186221968,"NDCG(10)":-0.004214897795457417}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.637664318084717,"mean_ms":12.660260820388794,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.002283474，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 14](#phase-14)。
- **后续假设**：保留GELU，单独改变FFN宽度以分离容量因素；见K15。
- **原判读**：保留门控/激活结构负对照，AUC 未超过父模型
- **证据**：`qformer-loop/runs/r14_geglu_history-s20262027-20260907T045344436148Z/manifest.json`（SHA256前缀 `15da04210675`）；`qformer-loop/runs/r14_geglu_history-s20262027-20260907T045344436148Z/result.json`（SHA256前缀 `54d9b5c0d6f3`）；`qformer-loop/runs/r14_geglu_history-s20262027-20260907T045344436148Z/inference-all_queries.json`（SHA256前缀 `5ba62a21a69d`）

<a id="k188"></a>
### K188 · `r14_geglu_both-s20262027-20260907T045344165025Z`

- **方法**：{"gated_ffn_kind":"geglu","gated_ffn_scope":"both","parameter_delta":-3424}；设置 {"model":"QFormerCross11GatedFFN","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Compare budget-matched shared gated FFNs and dense activation controls
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.189154740；AUC=0.868820903；UAUC=0.788935858；gAUC=0.800845920；NDCG(10)=0.410598896。
- **开销**：参数 8112049；训练含验证 443.0938801243901 秒；峰值CUDA 17268081152 bytes；更新 1240。
- **原记录差值**：-0.002413155；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.0010878817508100935,"AUC":-0.0024131552385273647,"UAUC":-0.0019011547783339688,"gAUC":-0.0029447417872582315,"NDCG(10)":0.0007427391127566296}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.702288150787354,"mean_ms":12.702475214004517,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.002413155，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 14](#phase-14)。
- **后续假设**：保留GELU，单独改变FFN宽度以分离容量因素；见K15。
- **原判读**：AUC 较低但 NDCG 本轮最高，保留负结果及排序对照
- **证据**：`qformer-loop/runs/r14_geglu_both-s20262027-20260907T045344165025Z/manifest.json`（SHA256前缀 `c2be5f721888`）；`qformer-loop/runs/r14_geglu_both-s20262027-20260907T045344165025Z/result.json`（SHA256前缀 `5d6edf70c7ba`）；`qformer-loop/runs/r14_geglu_both-s20262027-20260907T045344165025Z/inference-all_queries.json`（SHA256前缀 `2ef2638abd10`）

<a id="k189"></a>
### K189 · `r14_silu_history-s20262027-20260907T045344217569Z`

- **方法**：{"gated_ffn_kind":"silu","gated_ffn_scope":"history","parameter_delta":0}；设置 {"model":"QFormerCross11GatedFFN","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Compare budget-matched shared gated FFNs and dense activation controls
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.189437587；AUC=0.869456816；UAUC=0.789254349；gAUC=0.800993165；NDCG(10)=0.407890082。
- **开销**：参数 8115473；训练含验证 437.67081240564585 秒；峰值CUDA 16899675648 bytes；更新 1240。
- **原记录差值**：-0.001777242；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.001370728165769164,"AUC":-0.001777242442976279,"UAUC":-0.001582663624507985,"gAUC":-0.0027974975228305388,"NDCG(10)":-0.0019660753488324745}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.573040008544922,"mean_ms":12.582436013221741,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.001777242，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 14](#phase-14)。
- **后续假设**：保留GELU，单独改变FFN宽度以分离容量因素；见K15。
- **原判读**：保留门控/激活结构负对照，AUC 未超过父模型
- **证据**：`qformer-loop/runs/r14_silu_history-s20262027-20260907T045344217569Z/manifest.json`（SHA256前缀 `02757e2f685d`）；`qformer-loop/runs/r14_silu_history-s20262027-20260907T045344217569Z/result.json`（SHA256前缀 `32da4b67b800`）；`qformer-loop/runs/r14_silu_history-s20262027-20260907T045344217569Z/inference-all_queries.json`（SHA256前缀 `0d9cd599c7f3`）

<a id="k190"></a>
### K190 · `r14_silu_both-s20262027-20260907T050135045692Z`

- **方法**：{"gated_ffn_kind":"silu","gated_ffn_scope":"both","parameter_delta":0}；设置 {"model":"QFormerCross11GatedFFN","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Compare budget-matched shared gated FFNs and dense activation controls
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.189466552；AUC=0.869990565；UAUC=0.789255413；gAUC=0.800688777；NDCG(10)=0.410545535。
- **开销**：参数 8115473；训练含验证 438.63851922750473 秒；峰值CUDA 16901777920 bytes；更新 1240。
- **原记录差值**：-0.001243494；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.0013996930908728167,"AUC":-0.0012434938469988444,"UAUC":-0.0015815995322731036,"gAUC":-0.0031018846872179484,"NDCG(10)":0.0006893783397436404}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.572159767150879,"mean_ms":12.57473680973053,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.001243494，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 14](#phase-14)。
- **后续假设**：保留GELU，单独改变FFN宽度以分离容量因素；见K15。
- **原判读**：AUC 接近 Cross11，NDCG 高于父模型；保留纯激活对照
- **证据**：`qformer-loop/runs/r14_silu_both-s20262027-20260907T050135045692Z/manifest.json`（SHA256前缀 `a7106f15a421`）；`qformer-loop/runs/r14_silu_both-s20262027-20260907T050135045692Z/result.json`（SHA256前缀 `159a5c8bca23`）；`qformer-loop/runs/r14_silu_both-s20262027-20260907T050135045692Z/inference-all_queries.json`（SHA256前缀 `4eaabba95d90`）

<a id="k191"></a>
### K191 · `diagnostics-round14-r14_silu_both`

- **方法**：Query Zero/Shuffle, extra attention off and inference-depth interventions
- **原假设**：Measure Query and recurrence reliance with gated or SiLU FFNs
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"training_seed":20262027,"new_training_seed_values":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "rows": 57408,
  "users": 7176,
  "positives": 4472,
  "precision": "bf16",
  "scores": {
    "original": {
      "AUC": 0.86109313398121,
      "logloss": 0.20365484058856964
    },
    "all_queries": {
      "AUC": 0.86109313398121,
      "logloss": 0.20365484058856964
    },
    "q3_half": {
      "AUC": 0.8610091943983122,
      "logloss": 0.20451833307743073
    },
    "q0_zero": {
      "AUC": 0.8573024851895279,
      "logloss": 0.2063940167427063
    },
    "q0_shuffle": {
      "AUC": 0.8516472654189635,
      "logloss": 0.21159517765045166
    },
    "q1_zero": {
      "AUC": 0.8610134503054013,
      "logloss": 0.20399773120880127
    },
    "q1_shuffle": {
      "AUC": 0.8596813534985913,
      "logloss": 0.20489344000816345
    },
    "q2_zero": {
      "AUC": 0.8606685211804689,
      "logloss": 0.20418263971805573
    },
    "q2_shuffle": {
      "AUC": 0.8551694773592333,
      "logloss": 0.21034041047096252
    },
    "q3_zero": {
      "AUC": 0.8605283001304712,
      "logloss": 0.2063066065311432
    },
    "q3_shuffle": {
      "AUC": 0.8547856135488009,
      "logloss": 0.21040886640548706
    },
    "q4_zero": {
      "AUC": 0.8606933828590531,
      "logloss": 0.20407836139202118
    },
    "q4_shuffle": {
      "AUC": 0.8577744600054394,
      "logloss": 0.20701910555362701
    },
    "q5_zero": {
      "AUC": 0.8594897574192945,
      "logloss": 0.20521079003810883
    },
    "q5_shuffle": {
      "AUC": 0.8577046251111478,
      "logloss": 0.2087080180644989
    },
    "q6_zero": {
      "AUC": 0.859783098191545,
      "logloss": 0.20546536147594452
    },
    "q6_shuffle": {
      "AUC": 0.8547897891956074,
      "logloss": 0.20973767340183258
    },
    "q7_zero": {
      "AUC": 0.8603705358723925,
      "logloss": 0.20689144730567932
    },
    "q7_shuffle": {
      "AUC": 0.8583022347267555,
      "logloss": 0.20676887035369873
    },
    "inference_placement_steps1": {
      "AUC": 0.7860510856191687,
      "logloss": 0.23228907585144043
    },
    "inference_placement_steps3": {
      "AUC": 0.8411138193371116,
      "logloss": 0.2130054086446762
    },
    "extra_attention_off": {
      "AUC": 0.8521807660778075,
      "logloss": 0.20899304747581482
    }
  },
  "checkpoint_unchanged": true,
  "note": "subset diagnostics; within-block shuffle; blockwise loss means in toolkit differ from pooled full-validation metrics"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 14](#phase-14)。
- **后续假设**：保留GELU，单独改变FFN宽度以分离容量因素；见K15。
- **证据**：`qformer-loop/reports/round14/r14_silu_both/summary.json`（SHA256前缀 `69afc01c4b78`）；`qformer-loop/reports/round14/r14_silu_both/toolkit-hooks/hook_report.json`（SHA256前缀 `630be757a714`）；`qformer-loop/reports/round14/r14_silu_both/toolkit-perturbations/perturbation_report.json`（SHA256前缀 `187f9ce6ce8a`）

<a id="k192"></a>
### K192 · `diagnostics-round14-r14_swiglu_both`

- **方法**：Query Zero/Shuffle, extra attention off and inference-depth interventions
- **原假设**：Measure Query and recurrence reliance with gated or SiLU FFNs
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"training_seed":20262027,"new_training_seed_values":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "rows": 57408,
  "users": 7176,
  "positives": 4472,
  "precision": "bf16",
  "scores": {
    "original": {
      "AUC": 0.8613102148123377,
      "logloss": 0.20271864533424377
    },
    "all_queries": {
      "AUC": 0.8613102148123377,
      "logloss": 0.20271864533424377
    },
    "q3_half": {
      "AUC": 0.860980615401377,
      "logloss": 0.20304256677627563
    },
    "q0_zero": {
      "AUC": 0.8601425734366379,
      "logloss": 0.20682014524936676
    },
    "q0_shuffle": {
      "AUC": 0.8583460589531544,
      "logloss": 0.2054436355829239
    },
    "q1_zero": {
      "AUC": 0.8613979921040102,
      "logloss": 0.2040923833847046
    },
    "q1_shuffle": {
      "AUC": 0.8585856253360794,
      "logloss": 0.2064930647611618
    },
    "q2_zero": {
      "AUC": 0.8605383157688914,
      "logloss": 0.20607250928878784
    },
    "q2_shuffle": {
      "AUC": 0.8513052615701195,
      "logloss": 0.2112446129322052
    },
    "q3_zero": {
      "AUC": 0.8603645374723262,
      "logloss": 0.20411540567874908
    },
    "q3_shuffle": {
      "AUC": 0.8563279669506066,
      "logloss": 0.20756877958774567
    },
    "q4_zero": {
      "AUC": 0.8609610910315841,
      "logloss": 0.2032437026500702
    },
    "q4_shuffle": {
      "AUC": 0.8577959592006063,
      "logloss": 0.20663940906524658
    },
    "q5_zero": {
      "AUC": 0.8607693407680601,
      "logloss": 0.20355495810508728
    },
    "q5_shuffle": {
      "AUC": 0.8586280281106317,
      "logloss": 0.20528312027454376
    },
    "q6_zero": {
      "AUC": 0.8598883173098889,
      "logloss": 0.20381605625152588
    },
    "q6_shuffle": {
      "AUC": 0.8567998678425739,
      "logloss": 0.20665527880191803
    },
    "q7_zero": {
      "AUC": 0.8599384588653717,
      "logloss": 0.20437383651733398
    },
    "q7_shuffle": {
      "AUC": 0.8559522706799827,
      "logloss": 0.2065667062997818
    },
    "inference_placement_steps1": {
      "AUC": 0.8055328942290457,
      "logloss": 0.22888293862342834
    },
    "inference_placement_steps3": {
      "AUC": 0.8403158251412648,
      "logloss": 0.2150924652814865
    },
    "extra_attention_off": {
      "AUC": 0.8530894877819181,
      "logloss": 0.2098907232284546
    }
  },
  "checkpoint_unchanged": true,
  "note": "subset diagnostics; within-block shuffle; blockwise loss means in toolkit differ from pooled full-validation metrics"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 14](#phase-14)。
- **后续假设**：保留GELU，单独改变FFN宽度以分离容量因素；见K15。
- **证据**：`qformer-loop/reports/round14/r14_swiglu_both/summary.json`（SHA256前缀 `d22058f160cb`）；`qformer-loop/reports/round14/r14_swiglu_both/toolkit-hooks/hook_report.json`（SHA256前缀 `f41662a8c56d`）；`qformer-loop/reports/round14/r14_swiglu_both/toolkit-perturbations/perturbation_report.json`（SHA256前缀 `a46db1aa5a44`）

<a id="k193"></a>
### K193 · `round14-gated-parent-paired-bootstrap`

- **方法**：Resample existing validation users with the existing bootstrap seed; no training
- **原假设**：Quantify conditional AUC uncertainty between best gated model and r7
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"training_seed":20262027,"new_training_seed_values":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "method": "paired user-cluster bootstrap of global AUC",
  "repetitions": 500,
  "ci95": [
    -0.001477855676784584,
    -0.0006563400180776308
  ],
  "note": "conditional on these trained checkpoints; does not capture training-seed variance",
  "baseline": "qformer-loop/runs/r7_small_attention_first-s20262027-20260906T163655782605Z",
  "candidate": "qformer-loop/runs/r14_swiglu_both-s20262027-20260907T045344417858Z",
  "training_updates": 0,
  "training_seed": 20262027,
  "bootstrap_seed": 20260906,
  "test_evaluated": false
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 14](#phase-14)。
- **后续假设**：保留GELU，单独改变FFN宽度以分离容量因素；见K15。
- **证据**：`qformer-loop/reports/round14/paired-bootstrap.json`（SHA256前缀 `683d2e45c8a5`）

<a id="phase-15"></a>
## 阶段 15：普通FFN容量分配

**方法**：字段/历史/双位置宽度512、680及历史/双位置1536。

**归因与边界**：r15_w512_field AUC=0.871053474、7.853M参数，较r7少262656；更少参数有价值，但短基准没有证明明显提速。

**历史后续假设**：在相同更新预算下变化训练循环深度、固定两轮验证；见K16。

<a id="k194"></a>
### K194 · `r15_w512_field-s20262027-20260907T052153858349Z`

- **方法**：{"dense_ffn_width":512,"dense_ffn_scope":"field","parameter_delta":-262656}；设置 {"model":"QFormerCross11Width","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Change plain GELU FFN capacity with fixed shared recurrence and attention
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.188110974；AUC=0.871053474；UAUC=0.788709683；gAUC=0.801188848；NDCG(10)=0.408509683。
- **开销**：参数 7852817；训练含验证 439.1194362565875 秒；峰值CUDA 16627563008 bytes；更新 1240。
- **原记录差值**：-0.000180585；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":4.41151615204749e-05,"AUC":-0.00018058483974459083,"UAUC":-0.0021273300973583265,"gAUC":-0.002601814328944685,"NDCG(10)":-0.0013464741217236176}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.654784202575684,"mean_ms":12.659928846359254,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000180585，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 15](#phase-15)。
- **后续假设**：在相同更新预算下变化训练循环深度、固定两轮验证；见K16。
- **原判读**：本轮首选缩小候选：参数减少262656，AUC接近父模型，配对区间包含零；保留质量/成本取舍
- **证据**：`qformer-loop/runs/r15_w512_field-s20262027-20260907T052153858349Z/manifest.json`（SHA256前缀 `719eef02c0f5`）；`qformer-loop/runs/r15_w512_field-s20262027-20260907T052153858349Z/result.json`（SHA256前缀 `a21ce930998a`）；`qformer-loop/runs/r15_w512_field-s20262027-20260907T052153858349Z/inference-all_queries.json`（SHA256前缀 `20b2419cc1e2`）

<a id="k195"></a>
### K195 · `r15_w512_history-s20262027-20260907T052153719493Z`

- **方法**：{"dense_ffn_width":512,"dense_ffn_scope":"history","parameter_delta":-262656}；设置 {"model":"QFormerCross11Width","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Change plain GELU FFN capacity with fixed shared recurrence and attention
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.189338927；AUC=0.869505806；UAUC=0.788255944；gAUC=0.800652721；NDCG(10)=0.406279247。
- **开销**：参数 7852817；训练含验证 434.6634452268481 秒；峰值CUDA 16761453056 bytes；更新 1240。
- **原记录差值**：-0.001728253；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.0012720687020119081,"AUC":-0.0017282526574304846,"UAUC":-0.0025810691262988517,"gAUC":-0.0031379408022733646,"NDCG(10)":-0.00357690948957734}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.644976139068604,"mean_ms":12.692140817642212,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.001728253，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 15](#phase-15)。
- **后续假设**：在相同更新预算下变化训练循环深度、固定两轮验证；见K16。
- **原判读**：保留完整宽度对照；AUC 未超过父模型
- **证据**：`qformer-loop/runs/r15_w512_history-s20262027-20260907T052153719493Z/manifest.json`（SHA256前缀 `ffe1716c31bc`）；`qformer-loop/runs/r15_w512_history-s20262027-20260907T052153719493Z/result.json`（SHA256前缀 `18f52805ec3b`）；`qformer-loop/runs/r15_w512_history-s20262027-20260907T052153719493Z/inference-all_queries.json`（SHA256前缀 `96f2e3bfba73`）

<a id="k196"></a>
### K196 · `r15_w512_both-s20262027-20260907T052153886924Z`

- **方法**：{"dense_ffn_width":512,"dense_ffn_scope":"both","parameter_delta":-525312}；设置 {"model":"QFormerCross11Width","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Change plain GELU FFN capacity with fixed shared recurrence and attention
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.188816808；AUC=0.870463677；UAUC=0.790708073；gAUC=0.802558931；NDCG(10)=0.409079835。
- **开销**：参数 7590161；训练含验证 444.0506442785263 秒；峰值CUDA 16491442688 bytes；更新 1240。
- **原记录差值**：-0.000770381；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.0007499499026315803,"AUC":-0.00077038147831332,"UAUC":-0.00012894019671039203,"gAUC":-0.0012317315949808005,"NDCG(10)":-0.0007763215173687832}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.585792064666748,"mean_ms":12.592715239524841,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000770381，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 15](#phase-15)。
- **后续假设**：在相同更新预算下变化训练循环深度、固定两轮验证；见K16。
- **原判读**：本轮最小模型，AUC高于Cross11，保留更小容量候选
- **证据**：`qformer-loop/runs/r15_w512_both-s20262027-20260907T052153886924Z/manifest.json`（SHA256前缀 `98ca6edc7002`）；`qformer-loop/runs/r15_w512_both-s20262027-20260907T052153886924Z/result.json`（SHA256前缀 `5b123b5b34d8`）；`qformer-loop/runs/r15_w512_both-s20262027-20260907T052153886924Z/inference-all_queries.json`（SHA256前缀 `f67cc56a2bf9`）

<a id="k197"></a>
### K197 · `r15_w680_field-s20262027-20260907T052153748368Z`

- **方法**：{"dense_ffn_width":680,"dense_ffn_scope":"field","parameter_delta":-176472}；设置 {"model":"QFormerCross11Width","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Change plain GELU FFN capacity with fixed shared recurrence and attention
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.188920516；AUC=0.869975059；UAUC=0.789298901；gAUC=0.801445881；NDCG(10)=0.408986917。
- **开销**：参数 7939001；训练含验证 437.20650815963745 秒；峰值CUDA 16716851200 bytes；更新 1240。
- **原记录差值**：-0.001259000；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.0008536575627600684,"AUC":-0.0012589997548750897,"UAUC":-0.00153811198765208,"gAUC":-0.002344781648966232,"NDCG(10)":-0.0008692403484832356}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.691615581512451,"mean_ms":12.699785566329956,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.001259000，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 15](#phase-15)。
- **后续假设**：在相同更新预算下变化训练循环深度、固定两轮验证；见K16。
- **原判读**：AUC接近Cross11，保留中间字段宽度对照
- **证据**：`qformer-loop/runs/r15_w680_field-s20262027-20260907T052153748368Z/manifest.json`（SHA256前缀 `e1c4dd50cc3b`）；`qformer-loop/runs/r15_w680_field-s20262027-20260907T052153748368Z/result.json`（SHA256前缀 `2210cdd203eb`）；`qformer-loop/runs/r15_w680_field-s20262027-20260907T052153748368Z/inference-all_queries.json`（SHA256前缀 `8d2e06c73427`）

<a id="k198"></a>
### K198 · `r15_w680_history-s20262027-20260907T052153885943Z`

- **方法**：{"dense_ffn_width":680,"dense_ffn_scope":"history","parameter_delta":-176472}；设置 {"model":"QFormerCross11Width","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Change plain GELU FFN capacity with fixed shared recurrence and attention
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.189223429；AUC=0.869431990；UAUC=0.789975490；gAUC=0.801725767；NDCG(10)=0.408219565。
- **开销**：参数 7939001；训练含验证 438.0361124202609 秒；峰值CUDA 16807028736 bytes；更新 1240。
- **原记录差值**：-0.001802068；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.0011565709419050496,"AUC":-0.0018020684147087573,"UAUC":-0.0008615231557261849,"gAUC":-0.0020648947395185413,"NDCG(10)":-0.0016365918251613487}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.691247940063477,"mean_ms":12.692905569076538,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.001802068，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 15](#phase-15)。
- **后续假设**：在相同更新预算下变化训练循环深度、固定两轮验证；见K16。
- **原判读**：保留完整宽度对照；AUC 未超过父模型
- **证据**：`qformer-loop/runs/r15_w680_history-s20262027-20260907T052153885943Z/manifest.json`（SHA256前缀 `121829f472f6`）；`qformer-loop/runs/r15_w680_history-s20262027-20260907T052153885943Z/result.json`（SHA256前缀 `d010fbe9082e`）；`qformer-loop/runs/r15_w680_history-s20262027-20260907T052153885943Z/inference-all_queries.json`（SHA256前缀 `4bead19bd836`）

<a id="k199"></a>
### K199 · `r15_w680_both-s20262027-20260907T052153679180Z`

- **方法**：{"dense_ffn_width":680,"dense_ffn_scope":"both","parameter_delta":-352944}；设置 {"model":"QFormerCross11Width","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Change plain GELU FFN capacity with fixed shared recurrence and attention
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.189032395；AUC=0.869015799；UAUC=0.789517647；gAUC=0.801750494；NDCG(10)=0.406191538。
- **开销**：参数 7762529；训练含验证 459.24536541104317 秒；峰值CUDA 16626306560 bytes；更新 1240。
- **原记录差值**：-0.002218260；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.0009655361578499166,"AUC":-0.0022182595935306137,"UAUC":-0.0013193658426667865,"gAUC":-0.002040168015788657,"NDCG(10)":-0.0036646193250948755}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.627744197845459,"mean_ms":12.633384799957275,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.002218260，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 15](#phase-15)。
- **后续假设**：在相同更新预算下变化训练循环深度、固定两轮验证；见K16。
- **原判读**：保留完整宽度对照；AUC 未超过父模型
- **证据**：`qformer-loop/runs/r15_w680_both-s20262027-20260907T052153679180Z/manifest.json`（SHA256前缀 `f6e20d22a0f4`）；`qformer-loop/runs/r15_w680_both-s20262027-20260907T052153679180Z/result.json`（SHA256前缀 `ff762f92d93d`）；`qformer-loop/runs/r15_w680_both-s20262027-20260907T052153679180Z/inference-all_queries.json`（SHA256前缀 `d55a4fd3c66a`）

<a id="k200"></a>
### K200 · `r15_w1536_history-s20262027-20260907T052153898429Z`

- **方法**：{"dense_ffn_width":1536,"dense_ffn_scope":"history","parameter_delta":262656}；设置 {"model":"QFormerCross11Width","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Change plain GELU FFN capacity with fixed shared recurrence and attention
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.188325295；AUC=0.870377188；UAUC=0.789921108；gAUC=0.801934692；NDCG(10)=0.407630063。
- **开销**：参数 8378129；训练含验证 440.4127125814557 秒；峰值CUDA 17036915200 bytes；更新 1240。
- **原记录差值**：-0.000856871；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.0002584369772182238,"AUC":-0.0008568706990409458,"UAUC":-0.0009159046189000097,"gAUC":-0.0018559703305598418,"NDCG(10)":-0.002226093995921774}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.705696105957031,"mean_ms":12.883260798454284,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000856871，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 15](#phase-15)。
- **后续假设**：在相同更新预算下变化训练循环深度、固定两轮验证；见K16。
- **原判读**：最佳加宽对照；AUC低于字段缩减且参数更多
- **证据**：`qformer-loop/runs/r15_w1536_history-s20262027-20260907T052153898429Z/manifest.json`（SHA256前缀 `94e3def56b32`）；`qformer-loop/runs/r15_w1536_history-s20262027-20260907T052153898429Z/result.json`（SHA256前缀 `d09cc7d3cda4`）；`qformer-loop/runs/r15_w1536_history-s20262027-20260907T052153898429Z/inference-all_queries.json`（SHA256前缀 `f210d825a45d`）

<a id="k201"></a>
### K201 · `r15_w1536_both-s20262027-20260907T052939555012Z`

- **方法**：{"dense_ffn_width":1536,"dense_ffn_scope":"both","parameter_delta":525312}；设置 {"model":"QFormerCross11Width","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Change plain GELU FFN capacity with fixed shared recurrence and attention
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.188660020；AUC=0.870282624；UAUC=0.789213885；gAUC=0.801561926；NDCG(10)=0.410035927。
- **开销**：参数 8640785；训练含验证 443.14193906635046 秒；峰值CUDA 17311588864 bytes；更新 1240。
- **原记录差值**：-0.000951434；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.0005931612339620806,"AUC":-0.0009514341990274344,"UAUC":-0.0016231277585793125,"gAUC":-0.0022287361544280904,"NDCG(10)":0.00017977004130620422}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.871871948242188,"mean_ms":12.888731956481934,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000951434，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 15](#phase-15)。
- **后续假设**：在相同更新预算下变化训练循环深度、固定两轮验证；见K16。
- **原判读**：NDCG略高父模型，保留容量/排序取舍
- **证据**：`qformer-loop/runs/r15_w1536_both-s20262027-20260907T052939555012Z/manifest.json`（SHA256前缀 `e9e81c25e1a8`）；`qformer-loop/runs/r15_w1536_both-s20262027-20260907T052939555012Z/result.json`（SHA256前缀 `d9fce0fd1cdb`）；`qformer-loop/runs/r15_w1536_both-s20262027-20260907T052939555012Z/inference-all_queries.json`（SHA256前缀 `c6ec62d86d11`）

<a id="k202"></a>
### K202 · `diagnostics-round15-r15_w1536_history`

- **方法**：Query Zero/Shuffle, extra attention off and inference-depth interventions
- **原假设**：Measure Query and recurrence reliance after dense FFN width changes
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"training_seed":20262027,"new_training_seed_values":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "rows": 57408,
  "users": 7176,
  "positives": 4472,
  "precision": "bf16",
  "scores": {
    "original": {
      "AUC": 0.860636569139553,
      "logloss": 0.20267002284526825
    },
    "all_queries": {
      "AUC": 0.860636569139553,
      "logloss": 0.20267002284526825
    },
    "q3_half": {
      "AUC": 0.8604114496074917,
      "logloss": 0.2029542624950409
    },
    "q0_zero": {
      "AUC": 0.8599093539523744,
      "logloss": 0.20342306792736053
    },
    "q0_shuffle": {
      "AUC": 0.857787088327269,
      "logloss": 0.20617052912712097
    },
    "q1_zero": {
      "AUC": 0.8598549923957184,
      "logloss": 0.2048388123512268
    },
    "q1_shuffle": {
      "AUC": 0.8559844318200559,
      "logloss": 0.2078464776277542
    },
    "q2_zero": {
      "AUC": 0.8604363007255125,
      "logloss": 0.20365379750728607
    },
    "q2_shuffle": {
      "AUC": 0.858014155227239,
      "logloss": 0.20603208243846893
    },
    "q3_zero": {
      "AUC": 0.8595790406473216,
      "logloss": 0.2049049735069275
    },
    "q3_shuffle": {
      "AUC": 0.8544133431249752,
      "logloss": 0.20835383236408234
    },
    "q4_zero": {
      "AUC": 0.8607067778777924,
      "logloss": 0.20574219524860382
    },
    "q4_shuffle": {
      "AUC": 0.8584342227614511,
      "logloss": 0.20502403378486633
    },
    "q5_zero": {
      "AUC": 0.8603072421911306,
      "logloss": 0.20399489998817444
    },
    "q5_shuffle": {
      "AUC": 0.8585208764091677,
      "logloss": 0.205413356423378
    },
    "q6_zero": {
      "AUC": 0.8592831695640574,
      "logloss": 0.20871774852275848
    },
    "q6_shuffle": {
      "AUC": 0.857578960741874,
      "logloss": 0.20562410354614258
    },
    "q7_zero": {
      "AUC": 0.8602764349152978,
      "logloss": 0.2043275386095047
    },
    "q7_shuffle": {
      "AUC": 0.8585365292763828,
      "logloss": 0.20518028736114502
    },
    "inference_placement_steps1": {
      "AUC": 0.842762667573332,
      "logloss": 0.21994934976100922
    },
    "inference_placement_steps3": {
      "AUC": 0.8454128790008821,
      "logloss": 0.21728140115737915
    },
    "extra_attention_off": {
      "AUC": 0.8585784061348729,
      "logloss": 0.2068128138780594
    }
  },
  "checkpoint_unchanged": true,
  "note": "subset diagnostics; within-block shuffle; blockwise loss means in toolkit differ from pooled full-validation metrics"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 15](#phase-15)。
- **后续假设**：在相同更新预算下变化训练循环深度、固定两轮验证；见K16。
- **证据**：`qformer-loop/reports/round15/r15_w1536_history/summary.json`（SHA256前缀 `7316b793aae5`）；`qformer-loop/reports/round15/r15_w1536_history/toolkit-hooks/hook_report.json`（SHA256前缀 `410ec24b58b4`）；`qformer-loop/reports/round15/r15_w1536_history/toolkit-perturbations/perturbation_report.json`（SHA256前缀 `4d4abded39f4`）

<a id="k203"></a>
### K203 · `diagnostics-round15-r15_w512_field`

- **方法**：Query Zero/Shuffle, extra attention off and inference-depth interventions
- **原假设**：Measure Query and recurrence reliance after dense FFN width changes
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"training_seed":20262027,"new_training_seed_values":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "rows": 57408,
  "users": 7176,
  "positives": 4472,
  "precision": "bf16",
  "scores": {
    "original": {
      "AUC": 0.8622511884773675,
      "logloss": 0.20196528732776642
    },
    "all_queries": {
      "AUC": 0.8622511884773675,
      "logloss": 0.20196528732776642
    },
    "q3_half": {
      "AUC": 0.8616163592117717,
      "logloss": 0.20257216691970825
    },
    "q0_zero": {
      "AUC": 0.8623344669689906,
      "logloss": 0.20598958432674408
    },
    "q0_shuffle": {
      "AUC": 0.8577847713396376,
      "logloss": 0.20641744136810303
    },
    "q1_zero": {
      "AUC": 0.862158772986207,
      "logloss": 0.20390287041664124
    },
    "q1_shuffle": {
      "AUC": 0.8546873137116601,
      "logloss": 0.2100294977426529
    },
    "q2_zero": {
      "AUC": 0.8613356488734634,
      "logloss": 0.2057819813489914
    },
    "q2_shuffle": {
      "AUC": 0.8586171063758634,
      "logloss": 0.20576211810112
    },
    "q3_zero": {
      "AUC": 0.8602526757595427,
      "logloss": 0.20601359009742737
    },
    "q3_shuffle": {
      "AUC": 0.8586571055661638,
      "logloss": 0.20466455817222595
    },
    "q4_zero": {
      "AUC": 0.8619703260669448,
      "logloss": 0.20294055342674255
    },
    "q4_shuffle": {
      "AUC": 0.8599327075824915,
      "logloss": 0.20499221980571747
    },
    "q5_zero": {
      "AUC": 0.8615048079795551,
      "logloss": 0.20351654291152954
    },
    "q5_shuffle": {
      "AUC": 0.8578637495698048,
      "logloss": 0.20706187188625336
    },
    "q6_zero": {
      "AUC": 0.8620548718261875,
      "logloss": 0.2032139003276825
    },
    "q6_shuffle": {
      "AUC": 0.8612265244587383,
      "logloss": 0.20352348685264587
    },
    "q7_zero": {
      "AUC": 0.8618885323905493,
      "logloss": 0.2041495442390442
    },
    "q7_shuffle": {
      "AUC": 0.8594483156560201,
      "logloss": 0.20534098148345947
    },
    "inference_placement_steps1": {
      "AUC": 0.8187573957738281,
      "logloss": 0.22919988632202148
    },
    "inference_placement_steps3": {
      "AUC": 0.8464867299000541,
      "logloss": 0.21750587224960327
    },
    "extra_attention_off": {
      "AUC": 0.8591894551235866,
      "logloss": 0.20542696118354797
    }
  },
  "checkpoint_unchanged": true,
  "note": "subset diagnostics; within-block shuffle; blockwise loss means in toolkit differ from pooled full-validation metrics"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 15](#phase-15)。
- **后续假设**：在相同更新预算下变化训练循环深度、固定两轮验证；见K16。
- **证据**：`qformer-loop/reports/round15/r15_w512_field/summary.json`（SHA256前缀 `fbf2ced1d673`）；`qformer-loop/reports/round15/r15_w512_field/toolkit-hooks/hook_report.json`（SHA256前缀 `c10aa406d7d8`）；`qformer-loop/reports/round15/r15_w512_field/toolkit-perturbations/perturbation_report.json`（SHA256前缀 `520b408da918`）

<a id="k204"></a>
### K204 · `round15-reduced-field-parent-paired-bootstrap`

- **方法**：Resample existing validation users; no new training seed or model updates
- **原假设**：Quantify conditional AUC uncertainty for smaller field FFN versus r7
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"training_seed":20262027,"new_training_seed_values":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "method": "paired user-cluster bootstrap of global AUC",
  "repetitions": 500,
  "ci95": [
    -0.0006075657389414651,
    0.0002704879919653208
  ],
  "note": "conditional on these trained checkpoints; does not capture training-seed variance",
  "baseline": "qformer-loop/runs/r7_small_attention_first-s20262027-20260906T163655782605Z",
  "candidate": "qformer-loop/runs/r15_w512_field-s20262027-20260907T052153858349Z",
  "training_updates": 0,
  "training_seed": 20262027,
  "bootstrap_seed": 20260906,
  "test_evaluated": false
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 15](#phase-15)。
- **后续假设**：在相同更新预算下变化训练循环深度、固定两轮验证；见K16。
- **证据**：`qformer-loop/reports/round15/paired-bootstrap.json`（SHA256前缀 `0e5970f94f37`）

<a id="phase-16"></a>
## 阶段 16：可变训练深度

**方法**：alternate13、mixed1232、先1后3课程；原容量及field512。

**归因与边界**：6组AUC均低于固定两轮；base curriculum13的NDCG=0.421730872最高，但AUC=0.864854885明显下降。

**历史后续假设**：控制各深度更新次数并改变阶段顺序，以两轮结束；见K17。

<a id="k205"></a>
### K205 · `round16-synthetic-fixture-failures`

- **方法**：Correct forward-only fixture labels and feature-map mismatch; two CPU and one GPU failed invocations preserved
- **原假设**：Verify real train_step before launching variable-depth experiments
- **状态／预算**：原记录 `crashed`；{"training_runs":0,"new_training_seed_values":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "decision": "crashed",
  "validity": {
    "valid": false,
    "violations": [
      "synthetic fixture missing labels or containing undeclared seq_len"
    ],
    "notes": [
      "fixed before full training; final CPU/GPU tests passed"
    ]
  },
  "falsification_result": null,
  "note": "本条未保存独立性能指标；保留运行状态与证据位置，不补造数值。"
}
```

</details>

- **归因**：这是执行或有效性问题；本条不能用作结构性能反证。修复后完整训练需用独立运行ID核对。 本阶段证据解释见[阶段 16](#phase-16)。
- **后续假设**：控制各深度更新次数并改变阶段顺序，以两轮结束；见K17。
- **失败后续门槛**：先复核同配置/初始函数/有效梯度及完整预算，再判断科学假设；不把同一次修复重复算成独立支持。
- **证据**：`qformer-loop/reports/round16/failures.json`（SHA256前缀 `c8c8f601740f`）；`qformer-loop/reports/round16/cpu-tests-fixture-failure.txt`；`qformer-loop/reports/round16/cpu-tests-seqlen-fixture-failure.txt`；`qformer-loop/reports/round16/gpu-tests-seqlen-fixture-failure.txt`

<a id="k206"></a>
### K206 · `r16_base_alternate13-s20262027-20260907T071543287067Z`

- **方法**：{"parent":"r7_small_attention_first","training_depth_schedule":"alternate13","dense_ffn_width":0,"parameter_delta":0}；设置 {"model":"QFormerCross11DepthSchedule","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Vary full-backprop training depth with fixed depth2 evaluation
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027],"training_compute":{"schedule":"alternate13","planned_updates":1240,"completed_updates":1240,"by_depth":{"1":{"updates":620,"examples":4995558,"observed_history_tokens":4611588416},"3":{"updates":620,"examples":5026986,"observed_history_tokens":4640211178}},"total_cycle_calls":2480,"example_cycles":20076516,"observed_history_token_cycles":18532221950,"evaluation_depth":2,"note":"Cycle counts are not exact FLOPs: batches and padded histories differ; first-only extra attention is excluded from cycle count"}}。
- **结果**：logloss=0.200972187；AUC=0.863724447；UAUC=0.791713388；gAUC=0.803746836；NDCG(10)=0.408628268。
- **开销**：参数 8115473；训练含验证 437.00299564003944 秒；峰值CUDA 22717237760 bytes；更新 1240。
- **原记录差值**：-0.007509611；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.012905328962197676,"AUC":-0.007509611130158467,"UAUC":0.0008763748929245807,"gAUC":-4.382611492870225e-05,"NDCG(10)":-0.0012278886749954587}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.653599739074707,"mean_ms":12.654967260360717,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.007509611，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 16](#phase-16)。
- **后续假设**：控制各深度更新次数并改变阶段顺序，以两轮结束；见K17。
- **原判读**：保留可变深度负对照；固定两轮完整验证AUC低于对应父模型
- **证据**：`qformer-loop/runs/r16_base_alternate13-s20262027-20260907T071543287067Z/manifest.json`（SHA256前缀 `93e9b5bb279f`）；`qformer-loop/runs/r16_base_alternate13-s20262027-20260907T071543287067Z/result.json`（SHA256前缀 `9a94eef79950`）；`qformer-loop/runs/r16_base_alternate13-s20262027-20260907T071543287067Z/inference-all_queries.json`（SHA256前缀 `18a00e616f0c`）

<a id="k207"></a>
### K207 · `r16_base_mixed1232-s20262027-20260907T071543487684Z`

- **方法**：{"parent":"r7_small_attention_first","training_depth_schedule":"mixed1232","dense_ffn_width":0,"parameter_delta":0}；设置 {"model":"QFormerCross11DepthSchedule","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Vary full-backprop training depth with fixed depth2 evaluation
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027],"training_compute":{"schedule":"mixed1232","planned_updates":1240,"completed_updates":1240,"by_depth":{"1":{"updates":310,"examples":2494497,"observed_history_tokens":2302794948},"2":{"updates":620,"examples":5026986,"observed_history_tokens":4640211178},"3":{"updates":310,"examples":2501061,"observed_history_tokens":2308793468}},"total_cycle_calls":2480,"example_cycles":20051652,"observed_history_token_cycles":18509597708,"evaluation_depth":2,"note":"Cycle counts are not exact FLOPs: batches and padded histories differ; first-only extra attention is excluded from cycle count"}}。
- **结果**：logloss=0.189291512；AUC=0.868667909；UAUC=0.788412440；gAUC=0.800844824；NDCG(10)=0.402673785。
- **开销**：参数 8115473；训练含验证 444.7528090104461 秒；峰值CUDA 22717237760 bytes；更新 1240。
- **原记录差值**：-0.002566149；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.0012246538476509694,"AUC":-0.0025661490470145765,"UAUC":-0.002424572394072233,"gAUC":-0.0029458385039737855,"NDCG(10)":-0.007182371768237106}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.645663738250732,"mean_ms":12.648301529884339,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.002566149，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 16](#phase-16)。
- **后续假设**：控制各深度更新次数并改变阶段顺序，以两轮结束；见K17。
- **原判读**：原父模型可变深度中AUC最好，但低于固定2父模型和Cross8/Cross11
- **证据**：`qformer-loop/runs/r16_base_mixed1232-s20262027-20260907T071543487684Z/manifest.json`（SHA256前缀 `d859b5afd571`）；`qformer-loop/runs/r16_base_mixed1232-s20262027-20260907T071543487684Z/result.json`（SHA256前缀 `7130c7a2d184`）；`qformer-loop/runs/r16_base_mixed1232-s20262027-20260907T071543487684Z/inference-all_queries.json`（SHA256前缀 `f45aada6e738`）

<a id="k208"></a>
### K208 · `r16_base_curriculum13-s20262027-20260907T071543460834Z`

- **方法**：{"parent":"r7_small_attention_first","training_depth_schedule":"curriculum13","dense_ffn_width":0,"parameter_delta":0}；设置 {"model":"QFormerCross11DepthSchedule","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Vary full-backprop training depth with fixed depth2 evaluation
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027],"training_compute":{"schedule":"curriculum13","planned_updates":1240,"completed_updates":1240,"by_depth":{"1":{"updates":620,"examples":5015111,"observed_history_tokens":4623119615},"3":{"updates":620,"examples":5007433,"observed_history_tokens":4628679979}},"total_cycle_calls":2480,"example_cycles":20037410,"observed_history_token_cycles":18509159552,"evaluation_depth":2,"note":"Cycle counts are not exact FLOPs: batches and padded histories differ; first-only extra attention is excluded from cycle count"}}。
- **结果**：logloss=0.192462496；AUC=0.864854885；UAUC=0.795442921；gAUC=0.805846715；NDCG(10)=0.421730872。
- **开销**：参数 8115473；训练含验证 438.471804484725 秒；峰值CUDA 22717237760 bytes；更新 1240。
- **原记录差值**：-0.006379174；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.004395637323698576,"AUC":-0.006379173512010339,"UAUC":0.004605908402098624,"gAUC":0.002056052998401503,"NDCG(10)":0.011874715108006584}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.657279968261719,"mean_ms":12.665212798118592,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.006379174，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 16](#phase-16)。
- **后续假设**：控制各深度更新次数并改变阶段顺序，以两轮结束；见K17。
- **原判读**：同seed完整验证NDCG最高的新排序候选，但AUC/Logloss明显变差；不替换AUC首选
- **证据**：`qformer-loop/runs/r16_base_curriculum13-s20262027-20260907T071543460834Z/manifest.json`（SHA256前缀 `30ef5817a097`）；`qformer-loop/runs/r16_base_curriculum13-s20262027-20260907T071543460834Z/result.json`（SHA256前缀 `aae2273a6d49`）；`qformer-loop/runs/r16_base_curriculum13-s20262027-20260907T071543460834Z/inference-all_queries.json`（SHA256前缀 `b4dec7e52cb5`）

<a id="k209"></a>
### K209 · `r16_field512_alternate13-s20262027-20260907T071543464488Z`

- **方法**：{"parent":"r15_w512_field","training_depth_schedule":"alternate13","dense_ffn_width":512,"parameter_delta":0}；设置 {"model":"QFormerCross11DepthSchedule","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Vary full-backprop training depth with fixed depth2 evaluation
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027],"training_compute":{"schedule":"alternate13","planned_updates":1240,"completed_updates":1240,"by_depth":{"1":{"updates":620,"examples":4995558,"observed_history_tokens":4611588416},"3":{"updates":620,"examples":5026986,"observed_history_tokens":4640211178}},"total_cycle_calls":2480,"example_cycles":20076516,"observed_history_token_cycles":18532221950,"evaluation_depth":2,"note":"Cycle counts are not exact FLOPs: batches and padded histories differ; first-only extra attention is excluded from cycle count"}}。
- **结果**：logloss=0.191417404；AUC=0.866823095；UAUC=0.792706478；gAUC=0.803782982；NDCG(10)=0.411439324。
- **开销**：参数 7852817；训练含验证 438.0507006198168 秒；峰值CUDA 22313009664 bytes；更新 1240。
- **原记录差值**：-0.004230379；对照 `same-seed r15_w512_field`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r15_w512_field`；完整指标差 {"logloss":0.003306430716681269,"AUC":-0.004230379079090518,"UAUC":0.003996795208560111,"gAUC":0.002594134336762921,"NDCG(10)":0.002929641305300601}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.594751834869385,"mean_ms":12.712790417671204,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.004230379，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 16](#phase-16)。
- **后续假设**：控制各深度更新次数并改变阶段顺序，以两轮结束；见K17。
- **原判读**：保留可变深度负对照；固定两轮完整验证AUC低于对应父模型
- **证据**：`qformer-loop/runs/r16_field512_alternate13-s20262027-20260907T071543464488Z/manifest.json`（SHA256前缀 `7e24aecff7fe`）；`qformer-loop/runs/r16_field512_alternate13-s20262027-20260907T071543464488Z/result.json`（SHA256前缀 `b86d33f9ea87`）；`qformer-loop/runs/r16_field512_alternate13-s20262027-20260907T071543464488Z/inference-all_queries.json`（SHA256前缀 `ed1e1fb1d13a`）

<a id="k210"></a>
### K210 · `r16_field512_mixed1232-s20262027-20260907T071543357571Z`

- **方法**：{"parent":"r15_w512_field","training_depth_schedule":"mixed1232","dense_ffn_width":512,"parameter_delta":0}；设置 {"model":"QFormerCross11DepthSchedule","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Vary full-backprop training depth with fixed depth2 evaluation
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027],"training_compute":{"schedule":"mixed1232","planned_updates":1240,"completed_updates":1240,"by_depth":{"1":{"updates":310,"examples":2494497,"observed_history_tokens":2302794948},"2":{"updates":620,"examples":5026986,"observed_history_tokens":4640211178},"3":{"updates":310,"examples":2501061,"observed_history_tokens":2308793468}},"total_cycle_calls":2480,"example_cycles":20051652,"observed_history_token_cycles":18509597708,"evaluation_depth":2,"note":"Cycle counts are not exact FLOPs: batches and padded histories differ; first-only extra attention is excluded from cycle count"}}。
- **结果**：logloss=0.190242354；AUC=0.866962200；UAUC=0.789349291；gAUC=0.801612150；NDCG(10)=0.404870005。
- **开销**：参数 7852817；训练含验证 437.3663968965411 秒；峰值CUDA 22313009664 bytes；更新 1240。
- **原记录差值**：-0.004091274；对照 `same-seed r15_w512_field`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r15_w512_field`；完整指标差 {"logloss":0.002131379957913876,"AUC":-0.004091273817662167,"UAUC":0.0006396079064540094,"gAUC":0.00042330247520594444,"NDCG(10)":-0.0036396777121287527}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.504768371582031,"mean_ms":12.51431438922882,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.004091274，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 16](#phase-16)。
- **后续假设**：控制各深度更新次数并改变阶段顺序，以两轮结束；见K17。
- **原判读**：保留可变深度负对照；固定两轮完整验证AUC低于对应父模型
- **证据**：`qformer-loop/runs/r16_field512_mixed1232-s20262027-20260907T071543357571Z/manifest.json`（SHA256前缀 `bd7482920641`）；`qformer-loop/runs/r16_field512_mixed1232-s20262027-20260907T071543357571Z/result.json`（SHA256前缀 `058892d59138`）；`qformer-loop/runs/r16_field512_mixed1232-s20262027-20260907T071543357571Z/inference-all_queries.json`（SHA256前缀 `aa94be2967e7`）

<a id="k211"></a>
### K211 · `r16_field512_curriculum13-s20262027-20260907T071543495742Z`

- **方法**：{"parent":"r15_w512_field","training_depth_schedule":"curriculum13","dense_ffn_width":512,"parameter_delta":0}；设置 {"model":"QFormerCross11DepthSchedule","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Vary full-backprop training depth with fixed depth2 evaluation
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027],"training_compute":{"schedule":"curriculum13","planned_updates":1240,"completed_updates":1240,"by_depth":{"1":{"updates":620,"examples":5015111,"observed_history_tokens":4623119615},"3":{"updates":620,"examples":5007433,"observed_history_tokens":4628679979}},"total_cycle_calls":2480,"example_cycles":20037410,"observed_history_token_cycles":18509159552,"evaluation_depth":2,"note":"Cycle counts are not exact FLOPs: batches and padded histories differ; first-only extra attention is excluded from cycle count"}}。
- **结果**：logloss=0.190793547；AUC=0.867554033；UAUC=0.792242822；gAUC=0.804573502；NDCG(10)=0.409425358。
- **开销**：参数 7852817；训练含验证 439.1689646244049 秒；峰值CUDA 22313009664 bytes；更新 1240。
- **原记录差值**：-0.003499440；对照 `same-seed r15_w512_field`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r15_w512_field`；完整指标差 {"logloss":0.002682573626992979,"AUC":-0.003499440497069628,"UAUC":0.0035331391817647884,"gAUC":0.0033846542750283204,"NDCG(10)":0.0009156753154449526}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.50219202041626,"mean_ms":12.506739187240601,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.003499440，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 16](#phase-16)。
- **后续假设**：控制各深度更新次数并改变阶段顺序，以两轮结束；见K17。
- **原判读**：小模型可变深度中AUC最好，UAUC略高但整体AUC明显下降
- **证据**：`qformer-loop/runs/r16_field512_curriculum13-s20262027-20260907T071543495742Z/manifest.json`（SHA256前缀 `91c36cab3401`）；`qformer-loop/runs/r16_field512_curriculum13-s20262027-20260907T071543495742Z/result.json`（SHA256前缀 `4afb8c25fadb`）；`qformer-loop/runs/r16_field512_curriculum13-s20262027-20260907T071543495742Z/inference-all_queries.json`（SHA256前缀 `e2f1e21e6765`）

<a id="k212"></a>
### K212 · `diagnostics-round16-r16_base_curriculum13`

- **方法**：Query Zero/Shuffle, extra attention off and inference-depth interventions
- **原假设**：Measure Query and inference-depth reliance after variable-depth training
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"training_seed":20262027,"new_training_seed_values":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "rows": 57408,
  "users": 7176,
  "positives": 4472,
  "precision": "bf16",
  "scores": {
    "original": {
      "AUC": 0.8564577330427425,
      "logloss": 0.20647896826267242
    },
    "all_queries": {
      "AUC": 0.8564577330427425,
      "logloss": 0.20647896826267242
    },
    "q3_half": {
      "AUC": 0.8561985662539678,
      "logloss": 0.20569771528244019
    },
    "q0_zero": {
      "AUC": 0.856220209072798,
      "logloss": 0.2047811597585678
    },
    "q0_shuffle": {
      "AUC": 0.8511642653747612,
      "logloss": 0.21120800077915192
    },
    "q1_zero": {
      "AUC": 0.855253545781006,
      "logloss": 0.2053164541721344
    },
    "q1_shuffle": {
      "AUC": 0.8547714877390675,
      "logloss": 0.20801740884780884
    },
    "q2_zero": {
      "AUC": 0.8565297413010019,
      "logloss": 0.20481549203395844
    },
    "q2_shuffle": {
      "AUC": 0.8539718545437661,
      "logloss": 0.20951144397258759
    },
    "q3_zero": {
      "AUC": 0.8558065792580936,
      "logloss": 0.20581121742725372
    },
    "q3_shuffle": {
      "AUC": 0.8539881199236639,
      "logloss": 0.20765233039855957
    },
    "q4_zero": {
      "AUC": 0.8561112346180746,
      "logloss": 0.20587293803691864
    },
    "q4_shuffle": {
      "AUC": 0.853796705486059,
      "logloss": 0.20827357470989227
    },
    "q5_zero": {
      "AUC": 0.8567163190005253,
      "logloss": 0.20521879196166992
    },
    "q5_shuffle": {
      "AUC": 0.8553460309718854,
      "logloss": 0.20720338821411133
    },
    "q6_zero": {
      "AUC": 0.8568160571864144,
      "logloss": 0.20707206428050995
    },
    "q6_shuffle": {
      "AUC": 0.8557042157161191,
      "logloss": 0.20623107254505157
    },
    "q7_zero": {
      "AUC": 0.8555817216280072,
      "logloss": 0.20526136457920074
    },
    "q7_shuffle": {
      "AUC": 0.8525363951656748,
      "logloss": 0.21063271164894104
    },
    "inference_placement_steps1": {
      "AUC": 0.8525510426672448,
      "logloss": 0.22212980687618256
    },
    "inference_placement_steps3": {
      "AUC": 0.8609732272311547,
      "logloss": 0.20342111587524414
    },
    "extra_attention_off": {
      "AUC": 0.8560206355438356,
      "logloss": 0.2069547027349472
    }
  },
  "checkpoint_unchanged": true,
  "note": "subset diagnostics; within-block shuffle; blockwise loss means in toolkit differ from pooled full-validation metrics"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 16](#phase-16)。
- **后续假设**：控制各深度更新次数并改变阶段顺序，以两轮结束；见K17。
- **证据**：`qformer-loop/reports/round16/r16_base_curriculum13/summary.json`（SHA256前缀 `68bf42f3bd78`）；`qformer-loop/reports/round16/r16_base_curriculum13/toolkit-hooks/hook_report.json`（SHA256前缀 `6b825dad1a04`）；`qformer-loop/reports/round16/r16_base_curriculum13/toolkit-perturbations/perturbation_report.json`（SHA256前缀 `c9986362ce5e`）

<a id="k213"></a>
### K213 · `diagnostics-round16-r16_base_mixed1232`

- **方法**：Query Zero/Shuffle, extra attention off and inference-depth interventions
- **原假设**：Measure Query and inference-depth reliance after variable-depth training
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"training_seed":20262027,"new_training_seed_values":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "rows": 57408,
  "users": 7176,
  "positives": 4472,
  "precision": "bf16",
  "scores": {
    "original": {
      "AUC": 0.8601823867610208,
      "logloss": 0.20314528048038483
    },
    "all_queries": {
      "AUC": 0.8601823867610208,
      "logloss": 0.20314528048038483
    },
    "q3_half": {
      "AUC": 0.8600502149725202,
      "logloss": 0.20350053906440735
    },
    "q0_zero": {
      "AUC": 0.8596672614826613,
      "logloss": 0.20473702251911163
    },
    "q0_shuffle": {
      "AUC": 0.8581530900006029,
      "logloss": 0.2062261700630188
    },
    "q1_zero": {
      "AUC": 0.8599617343473187,
      "logloss": 0.20546117424964905
    },
    "q1_shuffle": {
      "AUC": 0.8585329893754986,
      "logloss": 0.2052646279335022
    },
    "q2_zero": {
      "AUC": 0.8595961550965245,
      "logloss": 0.20586861670017242
    },
    "q2_shuffle": {
      "AUC": 0.8582470008675545,
      "logloss": 0.20621152222156525
    },
    "q3_zero": {
      "AUC": 0.8597862642484813,
      "logloss": 0.20473189651966095
    },
    "q3_shuffle": {
      "AUC": 0.859238194236237,
      "logloss": 0.204482764005661
    },
    "q4_zero": {
      "AUC": 0.8599921931245562,
      "logloss": 0.2052043080329895
    },
    "q4_shuffle": {
      "AUC": 0.8582037447994716,
      "logloss": 0.20509052276611328
    },
    "q5_zero": {
      "AUC": 0.859607898443133,
      "logloss": 0.20632705092430115
    },
    "q5_shuffle": {
      "AUC": 0.8555838780950731,
      "logloss": 0.2070678174495697
    },
    "q6_zero": {
      "AUC": 0.859895756170816,
      "logloss": 0.2063753306865692
    },
    "q6_shuffle": {
      "AUC": 0.8587001504229769,
      "logloss": 0.20514340698719025
    },
    "q7_zero": {
      "AUC": 0.8601320403306061,
      "logloss": 0.2044406235218048
    },
    "q7_shuffle": {
      "AUC": 0.8582269189000091,
      "logloss": 0.20528684556484222
    },
    "inference_placement_steps1": {
      "AUC": 0.8585384449626011,
      "logloss": 0.20407304167747498
    },
    "inference_placement_steps3": {
      "AUC": 0.8597821097228017,
      "logloss": 0.20346057415008545
    },
    "extra_attention_off": {
      "AUC": 0.8592577312787062,
      "logloss": 0.20384351909160614
    }
  },
  "checkpoint_unchanged": true,
  "note": "subset diagnostics; within-block shuffle; blockwise loss means in toolkit differ from pooled full-validation metrics"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 16](#phase-16)。
- **后续假设**：控制各深度更新次数并改变阶段顺序，以两轮结束；见K17。
- **证据**：`qformer-loop/reports/round16/r16_base_mixed1232/summary.json`（SHA256前缀 `870e3edd4dc9`）；`qformer-loop/reports/round16/r16_base_mixed1232/toolkit-hooks/hook_report.json`（SHA256前缀 `75c76909a701`）；`qformer-loop/reports/round16/r16_base_mixed1232/toolkit-perturbations/perturbation_report.json`（SHA256前缀 `d852d4aeaeca`）

<a id="k214"></a>
### K214 · `diagnostics-round16-r16_field512_curriculum13`

- **方法**：Query Zero/Shuffle, extra attention off and inference-depth interventions
- **原假设**：Measure Query and inference-depth reliance after variable-depth training
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"training_seed":20262027,"new_training_seed_values":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "rows": 57408,
  "users": 7176,
  "positives": 4472,
  "precision": "bf16",
  "scores": {
    "original": {
      "AUC": 0.8576074193483851,
      "logloss": 0.20511089265346527
    },
    "all_queries": {
      "AUC": 0.8576074193483851,
      "logloss": 0.20511089265346527
    },
    "q3_half": {
      "AUC": 0.8567712128095818,
      "logloss": 0.2061757892370224
    },
    "q0_zero": {
      "AUC": 0.8560211318903199,
      "logloss": 0.2108372449874878
    },
    "q0_shuffle": {
      "AUC": 0.8505130080965898,
      "logloss": 0.210370272397995
    },
    "q1_zero": {
      "AUC": 0.8559445994866587,
      "logloss": 0.2074137181043625
    },
    "q1_shuffle": {
      "AUC": 0.8503202334584065,
      "logloss": 0.21035802364349365
    },
    "q2_zero": {
      "AUC": 0.8576305807762464,
      "logloss": 0.20713293552398682
    },
    "q2_shuffle": {
      "AUC": 0.8551711607130547,
      "logloss": 0.20786499977111816
    },
    "q3_zero": {
      "AUC": 0.8549792097988241,
      "logloss": 0.20792005956172943
    },
    "q3_shuffle": {
      "AUC": 0.8531554300525048,
      "logloss": 0.20859812200069427
    },
    "q4_zero": {
      "AUC": 0.8577543167866256,
      "logloss": 0.20790314674377441
    },
    "q4_shuffle": {
      "AUC": 0.8571919224260545,
      "logloss": 0.20646438002586365
    },
    "q5_zero": {
      "AUC": 0.8585958690826712,
      "logloss": 0.20775233209133148
    },
    "q5_shuffle": {
      "AUC": 0.8560308159270464,
      "logloss": 0.2072697877883911
    },
    "q6_zero": {
      "AUC": 0.8579861063705915,
      "logloss": 0.21478687226772308
    },
    "q6_shuffle": {
      "AUC": 0.8561666184372773,
      "logloss": 0.20691902935504913
    },
    "q7_zero": {
      "AUC": 0.8566602614173716,
      "logloss": 0.20972400903701782
    },
    "q7_shuffle": {
      "AUC": 0.8532486249132514,
      "logloss": 0.21027959883213043
    },
    "inference_placement_steps1": {
      "AUC": 0.8557317238719155,
      "logloss": 0.20773251354694366
    },
    "inference_placement_steps3": {
      "AUC": 0.8609687178705416,
      "logloss": 0.2030256688594818
    },
    "extra_attention_off": {
      "AUC": 0.8518263535668548,
      "logloss": 0.21209099888801575
    }
  },
  "checkpoint_unchanged": true,
  "note": "subset diagnostics; within-block shuffle; blockwise loss means in toolkit differ from pooled full-validation metrics"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 16](#phase-16)。
- **后续假设**：控制各深度更新次数并改变阶段顺序，以两轮结束；见K17。
- **证据**：`qformer-loop/reports/round16/r16_field512_curriculum13/summary.json`（SHA256前缀 `b11645927cb6`）；`qformer-loop/reports/round16/r16_field512_curriculum13/toolkit-hooks/hook_report.json`（SHA256前缀 `6a20764023fc`）；`qformer-loop/reports/round16/r16_field512_curriculum13/toolkit-perturbations/perturbation_report.json`（SHA256前缀 `41c77e5ab2bb`）

<a id="k215"></a>
### K215 · `round16-compute-ledger-replay`

- **方法**：Replay original epoch0 loader once; confirm original sample/raw-length ledgers then count actual masks
- **原假设**：Disambiguate raw historical sequence lengths from truncated valid/padded attention positions
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"new_training_seed_values":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "decision": "provisional",
  "validity": {
    "valid": true,
    "violations": [],
    "notes": [
      "read-only; no new model training or test evaluation"
    ]
  },
  "falsification_result": null,
  "note": "本条未保存独立性能指标；保留运行状态与证据位置，不补造数值。"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 16](#phase-16)。
- **后续假设**：控制各深度更新次数并改变阶段顺序，以两轮结束；见K17。
- **证据**：`qformer-loop/reports/round16/replayed-compute.json`（SHA256前缀 `694f46ae080f`）

<a id="phase-17"></a>
## 阶段 17：训练深度阶段顺序

**方法**：phase132/312/123，在两种容量上匹配各深度更新次数。

**归因与边界**：phase132相对其他顺序恢复部分AUC，但都未超过固定两轮；phase312也以两轮结束，故不能只归因于末期深度。

**历史后续假设**：按执行深度区分最终LayerNorm，保留共享主干；见K18。

<a id="k216"></a>
### K216 · `r17_base_phase132-s20262027-20260907T073714696072Z`

- **方法**：{"parent":"r7_small_attention_first","curriculum_order":"132","parameter_delta":0}；设置 {"model":"QFormerCross11Curriculum","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Compare phase order with equal depth counts and fixed evaluation2
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027],"training_compute":{"schedule":"curriculum132","planned_updates":1240,"completed_updates":1240,"by_depth":{"1":{"updates":310,"examples":2506222,"raw_history_length_sum":2344254157,"valid_history_tokens":241543097,"padded_history_positions":250622200},"2":{"updates":620,"examples":5007433,"raw_history_length_sum":4628679979,"valid_history_tokens":482597838,"padded_history_positions":500743300},"3":{"updates":310,"examples":2508889,"raw_history_length_sum":2278865458,"valid_history_tokens":241874359,"padded_history_positions":250888900}},"total_cycle_calls":2480,"example_cycles":20047755,"valid_history_token_cycles":1932361850,"padded_history_position_cycles":2004775500,"evaluation_depth":2,"note":"Raw historical lengths, truncated valid tokens, and padded positions are distinct; cycle exposure is not exact FLOPs"}}。
- **结果**：logloss=0.189195961；AUC=0.869312210；UAUC=0.791543495；gAUC=0.803111120；NDCG(10)=0.412551294。
- **开销**：参数 8115473；训练含验证 440.37038633972406 秒；峰值CUDA 22717237760 bytes；更新 1240。
- **原记录差值**：-0.001921849；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.0011291022432452502,"AUC":-0.0019218487851798693,"UAUC":0.0007064824351875609,"gAUC":-0.0006795425051399073,"NDCG(10)":0.002695137472592646}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.65390396118164,"mean_ms":12.667331218719482,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.001921849，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 17](#phase-17)。
- **后续假设**：按执行深度区分最终LayerNorm，保留共享主干；见K18。
- **原判读**：原模型课程中AUC最高，接近Cross8，NDCG高于r7；未超过固定两轮父模型
- **证据**：`qformer-loop/runs/r17_base_phase132-s20262027-20260907T073714696072Z/manifest.json`（SHA256前缀 `cd4573db5785`）；`qformer-loop/runs/r17_base_phase132-s20262027-20260907T073714696072Z/result.json`（SHA256前缀 `b7c551abb4c4`）；`qformer-loop/runs/r17_base_phase132-s20262027-20260907T073714696072Z/inference-all_queries.json`（SHA256前缀 `d2e5a97ce950`）

<a id="k217"></a>
### K217 · `r17_base_phase312-s20262027-20260907T073714772329Z`

- **方法**：{"parent":"r7_small_attention_first","curriculum_order":"312","parameter_delta":0}；设置 {"model":"QFormerCross11Curriculum","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Compare phase order with equal depth counts and fixed evaluation2
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027],"training_compute":{"schedule":"curriculum312","planned_updates":1240,"completed_updates":1240,"by_depth":{"1":{"updates":310,"examples":2508889,"raw_history_length_sum":2278865458,"valid_history_tokens":241874359,"padded_history_positions":250888900},"2":{"updates":620,"examples":5007433,"raw_history_length_sum":4628679979,"valid_history_tokens":482597838,"padded_history_positions":500743300},"3":{"updates":310,"examples":2506222,"raw_history_length_sum":2344254157,"valid_history_tokens":241543097,"padded_history_positions":250622200}},"total_cycle_calls":2480,"example_cycles":20042421,"valid_history_token_cycles":1931699326,"padded_history_position_cycles":2004242100,"evaluation_depth":2,"note":"Raw historical lengths, truncated valid tokens, and padded positions are distinct; cycle exposure is not exact FLOPs"}}。
- **结果**：logloss=0.189858589；AUC=0.867415296；UAUC=0.789459966；gAUC=0.801595447；NDCG(10)=0.407064086。
- **开销**：参数 8115473；训练含验证 435.8011753410101 秒；峰值CUDA 22716254720 bytes；更新 1240。
- **原记录差值**：-0.003818762；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.0017917304331141748,"AUC":-0.003818762219548577,"UAUC":-0.0013770467629168337,"gAUC":-0.002195215609525958,"NDCG(10)":-0.002792070563524207}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.663663864135742,"mean_ms":12.69030957221985,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.003818762，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 17](#phase-17)。
- **后续假设**：按执行深度区分最终LayerNorm，保留共享主干；见K18。
- **原判读**：保留阶段顺序对照；固定两轮AUC未超过对应父模型
- **证据**：`qformer-loop/runs/r17_base_phase312-s20262027-20260907T073714772329Z/manifest.json`（SHA256前缀 `4aa2644e9c28`）；`qformer-loop/runs/r17_base_phase312-s20262027-20260907T073714772329Z/result.json`（SHA256前缀 `e8ad0007d9bc`）；`qformer-loop/runs/r17_base_phase312-s20262027-20260907T073714772329Z/inference-all_queries.json`（SHA256前缀 `912c291dc8de`）

<a id="k218"></a>
### K218 · `r17_base_phase123-s20262027-20260907T073714793487Z`

- **方法**：{"parent":"r7_small_attention_first","curriculum_order":"123","parameter_delta":0}；设置 {"model":"QFormerCross11Curriculum","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Compare phase order with equal depth counts and fixed evaluation2
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027],"training_compute":{"schedule":"curriculum123","planned_updates":1240,"completed_updates":1240,"by_depth":{"1":{"updates":310,"examples":2506222,"raw_history_length_sum":2344254157,"valid_history_tokens":241543097,"padded_history_positions":250622200},"2":{"updates":620,"examples":5014470,"raw_history_length_sum":4597880089,"valid_history_tokens":483415417,"padded_history_positions":501447000},"3":{"updates":310,"examples":2501852,"raw_history_length_sum":2309665348,"valid_history_tokens":241056780,"padded_history_positions":250185200}},"total_cycle_calls":2480,"example_cycles":20040718,"valid_history_token_cycles":1931544271,"padded_history_position_cycles":2004071800,"evaluation_depth":2,"note":"Raw historical lengths, truncated valid tokens, and padded positions are distinct; cycle exposure is not exact FLOPs"}}。
- **结果**：logloss=0.194553944；AUC=0.866167277；UAUC=0.795142822；gAUC=0.805675287；NDCG(10)=0.418181154。
- **开销**：参数 8115473；训练含验证 437.4275412559509 秒；峰值CUDA 22717237760 bytes；更新 1240。
- **原记录差值**：-0.005066781；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.006487085817191679,"AUC":-0.0050667813049857235,"UAUC":0.004305809028639973,"gAUC":0.0018846245616032098,"NDCG(10)":0.008324997153705938}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.654592037200928,"mean_ms":12.662415170669556,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.005066781，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 17](#phase-17)。
- **后续假设**：按执行深度区分最终LayerNorm，保留共享主干；见K18。
- **原判读**：本轮NDCG最高，但AUC明显下降；保留排序对照
- **证据**：`qformer-loop/runs/r17_base_phase123-s20262027-20260907T073714793487Z/manifest.json`（SHA256前缀 `b22b026e37b1`）；`qformer-loop/runs/r17_base_phase123-s20262027-20260907T073714793487Z/result.json`（SHA256前缀 `3d0c2186bdc6`）；`qformer-loop/runs/r17_base_phase123-s20262027-20260907T073714793487Z/inference-all_queries.json`（SHA256前缀 `7eaf10f186b5`）

<a id="k219"></a>
### K219 · `r17_field512_phase132-s20262027-20260907T073714767736Z`

- **方法**：{"parent":"r15_w512_field","curriculum_order":"132","parameter_delta":0}；设置 {"model":"QFormerCross11Curriculum","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Compare phase order with equal depth counts and fixed evaluation2
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027],"training_compute":{"schedule":"curriculum132","planned_updates":1240,"completed_updates":1240,"by_depth":{"1":{"updates":310,"examples":2506222,"raw_history_length_sum":2344254157,"valid_history_tokens":241543097,"padded_history_positions":250622200},"2":{"updates":620,"examples":5007433,"raw_history_length_sum":4628679979,"valid_history_tokens":482597838,"padded_history_positions":500743300},"3":{"updates":310,"examples":2508889,"raw_history_length_sum":2278865458,"valid_history_tokens":241874359,"padded_history_positions":250888900}},"total_cycle_calls":2480,"example_cycles":20047755,"valid_history_token_cycles":1932361850,"padded_history_position_cycles":2004775500,"evaluation_depth":2,"note":"Raw historical lengths, truncated valid tokens, and padded positions are distinct; cycle exposure is not exact FLOPs"}}。
- **结果**：logloss=0.188534247；AUC=0.870438637；UAUC=0.790164573；gAUC=0.802682984；NDCG(10)=0.410000318。
- **开销**：参数 7852817；训练含验证 439.21698615700006 秒；峰值CUDA 22313009664 bytes；更新 1240。
- **原记录差值**：-0.000614837；对照 `same-seed r15_w512_field`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r15_w512_field`；完整指标差 {"logloss":0.00042327371066280817,"AUC":-0.0006148369677150267,"UAUC":0.0014548904397824192,"gAUC":0.0014941363435879707,"NDCG(10)":0.001490634820706438}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.593088150024414,"mean_ms":12.599486422538757,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000614837，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 17](#phase-17)。
- **后续假设**：按执行深度区分最终LayerNorm，保留共享主干；见K18。
- **原判读**：本轮AUC最高；高于Cross8/Cross11，UAUC/NDCG高于字段512父模型，保留取舍候选
- **证据**：`qformer-loop/runs/r17_field512_phase132-s20262027-20260907T073714767736Z/manifest.json`（SHA256前缀 `ee7bb34ac7c8`）；`qformer-loop/runs/r17_field512_phase132-s20262027-20260907T073714767736Z/result.json`（SHA256前缀 `55b7458e2a6b`）；`qformer-loop/runs/r17_field512_phase132-s20262027-20260907T073714767736Z/inference-all_queries.json`（SHA256前缀 `362bb70c0ee7`）

<a id="k220"></a>
### K220 · `r17_field512_phase312-s20262027-20260907T073714677736Z`

- **方法**：{"parent":"r15_w512_field","curriculum_order":"312","parameter_delta":0}；设置 {"model":"QFormerCross11Curriculum","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Compare phase order with equal depth counts and fixed evaluation2
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027],"training_compute":{"schedule":"curriculum312","planned_updates":1240,"completed_updates":1240,"by_depth":{"1":{"updates":310,"examples":2508889,"raw_history_length_sum":2278865458,"valid_history_tokens":241874359,"padded_history_positions":250888900},"2":{"updates":620,"examples":5007433,"raw_history_length_sum":4628679979,"valid_history_tokens":482597838,"padded_history_positions":500743300},"3":{"updates":310,"examples":2506222,"raw_history_length_sum":2344254157,"valid_history_tokens":241543097,"padded_history_positions":250622200}},"total_cycle_calls":2480,"example_cycles":20042421,"valid_history_token_cycles":1931699326,"padded_history_position_cycles":2004242100,"evaluation_depth":2,"note":"Raw historical lengths, truncated valid tokens, and padded positions are distinct; cycle exposure is not exact FLOPs"}}。
- **结果**：logloss=0.189704315；AUC=0.868432615；UAUC=0.788064698；gAUC=0.800789043；NDCG(10)=0.408117526。
- **开销**：参数 7852817；训练含验证 436.12839875370264 秒；峰值CUDA 22312026624 bytes；更新 1240。
- **原记录差值**：-0.002620858；对照 `same-seed r15_w512_field`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r15_w512_field`；完整指标差 {"logloss":0.0015933415928283956,"AUC":-0.002620858410953497,"UAUC":-0.0006449845001006427,"gAUC":-0.0003998052003109098,"NDCG(10)":-0.00039215679729603936}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.601071834564209,"mean_ms":12.701791214942933,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.002620858，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 17](#phase-17)。
- **后续假设**：按执行深度区分最终LayerNorm，保留共享主干；见K18。
- **原判读**：保留阶段顺序对照；固定两轮AUC未超过对应父模型
- **证据**：`qformer-loop/runs/r17_field512_phase312-s20262027-20260907T073714677736Z/manifest.json`（SHA256前缀 `48649aac8f7d`）；`qformer-loop/runs/r17_field512_phase312-s20262027-20260907T073714677736Z/result.json`（SHA256前缀 `6bed4a57500e`）；`qformer-loop/runs/r17_field512_phase312-s20262027-20260907T073714677736Z/inference-all_queries.json`（SHA256前缀 `b31b53c0ec8a`）

<a id="k221"></a>
### K221 · `r17_field512_phase123-s20262027-20260907T073714793481Z`

- **方法**：{"parent":"r15_w512_field","curriculum_order":"123","parameter_delta":0}；设置 {"model":"QFormerCross11Curriculum","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Compare phase order with equal depth counts and fixed evaluation2
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027],"training_compute":{"schedule":"curriculum123","planned_updates":1240,"completed_updates":1240,"by_depth":{"1":{"updates":310,"examples":2506222,"raw_history_length_sum":2344254157,"valid_history_tokens":241543097,"padded_history_positions":250622200},"2":{"updates":620,"examples":5014470,"raw_history_length_sum":4597880089,"valid_history_tokens":483415417,"padded_history_positions":501447000},"3":{"updates":310,"examples":2501852,"raw_history_length_sum":2309665348,"valid_history_tokens":241056780,"padded_history_positions":250185200}},"total_cycle_calls":2480,"example_cycles":20040718,"valid_history_token_cycles":1931544271,"padded_history_position_cycles":2004071800,"evaluation_depth":2,"note":"Raw historical lengths, truncated valid tokens, and padded positions are distinct; cycle exposure is not exact FLOPs"}}。
- **结果**：logloss=0.189149204；AUC=0.868986127；UAUC=0.792152667；gAUC=0.804322238；NDCG(10)=0.411638543。
- **开销**：参数 7852817；训练含验证 436.2903741002083 秒；峰值CUDA 22313009664 bytes；更新 1240。
- **原记录差值**：-0.002067347；对照 `same-seed r15_w512_field`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r15_w512_field`；完整指标差 {"logloss":0.0010382300510247133,"AUC":-0.0020673465550309356,"UAUC":0.0034429847925591117,"gAUC":0.003133390558840876,"NDCG(10)":0.003128860468266381}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.593488216400146,"mean_ms":12.620984816551209,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.002067347，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 17](#phase-17)。
- **后续假设**：按执行深度区分最终LayerNorm，保留共享主干；见K18。
- **原判读**：保留阶段顺序对照；固定两轮AUC未超过对应父模型
- **证据**：`qformer-loop/runs/r17_field512_phase123-s20262027-20260907T073714793481Z/manifest.json`（SHA256前缀 `2fcc55d8b293`）；`qformer-loop/runs/r17_field512_phase123-s20262027-20260907T073714793481Z/result.json`（SHA256前缀 `187f5d511b95`）；`qformer-loop/runs/r17_field512_phase123-s20262027-20260907T073714793481Z/inference-all_queries.json`（SHA256前缀 `8675d4886d58`）

<a id="k222"></a>
### K222 · `diagnostics-round17-r17_base_phase132`

- **方法**：Query Zero/Shuffle, extra attention off and inference-depth interventions
- **原假设**：Measure Query and inference-depth reliance after variable-depth training
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"training_seed":20262027,"new_training_seed_values":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "rows": 57408,
  "users": 7176,
  "positives": 4472,
  "precision": "bf16",
  "scores": {
    "original": {
      "AUC": 0.8602277570539157,
      "logloss": 0.20343148708343506
    },
    "all_queries": {
      "AUC": 0.8602277570539157,
      "logloss": 0.20343148708343506
    },
    "q3_half": {
      "AUC": 0.8603841294297254,
      "logloss": 0.20346583425998688
    },
    "q0_zero": {
      "AUC": 0.8599410504276538,
      "logloss": 0.20391122996807098
    },
    "q0_shuffle": {
      "AUC": 0.8528878232613831,
      "logloss": 0.2095540165901184
    },
    "q1_zero": {
      "AUC": 0.8592040920645933,
      "logloss": 0.20588314533233643
    },
    "q1_shuffle": {
      "AUC": 0.855190028215798,
      "logloss": 0.20883877575397491
    },
    "q2_zero": {
      "AUC": 0.8600426895149724,
      "logloss": 0.20432649552822113
    },
    "q2_shuffle": {
      "AUC": 0.8562936197738897,
      "logloss": 0.2076442688703537
    },
    "q3_zero": {
      "AUC": 0.8604956067379979,
      "logloss": 0.2037801444530487
    },
    "q3_shuffle": {
      "AUC": 0.8571975131883697,
      "logloss": 0.20790112018585205
    },
    "q4_zero": {
      "AUC": 0.8600659861180464,
      "logloss": 0.2097422331571579
    },
    "q4_shuffle": {
      "AUC": 0.859037226712893,
      "logloss": 0.2051829993724823
    },
    "q5_zero": {
      "AUC": 0.8598848787904143,
      "logloss": 0.20327483117580414
    },
    "q5_shuffle": {
      "AUC": 0.8589879933658708,
      "logloss": 0.2043500542640686
    },
    "q6_zero": {
      "AUC": 0.8603397602782501,
      "logloss": 0.2080569714307785
    },
    "q6_shuffle": {
      "AUC": 0.8587090022872998,
      "logloss": 0.205308735370636
    },
    "q7_zero": {
      "AUC": 0.8604148099787964,
      "logloss": 0.20416714251041412
    },
    "q7_shuffle": {
      "AUC": 0.8569570808392379,
      "logloss": 0.2083635777235031
    },
    "inference_placement_steps1": {
      "AUC": 0.8531043443826454,
      "logloss": 0.22429870069026947
    },
    "inference_placement_steps3": {
      "AUC": 0.8514401136296357,
      "logloss": 0.21005617082118988
    },
    "extra_attention_off": {
      "AUC": 0.8559103790367035,
      "logloss": 0.20787565410137177
    }
  },
  "checkpoint_unchanged": true,
  "note": "subset diagnostics; within-block shuffle; blockwise loss means in toolkit differ from pooled full-validation metrics"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 17](#phase-17)。
- **后续假设**：按执行深度区分最终LayerNorm，保留共享主干；见K18。
- **证据**：`qformer-loop/reports/round17/r17_base_phase132/summary.json`（SHA256前缀 `8ab846e0d27f`）；`qformer-loop/reports/round17/r17_base_phase132/toolkit-hooks/hook_report.json`（SHA256前缀 `754a099db5d9`）；`qformer-loop/reports/round17/r17_base_phase132/toolkit-perturbations/perturbation_report.json`（SHA256前缀 `f4df1e54aa62`）

<a id="k223"></a>
### K223 · `diagnostics-round17-r17_field512_phase132`

- **方法**：Query Zero/Shuffle, extra attention off and inference-depth interventions
- **原假设**：Measure Query and inference-depth reliance after variable-depth training
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"training_seed":20262027,"new_training_seed_values":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "rows": 57408,
  "users": 7176,
  "positives": 4472,
  "precision": "bf16",
  "scores": {
    "original": {
      "AUC": 0.8610073251785733,
      "logloss": 0.20264498889446259
    },
    "all_queries": {
      "AUC": 0.8610073251785733,
      "logloss": 0.20264498889446259
    },
    "q3_half": {
      "AUC": 0.8611702704491033,
      "logloss": 0.20253239572048187
    },
    "q0_zero": {
      "AUC": 0.8608681876423901,
      "logloss": 0.20472091436386108
    },
    "q0_shuffle": {
      "AUC": 0.8592600060240833,
      "logloss": 0.2057776302099228
    },
    "q1_zero": {
      "AUC": 0.8611194762507965,
      "logloss": 0.20289407670497894
    },
    "q1_shuffle": {
      "AUC": 0.8561592936304359,
      "logloss": 0.20769299566745758
    },
    "q2_zero": {
      "AUC": 0.8591099847711605,
      "logloss": 0.20440909266471863
    },
    "q2_shuffle": {
      "AUC": 0.8573992347359474,
      "logloss": 0.2065279632806778
    },
    "q3_zero": {
      "AUC": 0.8611633089256463,
      "logloss": 0.20437732338905334
    },
    "q3_shuffle": {
      "AUC": 0.8577126426909546,
      "logloss": 0.20683051645755768
    },
    "q4_zero": {
      "AUC": 0.860738282150816,
      "logloss": 0.20356294512748718
    },
    "q4_shuffle": {
      "AUC": 0.856401491705784,
      "logloss": 0.207889124751091
    },
    "q5_zero": {
      "AUC": 0.860089565744222,
      "logloss": 0.20416580140590668
    },
    "q5_shuffle": {
      "AUC": 0.8583405273299949,
      "logloss": 0.2056564837694168
    },
    "q6_zero": {
      "AUC": 0.8609153764643193,
      "logloss": 0.20501092076301575
    },
    "q6_shuffle": {
      "AUC": 0.8583622609696713,
      "logloss": 0.20648853480815887
    },
    "q7_zero": {
      "AUC": 0.8601590964942849,
      "logloss": 0.2037009596824646
    },
    "q7_shuffle": {
      "AUC": 0.8589030230719755,
      "logloss": 0.20533965528011322
    },
    "inference_placement_steps1": {
      "AUC": 0.8544516589614543,
      "logloss": 0.211549311876297
    },
    "inference_placement_steps3": {
      "AUC": 0.852919186022856,
      "logloss": 0.20969410240650177
    },
    "extra_attention_off": {
      "AUC": 0.8569848128789807,
      "logloss": 0.21007055044174194
    }
  },
  "checkpoint_unchanged": true,
  "note": "subset diagnostics; within-block shuffle; blockwise loss means in toolkit differ from pooled full-validation metrics"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 17](#phase-17)。
- **后续假设**：按执行深度区分最终LayerNorm，保留共享主干；见K18。
- **证据**：`qformer-loop/reports/round17/r17_field512_phase132/summary.json`（SHA256前缀 `7a4192e163eb`）；`qformer-loop/reports/round17/r17_field512_phase132/toolkit-hooks/hook_report.json`（SHA256前缀 `0ce3704c8aee`）；`qformer-loop/reports/round17/r17_field512_phase132/toolkit-perturbations/perturbation_report.json`（SHA256前缀 `b5a570b99c16`）

<a id="phase-18"></a>
## 阶段 18：深度专属最终norm

**方法**：两容量×phase132/mixed1232，克隆初始化独立最终norm。

**归因与边界**：4组均低于固定两轮父模型；只在原容量部分安排恢复AUC，不能概括为解决归一化冲突。

**历史后续假设**：固定两轮下区分block内部轮次norm；K19仅准备、后因淘宝迁移延后，未训练。

<a id="k224"></a>
### K224 · `round18-launch-yaml-r18_base_norm_phase132`

- **方法**：Configuration serialization failed before training
- **原假设**：Launch depth-specific normalization trial
- **状态／预算**：原记录 `crash`；{"training_updates":0,"training_seed":20262027}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "decision": "crash",
  "validity": {
    "valid": false,
    "violations": [
      "YAML duplicate anchor"
    ],
    "notes": [
      "No model constructed; original seed retried after parse/semantic checks"
    ]
  },
  "falsification_result": null,
  "note": "本条未保存独立性能指标；保留运行状态与证据位置，不补造数值。"
}
```

</details>

- **归因**：这是执行或有效性问题；本条不能用作结构性能反证。修复后完整训练需用独立运行ID核对。 本阶段证据解释见[阶段 18](#phase-18)。
- **后续假设**：固定两轮下区分block内部轮次norm；K19仅准备、后因淘宝迁移延后，未训练。
- **失败后续门槛**：先复核同配置/初始函数/有效梯度及完整预算，再判断科学假设；不把同一次修复重复算成独立支持。
- **证据**：`qformer-loop/reports/round18/training-config-failure/r18_base_norm_phase132-s20262027.log`；`qformer-loop/reports/round18/launch-failure.json`（SHA256前缀 `5cad8d873f21`）

<a id="k225"></a>
### K225 · `round18-launch-yaml-r18_base_norm_mixed1232`

- **方法**：Configuration serialization failed before training
- **原假设**：Launch depth-specific normalization trial
- **状态／预算**：原记录 `crash`；{"training_updates":0,"training_seed":20262027}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "decision": "crash",
  "validity": {
    "valid": false,
    "violations": [
      "YAML duplicate anchor"
    ],
    "notes": [
      "No model constructed; original seed retried after parse/semantic checks"
    ]
  },
  "falsification_result": null,
  "note": "本条未保存独立性能指标；保留运行状态与证据位置，不补造数值。"
}
```

</details>

- **归因**：这是执行或有效性问题；本条不能用作结构性能反证。修复后完整训练需用独立运行ID核对。 本阶段证据解释见[阶段 18](#phase-18)。
- **后续假设**：固定两轮下区分block内部轮次norm；K19仅准备、后因淘宝迁移延后，未训练。
- **失败后续门槛**：先复核同配置/初始函数/有效梯度及完整预算，再判断科学假设；不把同一次修复重复算成独立支持。
- **证据**：`qformer-loop/reports/round18/training-config-failure/r18_base_norm_mixed1232-s20262027.log`；`qformer-loop/reports/round18/launch-failure.json`（SHA256前缀 `5cad8d873f21`）

<a id="k226"></a>
### K226 · `round18-launch-yaml-r18_field512_norm_phase132`

- **方法**：Configuration serialization failed before training
- **原假设**：Launch depth-specific normalization trial
- **状态／预算**：原记录 `crash`；{"training_updates":0,"training_seed":20262027}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "decision": "crash",
  "validity": {
    "valid": false,
    "violations": [
      "YAML duplicate anchor"
    ],
    "notes": [
      "No model constructed; original seed retried after parse/semantic checks"
    ]
  },
  "falsification_result": null,
  "note": "本条未保存独立性能指标；保留运行状态与证据位置，不补造数值。"
}
```

</details>

- **归因**：这是执行或有效性问题；本条不能用作结构性能反证。修复后完整训练需用独立运行ID核对。 本阶段证据解释见[阶段 18](#phase-18)。
- **后续假设**：固定两轮下区分block内部轮次norm；K19仅准备、后因淘宝迁移延后，未训练。
- **失败后续门槛**：先复核同配置/初始函数/有效梯度及完整预算，再判断科学假设；不把同一次修复重复算成独立支持。
- **证据**：`qformer-loop/reports/round18/training-config-failure/r18_field512_norm_phase132-s20262027.log`；`qformer-loop/reports/round18/launch-failure.json`（SHA256前缀 `5cad8d873f21`）

<a id="k227"></a>
### K227 · `round18-launch-yaml-r18_field512_norm_mixed1232`

- **方法**：Configuration serialization failed before training
- **原假设**：Launch depth-specific normalization trial
- **状态／预算**：原记录 `crash`；{"training_updates":0,"training_seed":20262027}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "decision": "crash",
  "validity": {
    "valid": false,
    "violations": [
      "YAML duplicate anchor"
    ],
    "notes": [
      "No model constructed; original seed retried after parse/semantic checks"
    ]
  },
  "falsification_result": null,
  "note": "本条未保存独立性能指标；保留运行状态与证据位置，不补造数值。"
}
```

</details>

- **归因**：这是执行或有效性问题；本条不能用作结构性能反证。修复后完整训练需用独立运行ID核对。 本阶段证据解释见[阶段 18](#phase-18)。
- **后续假设**：固定两轮下区分block内部轮次norm；K19仅准备、后因淘宝迁移延后，未训练。
- **失败后续门槛**：先复核同配置/初始函数/有效梯度及完整预算，再判断科学假设；不把同一次修复重复算成独立支持。
- **证据**：`qformer-loop/reports/round18/training-config-failure/r18_field512_norm_mixed1232-s20262027.log`；`qformer-loop/reports/round18/launch-failure.json`（SHA256前缀 `5cad8d873f21`）

<a id="k228"></a>
### K228 · `r18_base_norm_phase132-s20262027-20260907T080802807230Z`

- **方法**：{"parent":"r7_small_attention_first","matched_schedule_parent":"r17_base_phase132","depth_norm_schedule":"phase132","parameter_delta":1024}；设置 {"model":"QFormerCross11DepthNorm","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Separate final norm by executed depth while retaining shared recurrent core and tower
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027],"training_compute":{"schedule":"phase132","planned_updates":1240,"completed_updates":1240,"by_depth":{"1":{"updates":310,"examples":2506222,"raw_history_length_sum":2344254157,"valid_history_tokens":241543097,"padded_history_positions":250622200},"2":{"updates":620,"examples":5007433,"raw_history_length_sum":4628679979,"valid_history_tokens":482597838,"padded_history_positions":500743300},"3":{"updates":310,"examples":2508889,"raw_history_length_sum":2278865458,"valid_history_tokens":241874359,"padded_history_positions":250888900}},"total_cycle_calls":2480,"example_cycles":20047755,"valid_history_token_cycles":1932361850,"padded_history_position_cycles":2004775500,"evaluation_depth":2,"note":"Raw historical lengths, truncated valid tokens, and padded positions are distinct; cycle exposure is not exact FLOPs","final_norm":"independent by executed depth; depth2 used for validation"}}。
- **结果**：logloss=0.188665791；AUC=0.870212571；UAUC=0.791300036；gAUC=0.803572726；NDCG(10)=0.410909686。
- **开销**：参数 8116497；训练含验证 437.0429310053587 秒；峰值CUDA 22717245952 bytes；更新 1240。
- **原记录差值**：-0.001021488；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.0005989330157633588,"AUC":-0.0010214878607671674,"UAUC":0.0004630236620950523,"gAUC":-0.00021793666145697443,"NDCG(10)":0.001053529078311466}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.711423873901367,"mean_ms":12.720558428764344,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.001021488，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 18](#phase-18)。
- **后续假设**：固定两轮下区分block内部轮次norm；K19仅准备、后因淘宝迁移延后，未训练。
- **原判读**：本轮AUC最佳；超过Cross8/Cross11且UAUC/NDCG略高于r7，保留取舍候选；未替换AUC首选
- **证据**：`qformer-loop/runs/r18_base_norm_phase132-s20262027-20260907T080802807230Z/manifest.json`（SHA256前缀 `71179789d429`）；`qformer-loop/runs/r18_base_norm_phase132-s20262027-20260907T080802807230Z/result.json`（SHA256前缀 `e764086e0aff`）；`qformer-loop/runs/r18_base_norm_phase132-s20262027-20260907T080802807230Z/inference-all_queries.json`（SHA256前缀 `35497e52529f`）

<a id="k229"></a>
### K229 · `r18_base_norm_mixed1232-s20262027-20260907T080802773837Z`

- **方法**：{"parent":"r7_small_attention_first","matched_schedule_parent":"r16_base_mixed1232","depth_norm_schedule":"mixed1232","parameter_delta":1024}；设置 {"model":"QFormerCross11DepthNorm","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Separate final norm by executed depth while retaining shared recurrent core and tower
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027],"training_compute":{"schedule":"mixed1232","planned_updates":1240,"completed_updates":1240,"by_depth":{"1":{"updates":310,"examples":2494497,"raw_history_length_sum":2302794948,"valid_history_tokens":240439907,"padded_history_positions":249449700},"2":{"updates":620,"examples":5026986,"raw_history_length_sum":4640211178,"valid_history_tokens":484502513,"padded_history_positions":502698600},"3":{"updates":310,"examples":2501061,"raw_history_length_sum":2308793468,"valid_history_tokens":241072874,"padded_history_positions":250106100}},"total_cycle_calls":2480,"example_cycles":20051652,"valid_history_token_cycles":1932663555,"padded_history_position_cycles":2005165200,"evaluation_depth":2,"note":"Raw historical lengths, truncated valid tokens, and padded positions are distinct; cycle exposure is not exact FLOPs","final_norm":"independent by executed depth; depth2 used for validation"}}。
- **结果**：logloss=0.189517064；AUC=0.868730009；UAUC=0.788384109；gAUC=0.800872259；NDCG(10)=0.402111281。
- **开销**：参数 8116497；训练含验证 438.7144851014018 秒；峰值CUDA 22717250048 bytes；更新 1240。
- **原记录差值**：-0.002504050；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.0014502054991392954,"AUC":-0.0025040496317718652,"UAUC":-0.0024529040643633815,"gAUC":-0.0029184028947824014,"NDCG(10)":-0.007744875979167232}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.715456008911133,"mean_ms":12.719017577171325,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.002504050，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 18](#phase-18)。
- **后续假设**：固定两轮下区分block内部轮次norm；K19仅准备、后因淘宝迁移延后，未训练。
- **原判读**：相对共享norm AUC仅+0.000062，Logloss和排序未改善，保留近零对照
- **证据**：`qformer-loop/runs/r18_base_norm_mixed1232-s20262027-20260907T080802773837Z/manifest.json`（SHA256前缀 `b0dde16ca6ce`）；`qformer-loop/runs/r18_base_norm_mixed1232-s20262027-20260907T080802773837Z/result.json`（SHA256前缀 `12e5385db623`）；`qformer-loop/runs/r18_base_norm_mixed1232-s20262027-20260907T080802773837Z/inference-all_queries.json`（SHA256前缀 `f150ba72a3a0`）

<a id="k230"></a>
### K230 · `r18_field512_norm_phase132-s20262027-20260907T080802844563Z`

- **方法**：{"parent":"r15_w512_field","matched_schedule_parent":"r17_field512_phase132","depth_norm_schedule":"phase132","parameter_delta":1024}；设置 {"model":"QFormerCross11DepthNorm","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Separate final norm by executed depth while retaining shared recurrent core and tower
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027],"training_compute":{"schedule":"phase132","planned_updates":1240,"completed_updates":1240,"by_depth":{"1":{"updates":310,"examples":2506222,"raw_history_length_sum":2344254157,"valid_history_tokens":241543097,"padded_history_positions":250622200},"2":{"updates":620,"examples":5007433,"raw_history_length_sum":4628679979,"valid_history_tokens":482597838,"padded_history_positions":500743300},"3":{"updates":310,"examples":2508889,"raw_history_length_sum":2278865458,"valid_history_tokens":241874359,"padded_history_positions":250888900}},"total_cycle_calls":2480,"example_cycles":20047755,"valid_history_token_cycles":1932361850,"padded_history_position_cycles":2004775500,"evaluation_depth":2,"note":"Raw historical lengths, truncated valid tokens, and padded positions are distinct; cycle exposure is not exact FLOPs","final_norm":"independent by executed depth; depth2 used for validation"}}。
- **结果**：logloss=0.189286233；AUC=0.869173143；UAUC=0.788823060；gAUC=0.801591598；NDCG(10)=0.405762483。
- **开销**：参数 7853841；训练含验证 432.95935529470444 秒；峰值CUDA 22313017856 bytes；更新 1240。
- **原记录差值**：-0.001880330；对照 `same-seed r15_w512_field`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r15_w512_field`；完整指标差 {"logloss":0.001175259463482875,"AUC":-0.001880330320208956,"UAUC":0.0001133775935862058,"gAUC":0.00040275004251222324,"NDCG(10)":-0.0027472001999913065}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.655136108398438,"mean_ms":12.680660057067872,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.001880330，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 18](#phase-18)。
- **后续假设**：固定两轮下区分block内部轮次norm；K19仅准备、后因淘宝迁移延后，未训练。
- **原判读**：保留负对照；未超过固定两轮父模型
- **证据**：`qformer-loop/runs/r18_field512_norm_phase132-s20262027-20260907T080802844563Z/manifest.json`（SHA256前缀 `f0ad9318417d`）；`qformer-loop/runs/r18_field512_norm_phase132-s20262027-20260907T080802844563Z/result.json`（SHA256前缀 `d6fa124590dc`）；`qformer-loop/runs/r18_field512_norm_phase132-s20262027-20260907T080802844563Z/inference-all_queries.json`（SHA256前缀 `4445ec9ea5a6`）

<a id="k231"></a>
### K231 · `r18_field512_norm_mixed1232-s20262027-20260907T080802738804Z`

- **方法**：{"parent":"r15_w512_field","matched_schedule_parent":"r16_field512_mixed1232","depth_norm_schedule":"mixed1232","parameter_delta":1024}；设置 {"model":"QFormerCross11DepthNorm","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Separate final norm by executed depth while retaining shared recurrent core and tower
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027],"training_compute":{"schedule":"mixed1232","planned_updates":1240,"completed_updates":1240,"by_depth":{"1":{"updates":310,"examples":2494497,"raw_history_length_sum":2302794948,"valid_history_tokens":240439907,"padded_history_positions":249449700},"2":{"updates":620,"examples":5026986,"raw_history_length_sum":4640211178,"valid_history_tokens":484502513,"padded_history_positions":502698600},"3":{"updates":310,"examples":2501061,"raw_history_length_sum":2308793468,"valid_history_tokens":241072874,"padded_history_positions":250106100}},"total_cycle_calls":2480,"example_cycles":20051652,"valid_history_token_cycles":1932663555,"padded_history_position_cycles":2005165200,"evaluation_depth":2,"note":"Raw historical lengths, truncated valid tokens, and padded positions are distinct; cycle exposure is not exact FLOPs","final_norm":"independent by executed depth; depth2 used for validation"}}。
- **结果**：logloss=0.190201963；AUC=0.866747628；UAUC=0.789883262；gAUC=0.802039049；NDCG(10)=0.404579003。
- **开销**：参数 7853841；训练含验证 436.33506136387587 秒；峰值CUDA 22313021952 bytes；更新 1240。
- **原记录差值**：-0.004305846；对照 `same-seed r15_w512_field`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r15_w512_field`；完整指标差 {"logloss":0.002090989359155998,"AUC":-0.0043058460909630725,"UAUC":0.0011735793163321073,"gAUC":0.0008502015513812688,"NDCG(10)":-0.003930680048974511}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.661375999450684,"mean_ms":12.755928039550781,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.004305846，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 18](#phase-18)。
- **后续假设**：固定两轮下区分block内部轮次norm；K19仅准备、后因淘宝迁移延后，未训练。
- **原判读**：保留负对照；未超过固定两轮父模型
- **证据**：`qformer-loop/runs/r18_field512_norm_mixed1232-s20262027-20260907T080802738804Z/manifest.json`（SHA256前缀 `516be79024d6`）；`qformer-loop/runs/r18_field512_norm_mixed1232-s20262027-20260907T080802738804Z/result.json`（SHA256前缀 `c7a274ce4d7b`）；`qformer-loop/runs/r18_field512_norm_mixed1232-s20262027-20260907T080802738804Z/inference-all_queries.json`（SHA256前缀 `8baa4ebf02a6`）

<a id="k232"></a>
### K232 · `diagnostics-round18-r18_base_norm_phase132`

- **方法**：Query Zero/Shuffle, extra attention off and inference-depth interventions
- **原假设**：Measure Query and inference-depth reliance after variable-depth training
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"training_seed":20262027,"new_training_seed_values":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "rows": 57408,
  "users": 7176,
  "positives": 4472,
  "precision": "bf16",
  "scores": {
    "original": {
      "AUC": 0.8623518728897459,
      "logloss": 0.20221221446990967
    },
    "all_queries": {
      "AUC": 0.8623518728897459,
      "logloss": 0.20221221446990967
    },
    "q3_half": {
      "AUC": 0.8622539680176797,
      "logloss": 0.2023991048336029
    },
    "q0_zero": {
      "AUC": 0.8617360885443602,
      "logloss": 0.20348133146762848
    },
    "q0_shuffle": {
      "AUC": 0.855682893938419,
      "logloss": 0.20836827158927917
    },
    "q1_zero": {
      "AUC": 0.8621197707131006,
      "logloss": 0.20275035500526428
    },
    "q1_shuffle": {
      "AUC": 0.859406100859498,
      "logloss": 0.20606397092342377
    },
    "q2_zero": {
      "AUC": 0.862222218739583,
      "logloss": 0.20440012216567993
    },
    "q2_shuffle": {
      "AUC": 0.8603895469987993,
      "logloss": 0.2048659771680832
    },
    "q3_zero": {
      "AUC": 0.8619762970095458,
      "logloss": 0.20304740965366364
    },
    "q3_shuffle": {
      "AUC": 0.8575929323673802,
      "logloss": 0.2073456048965454
    },
    "q4_zero": {
      "AUC": 0.8610951890668667,
      "logloss": 0.20923297107219696
    },
    "q4_shuffle": {
      "AUC": 0.8593377951348008,
      "logloss": 0.20505784451961517
    },
    "q5_zero": {
      "AUC": 0.861922537405009,
      "logloss": 0.20199576020240784
    },
    "q5_shuffle": {
      "AUC": 0.8613309451984816,
      "logloss": 0.20299847424030304
    },
    "q6_zero": {
      "AUC": 0.8616609839288839,
      "logloss": 0.2030770480632782
    },
    "q6_shuffle": {
      "AUC": 0.859964232976642,
      "logloss": 0.20509976148605347
    },
    "q7_zero": {
      "AUC": 0.8620441338452238,
      "logloss": 0.20407892763614655
    },
    "q7_shuffle": {
      "AUC": 0.8575864312844916,
      "logloss": 0.2078067660331726
    },
    "inference_placement_steps1": {
      "AUC": 0.8511374204223522,
      "logloss": 0.2186541110277176
    },
    "inference_placement_steps3": {
      "AUC": 0.8523107940719181,
      "logloss": 0.2104348987340927
    },
    "extra_attention_off": {
      "AUC": 0.8560716937562298,
      "logloss": 0.2068641632795334
    }
  },
  "checkpoint_unchanged": true,
  "note": "subset diagnostics; within-block shuffle; blockwise loss means in toolkit differ from pooled full-validation metrics"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 18](#phase-18)。
- **后续假设**：固定两轮下区分block内部轮次norm；K19仅准备、后因淘宝迁移延后，未训练。
- **证据**：`qformer-loop/reports/round18/r18_base_norm_phase132/summary.json`（SHA256前缀 `54a32c34ff5a`）；`qformer-loop/reports/round18/r18_base_norm_phase132/toolkit-hooks/hook_report.json`（SHA256前缀 `a3696da620a8`）；`qformer-loop/reports/round18/r18_base_norm_phase132/toolkit-perturbations/perturbation_report.json`（SHA256前缀 `ad7c0719bc1f`）

<a id="k233"></a>
### K233 · `diagnostics-round18-r18_field512_norm_phase132`

- **方法**：Query Zero/Shuffle, extra attention off and inference-depth interventions
- **原假设**：Measure Query and inference-depth reliance after variable-depth training
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"training_seed":20262027,"new_training_seed_values":[]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "rows": 57408,
  "users": 7176,
  "positives": 4472,
  "precision": "bf16",
  "scores": {
    "original": {
      "AUC": 0.8597392950862729,
      "logloss": 0.20360484719276428
    },
    "all_queries": {
      "AUC": 0.8597392950862729,
      "logloss": 0.20360484719276428
    },
    "q3_half": {
      "AUC": 0.8595438021590455,
      "logloss": 0.20408987998962402
    },
    "q0_zero": {
      "AUC": 0.8602126110937486,
      "logloss": 0.20430925488471985
    },
    "q0_shuffle": {
      "AUC": 0.8580880094719975,
      "logloss": 0.20615175366401672
    },
    "q1_zero": {
      "AUC": 0.8595946850660858,
      "logloss": 0.20475785434246063
    },
    "q1_shuffle": {
      "AUC": 0.855819796859366,
      "logloss": 0.20747505128383636
    },
    "q2_zero": {
      "AUC": 0.8585532825543141,
      "logloss": 0.2045438587665558
    },
    "q2_shuffle": {
      "AUC": 0.8582889622105527,
      "logloss": 0.2051324099302292
    },
    "q3_zero": {
      "AUC": 0.8588514685130969,
      "logloss": 0.20638053119182587
    },
    "q3_shuffle": {
      "AUC": 0.8555785344499437,
      "logloss": 0.20784029364585876
    },
    "q4_zero": {
      "AUC": 0.8594500137946305,
      "logloss": 0.2063281238079071
    },
    "q4_shuffle": {
      "AUC": 0.8580662716080958,
      "logloss": 0.20597516000270844
    },
    "q5_zero": {
      "AUC": 0.859392921276254,
      "logloss": 0.20436029136180878
    },
    "q5_shuffle": {
      "AUC": 0.855785071614476,
      "logloss": 0.20682452619075775
    },
    "q6_zero": {
      "AUC": 0.8598720477057658,
      "logloss": 0.20569711923599243
    },
    "q6_shuffle": {
      "AUC": 0.8560962196088949,
      "logloss": 0.20748130977153778
    },
    "q7_zero": {
      "AUC": 0.8596903173048874,
      "logloss": 0.20480433106422424
    },
    "q7_shuffle": {
      "AUC": 0.8580508510732776,
      "logloss": 0.20604868233203888
    },
    "inference_placement_steps1": {
      "AUC": 0.8564938290487748,
      "logloss": 0.20781712234020233
    },
    "inference_placement_steps3": {
      "AUC": 0.8506169198171728,
      "logloss": 0.21023893356323242
    },
    "extra_attention_off": {
      "AUC": 0.8554421912388618,
      "logloss": 0.20962511003017426
    }
  },
  "checkpoint_unchanged": true,
  "note": "subset diagnostics; within-block shuffle; blockwise loss means in toolkit differ from pooled full-validation metrics"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 18](#phase-18)。
- **后续假设**：固定两轮下区分block内部轮次norm；K19仅准备、后因淘宝迁移延后，未训练。
- **证据**：`qformer-loop/reports/round18/r18_field512_norm_phase132/summary.json`（SHA256前缀 `de7499d61467`）；`qformer-loop/reports/round18/r18_field512_norm_phase132/toolkit-hooks/hook_report.json`（SHA256前缀 `9b2deda4233d`）；`qformer-loop/reports/round18/r18_field512_norm_phase132/toolkit-perturbations/perturbation_report.json`（SHA256前缀 `6498517bda81`）

## 跨实验归因与可证伪条件

以下保留已有lesson的限定范围与状态；`supported`是当时收录状态，不自动等于本报告完成了独立确认。历史要求增加seed的设想不作为当前执行授权；当前仍遵守不增加训练seed的约束。

| Lesson ID | 归因记录 | 状态 | 可证伪条件／后续假设 |
|---|---|---|---|
| round2-query-route-mismatch | 在本 KuaiRec 协议和已训练的 Cross11/middle2 checkpoint 上，恢复训练时的全部 Query 改善三个 seed 的平均 AUC 和所有 seed 的 Logloss；单 seed AUC 不保证改善。 | supported | 在相同标签/切分的独立 seed 或更长训练预算下，统一路由不再改善平均 AUC/Logloss。 |
| round2-zero-gradient-routing | 被固定路由置零的 Query 的零梯度与零 Shuffle 效应不能作为删除模块的证据。 | supported | 如果在未屏蔽且存在有效监督的路由下仍无敏感性，应重新检验。 |
| round3-field-feedback-candidate | 在固定 KuaiRec 一轮训练中，历史后字段回读的三 seed 平均 AUC 接近 Cross8 和原 middle2；相对原 middle2 少一次历史块调用，固定批量前向更快，可作为结构优化候选。 | supported | 扩展 seed/预算后明显低于基线，或端到端测量不保留成本优势。 |
| round3-field-recurrence-variance | 字段末层循环在本次三个 seed 中 AUC 波动大，不能只看 seed=20262028 的较高结果。 | provisional | 增加 seed 或控制用户/目标接口后波动消失。 |
| round4-context-feedback | 同协议三 seed 中，历史后只回读用户/上下文较全部字段反馈改善平均 AUC 和 Logloss；初始字段编码仍读取全部特征。 | provisional | 独立 seed、固定计算预算或更长训练预算下平均 AUC 收益消失。 |
| round5-single-history-recurrence | 本协议一轮训练下，保留两字段块和一个共享历史块的 compact_seq1 在三个 seed 均超过 Cross8 AUC，参数略少且固定 batch 延迟接近。 | supported | 独立 seed、更长预算或固定 GPU 时间比较时收益消失。 |
| round5-extra-attention-depth-interaction | 本轮额外共享 Query self-attention 在三历史块和两历史块结构提高平均 AUC，在单历史块结构降低平均 AUC，收益依赖保留的结构。 | provisional | 独立 seed 或更多深度/位置对照不再出现这种交互差异。 |
| round6-terminal-context-feedback | 固定 seed 20262027 下，紧凑历史循环完成后回读上下文较中途回读提高 AUC，但 Logloss/UAUC/NDCG 略降；两种位置均应保留指标取舍。 | provisional | 固定 seed 的结构控制不再保留收益，或更改预算后收益消失。 |
| round6-compact-post-history-attention | 固定 seed 下，1 字段/1 历史块循环加历史后共享 attention 可在 AUC 接近父模型时减少参数/延迟，并改善 Logloss。 | provisional | 固定预算的进一步结构控制下质量或实际成本优势消失。 |
| round6-sharing-and-cycle-controls | 同一个已有 seed 中，去掉额外循环或改为独立历史权重均未超过父模型；从头训练三次循环也低于两次。 | provisional | 后续固定 seed 的独立结构或预算对照改变这些关系。 |
| round7-first-pass-attention | 固定已有 seed 下，1 字段/1 历史块两次循环只保留第一次额外 attention，较两次 attention 提高 AUC/Logloss 并减少前向耗时；只保留最后一次则降低 AUC。 | provisional | 固定 seed 的进一步结构控制或训练预算变化改变该关系。 |
| round7-norm-location | 每次历史读取后、额外 attention 前应用序列 norm 改善小模型 AUC/Logloss；将 norm 移到末尾字段反馈之前未提高 terminal 父模型 AUC。 | provisional | 固定 seed 的进一步结构控制或训练预算变化改变该关系。 |
| round7-memory-kv | 在本轮固定 seed 的四个对照中，额外字段 K/V 拼接初始 Query 或首次字段编码均未提高直接父模型 AUC，且增加少量前向计算。 | provisional | 固定 seed 的进一步结构控制或训练预算变化改变该关系。 |
| round8-trained-cycle-depth | 同参数、同 seed、同更新预算从头训练时，当前首次 attention 小模型两次共享循环 AUC 高于一次和三次；增加到三次退化。 | provisional | 固定 seed 下更改回读位置、独立权重或训练预算后关系改变。 |
| round8-norm-scale-interaction | 仅首次 attention 的父模型上，post-norm 残差减半在原接口和每轮序列 norm 两种结构均降低 AUC；pre-norm 后减半优于全强度 pre-norm，但仍未超过父模型。 | provisional | 固定 seed 下更改回读位置、独立权重或训练预算后关系改变。 |
| round8-ranking-cost-frontier | 一次 N S A 前向在本固定 seed 达到接近 Cross8/Cross11 的 AUC，UAUC/NDCG 高于两次循环父模型，前向耗时减少约三分之一，可作为低成本排序候选。 | provisional | 固定 seed 下更改回读位置、独立权重或训练预算后关系改变。 |
| round9-context-feedback-position | 当前首次 attention 小模型中，去掉第二次上下文回读、替换为历史回读，或将它提前/后移，均未超过原交替结构完整训练后的 AUC。 | provisional | 固定 seed 下修改回读子层或预算后关系改变。 |
| round9-function-matched-history-sharing | 从相同初始历史块复制并独立训练第二次读取，AUC 比共享版低 0.000461，且增加 1,185,536 参数；本固定预算下继续保留共享。 | provisional | 固定 seed 下修改回读子层或预算后关系改变。 |
| round9-interface-sharing | 两次字段 norm 独立未超过共享父模型；历史/双接口分轮 norm 均未超过归一化位置匹配的 r8_first_midnorm AUC。 | provisional | 固定 seed 下修改回读子层或预算后关系改变。 |
| round10-selective-repeated-sublayers | 首次字段/历史读取保持完整、第二次精简 self-attention 或 FFN 的八种结构，均未超过完整共享 block 父模型 AUC。 | provisional | 保留被跳过的 norm 或引入条件化的共享更新后，同预算关系改变。 |
| round10-selective-cost-quality | 选择性执行重复子层可降低固定 batch 前向耗时，历史 CrossValue+FFN 方案 AUC 接近 Cross11，保留为质量/成本候选。 | provisional | 保留被跳过的 norm 或引入条件化的共享更新后，同预算关系改变。 |
| round11-neutral-repeat-conditioning | 八个初始函数兼容的轮次向量/Query门控变体未超过父模型；历史向量接近父模型但无主要指标优势。 | provisional | 改变 block 内信息流、条件作用点或训练预算后关系改变。 |
| round12-full-sublayer-order | 完整循环 block 的八种子层重排均未超过标准 Self/Cross/FFN 父模型；两轮均 CAF 的 AUC 高于仅重复 CAF，但不能推出全部 guardrail 更优。 | provisional | 改变历史源编码、循环更新接口或预算后关系改变。 |
| round13-history-source-update | Query反馈/历史self-attention的八种历史更新结构未刷新AUC；两轮之间零初始Query反馈有UAUC取舍，历史self-attention warm有NDCG取舍。 | provisional | 改变共享FFN、历史更新方式或训练预算后关系改变。 |
| round14-gated-activation-budget | 同初始化及宽度的门控对照中，SwiGLU AUC在三个位置均高于GEGLU；八组门控/SiLU对照未刷新r7，双位置SwiGLU有UAUC/NDCG取舍。 | provisional | 只改普通FFN宽度、初始化或训练预算后关系改变。 |
| round15-plain-width-capacity | 字段GELU FFN512以更少参数达到接近r7的AUC；同预算加宽未超过它，宽度与AUC关系不单调。 | provisional | 可变训练深度、初始化方式或训练预算变化后关系改变。 |
| round16-variable-depth-ranking-calibration | 可变训练深度固定两轮验证未提升AUC；先1后3课程创造较高用户排序指标但全局AUC/Logloss恶化，后续需检验结束深度与验证深度对齐。 | provisional | 课程最终在两轮训练结束后，AUC和排序的关系改变。 |
| round17-phase-order-ending-depth | 相同深度更新次数下，phase132在两类父模型上均比phase123/312的AUC高，但都未超过固定两轮父模型；阶段顺序不能简化为仅结束深度效应。 | provisional | 按执行深度区分最终归一化后，同预算关系改变。 |
| round18-depth-final-norm-not-general | 按执行深度独立最终LayerNorm，仅原容量phase132出现较明显AUC恢复，field512两种安排均下降，四组均低于固定两轮父模型。 | provisional | 固定两轮内部轮次norm或预先限定调参改变当前关系；不能外推当前单seed结果。 |

## 失败登记补充

以下是failures.jsonl的独立避免清单，可能与上方episode指向同一事件；不能相加作为独立实验次数。

### `round2-diagnostic-bce-autocast`

- **方法／触发**：autocast_runtime_error
- **结果／归因／后续避免**：

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
  "artifact_path": "qformer-loop/reports/round2-middle2.log",
  "superseded_by": "diagnostics-round2-middle2-v2"
}
```

### `round2-premature-diagnostic-analysis`

- **方法／触发**：premature_input_read
- **结果／归因／后续避免**：

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

### `round8-yaml-duplicate-anchor`

- **方法／触发**：configuration_parser_error
- **结果／归因／后续避免**：

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
  "artifact_path": "qformer-loop/reports/round8/training-launch-failed/status.json",
  "recovery": "Regenerated appended entries from independent JSON objects; full merged parse and semantic equality checks passed"
}
```

### `round18-yaml-duplicate-anchor`

- **方法／触发**：configuration_parser_error
- **结果／归因／后续避免**：

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
  "artifact_path": "qformer-loop/reports/round18/launch-failure.json"
}
```

## 早期工程smoke补充

下列两次仅保存smoke结果，不计入167次完整训练；loss不是完整验证AUC。

### `loop4-s20262027-20260906T112022903636Z`

- **方法**：loop4，seed20262027，batch8192，训练启动前smoke。
- **结果**：{"smoke_only":true,"loss":1.440380334854126,"batch_size":8192,"peak_cuda_bytes":54106891776}
- **归因**：相同loss与显存记录只证明该smoke配置执行并保存输出；不能作为两次独立质量确认。
- **后续假设**：正式完整训练才检验循环深度的性能；相关后续结果已在阶段1保留。
- **证据**：`qformer-loop/runs/loop4-s20262027-20260906T112022903636Z/smoke.json`（SHA256前缀 `8a95f591560a`）；`qformer-loop/runs/loop4-s20262027-20260906T112022903636Z/manifest.json`（SHA256前缀 `708aa37533b5`）

### `loop4-s20262027-20260906T112222271290Z`

- **方法**：loop4，seed20262027，batch8192，训练启动前smoke。
- **结果**：{"smoke_only":true,"loss":1.440380334854126,"batch_size":8192,"peak_cuda_bytes":54106891776}
- **归因**：相同loss与显存记录只证明该smoke配置执行并保存输出；不能作为两次独立质量确认。
- **后续假设**：正式完整训练才检验循环深度的性能；相关后续结果已在阶段1保留。
- **证据**：`qformer-loop/runs/loop4-s20262027-20260906T112222271290Z/smoke.json`（SHA256前缀 `8a95f591560a`）；`qformer-loop/runs/loop4-s20262027-20260906T112222271290Z/manifest.json`（SHA256前缀 `f8785168e25f`）
