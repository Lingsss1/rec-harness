# KuaiRec：全部迭代实验明细

快照时间：2026-09-09T07:36:23.974768+00:00。返回[总览](README.md)。

[阶段1–9](KUAIREC_EXPERIMENTS.md) · [阶段10–18、跨实验归因与工程补充](KUAIREC_ROUNDS10_18.md)

每条保留原始实验ID；历史decision只作记录，不等于当前推荐。原始历史假设按当时语境保留；阶段后续方向是研究链条，后续已完成的实验不重新列为待执行任务。

历史episode 233条；本次补充发现的未归档运行 0条。完整训练结果文件 167份。诊断、统计、启动失败不增加完整训练计数。

结果来源是保存的JSON/TSV/Markdown，本次未训练、未加载模型或预测数组、未重新评估数据。证据路径为工作区逻辑相对路径，并非Git附件；大数据、权重和日志不随文档上传。行内为SHA256前12位，完整值见[证据索引](EVIDENCE_INDEX.md)。

<a id="phase-1"></a>
## 阶段 1：基线、整段与中间层循环

**方法**：Cross8 主基线、同优化设置 Cross8、Cross11；整段/中间循环2或4次；独立6/12层。

**归因与边界**：初期同时存在clip及验证Query路由差异。三seed均值的middle2接近参考；整段增加深度没有稳定收益。不能把跨设置差值全归于循环。

**历史后续假设**：先统一训练/验证Query路由，再研究入口锚定和更新阻尼；这已在K02及后续轮检查。

<a id="k001"></a>
### K001 · `baseline-s20262027-20260906T112348253214Z`

- **方法**：{"model":"QFormerCross11","query_fusion_fixed_scales":[1,1,1,0,1,1,1,1]}；设置 {"model":"QFormerCross11","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Round-1 architecture and matched-budget baseline comparison
- **状态／预算**：原记录 `keep`；{"epochs":1,"seeds":[20262027],"optimizer_steps":1240}。
- **结果**：logloss=0.190083399；AUC=0.869483096；UAUC=0.787044955；gAUC=0.799239771；NDCG(10)=0.406243284。
- **开销**：参数 11408145；训练含验证 495.1186923235655 秒；峰值CUDA 20277356544 bytes；更新 1240。
- **原记录差值**：-0.000063924；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":14.95902395248413,"mean_ms":15.015724802017212,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000063924，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 1](#phase-1)。
- **后续假设**：先统一训练/验证Query路由，再研究入口锚定和更新阻尼；这已在K02及后续轮检查。
- **原判读**：reference retained
- **证据**：`qformer-loop/runs/baseline-s20262027-20260906T112348253214Z/manifest.json`（SHA256前缀 `f36a36a85753`）；`qformer-loop/runs/baseline-s20262027-20260906T112348253214Z/result.json`（SHA256前缀 `e298a2c3df29`）

<a id="k002"></a>
### K002 · `baseline-s20262028-20260906T112713704038Z`

- **方法**：{"model":"QFormerCross11","query_fusion_fixed_scales":[1,1,1,0,1,1,1,1]}；设置 {"model":"QFormerCross11","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Round-1 architecture and matched-budget baseline comparison
- **状态／预算**：原记录 `keep`；{"epochs":1,"seeds":[20262028],"optimizer_steps":1240}。
- **结果**：logloss=0.189157275；AUC=0.870540183；UAUC=0.790406054；gAUC=0.801658471；NDCG(10)=0.410009447。
- **开销**：参数 11408145；训练含验证 495.6442411839962 秒；峰值CUDA 20277356544 bytes；更新 1240。
- **原记录差值**：+0.000810285；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":14.95902395248413,"mean_ms":15.015724802017212,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计提高0.000810285，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 1](#phase-1)。
- **后续假设**：先统一训练/验证Query路由，再研究入口锚定和更新阻尼；这已在K02及后续轮检查。
- **原判读**：reference retained
- **证据**：`qformer-loop/runs/baseline-s20262028-20260906T112713704038Z/manifest.json`（SHA256前缀 `385ef3b4292a`）；`qformer-loop/runs/baseline-s20262028-20260906T112713704038Z/result.json`（SHA256前缀 `febd26fafe0b`）

<a id="k003"></a>
### K003 · `baseline-s20262029-20260906T112713728411Z`

- **方法**：{"model":"QFormerCross11","query_fusion_fixed_scales":[1,1,1,0,1,1,1,1]}；设置 {"model":"QFormerCross11","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Round-1 architecture and matched-budget baseline comparison
- **状态／预算**：原记录 `keep`；{"epochs":1,"seeds":[20262029],"optimizer_steps":1240}。
- **结果**：logloss=0.194508271；AUC=0.867903798；UAUC=0.788130245；gAUC=0.800916595；NDCG(10)=0.405873755。
- **开销**：参数 11408145；训练含验证 499.08218017220497 秒；峰值CUDA 20277356544 bytes；更新 1240。
- **原记录差值**：-0.000911363；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":14.95902395248413,"mean_ms":15.015724802017212,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000911363，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 1](#phase-1)。
- **后续假设**：先统一训练/验证Query路由，再研究入口锚定和更新阻尼；这已在K02及后续轮检查。
- **原判读**：reference retained
- **证据**：`qformer-loop/runs/baseline-s20262029-20260906T112713728411Z/manifest.json`（SHA256前缀 `5619a8bdef61`）；`qformer-loop/runs/baseline-s20262029-20260906T112713728411Z/result.json`（SHA256前缀 `2fd166fe3056`）

<a id="k004"></a>
### K004 · `cross8_matched-s20262027-20260906T122912902434Z`

- **方法**：{"model":"QFormerCross8","query_fusion_fixed_scales":[1,1,1,0,1,1,1,1]}；设置 {"model":"QFormerCross8","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Round-1 architecture and matched-budget baseline comparison
- **状态／预算**：原记录 `keep`；{"epochs":1,"seeds":[20262027],"optimizer_steps":1240}。
- **结果**：logloss=0.189930722；AUC=0.870197742；UAUC=0.788878004；gAUC=0.801130214；NDCG(10)=0.401590634。
- **开销**：参数 9231633；训练含验证 449.74645026773214 秒；峰值CUDA 17897472000 bytes；更新 1240。
- **原记录差值**：+0.000650721；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":13.591375827789307,"mean_ms":13.604167985916138,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计提高0.000650721，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 1](#phase-1)。
- **后续假设**：先统一训练/验证Query路由，再研究入口锚定和更新阻尼；这已在K02及后续轮检查。
- **原判读**：reference retained
- **证据**：`qformer-loop/runs/cross8_matched-s20262027-20260906T122912902434Z/manifest.json`（SHA256前缀 `8750fc1be30b`）；`qformer-loop/runs/cross8_matched-s20262027-20260906T122912902434Z/result.json`（SHA256前缀 `1e62cedec999`）

<a id="k005"></a>
### K005 · `cross8_matched-s20262028-20260906T122651911367Z`

- **方法**：{"model":"QFormerCross8","query_fusion_fixed_scales":[1,1,1,0,1,1,1,1]}；设置 {"model":"QFormerCross8","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Round-1 architecture and matched-budget baseline comparison
- **状态／预算**：原记录 `keep`；{"epochs":1,"seeds":[20262028],"optimizer_steps":1240}。
- **结果**：logloss=0.198899876；AUC=0.869493782；UAUC=0.790079770；gAUC=0.801719027；NDCG(10)=0.405226701。
- **开销**：参数 9231633；训练含验证 452.677510432899 秒；峰值CUDA 17897472000 bytes；更新 1240。
- **原记录差值**：-0.000236116；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":13.591375827789307,"mean_ms":13.604167985916138,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000236116，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 1](#phase-1)。
- **后续假设**：先统一训练/验证Query路由，再研究入口锚定和更新阻尼；这已在K02及后续轮检查。
- **原判读**：reference retained
- **证据**：`qformer-loop/runs/cross8_matched-s20262028-20260906T122651911367Z/manifest.json`（SHA256前缀 `d82e10cd5be3`）；`qformer-loop/runs/cross8_matched-s20262028-20260906T122651911367Z/result.json`（SHA256前缀 `ec8f59d7fbc2`）

<a id="k006"></a>
### K006 · `cross8_matched-s20262029-20260906T122641812195Z`

- **方法**：{"model":"QFormerCross8","query_fusion_fixed_scales":[1,1,1,0,1,1,1,1]}；设置 {"model":"QFormerCross8","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Round-1 architecture and matched-budget baseline comparison
- **状态／预算**：原记录 `keep`；{"epochs":1,"seeds":[20262029],"optimizer_steps":1240}。
- **结果**：logloss=0.190784205；AUC=0.869617307；UAUC=0.789363720；gAUC=0.801313879；NDCG(10)=0.408185481。
- **开销**：参数 9231633；训练含验证 448.6968127191067 秒；峰值CUDA 17897472000 bytes；更新 1240。
- **原记录差值**：+0.000802146；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":13.591375827789307,"mean_ms":13.604167985916138,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计提高0.000802146，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 1](#phase-1)。
- **后续假设**：先统一训练/验证Query路由，再研究入口锚定和更新阻尼；这已在K02及后续轮检查。
- **原判读**：reference retained
- **证据**：`qformer-loop/runs/cross8_matched-s20262029-20260906T122641812195Z/manifest.json`（SHA256前缀 `714088ac467e`）；`qformer-loop/runs/cross8_matched-s20262029-20260906T122641812195Z/result.json`（SHA256前缀 `d0e13b9e0854`）

<a id="k007"></a>
### K007 · `cross8_original-s20262027-20260906T122501184770Z`

- **方法**：{"model":"QFormerCross8","query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross8","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":10}
- **原假设**：Round-1 architecture and matched-budget baseline comparison
- **状态／预算**：原记录 `keep`；{"epochs":1,"seeds":[20262027],"optimizer_steps":1240}。
- **结果**：logloss=0.189132309；AUC=0.869547020；UAUC=0.788346871；gAUC=0.800511887；NDCG(10)=0.404062237。
- **开销**：参数 9231633；训练含验证 448.74054527282715 秒；峰值CUDA 17897472000 bytes；更新 1240。
- **原记录差值**：+0.000000000；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":13.652032375335693,"mean_ms":13.732988023757935,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计持平0.000000000，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 1](#phase-1)。
- **后续假设**：先统一训练/验证Query路由，再研究入口锚定和更新阻尼；这已在K02及后续轮检查。
- **原判读**：reference retained
- **证据**：`qformer-loop/runs/cross8_original-s20262027-20260906T122501184770Z/manifest.json`（SHA256前缀 `84635f46d53f`）；`qformer-loop/runs/cross8_original-s20262027-20260906T122501184770Z/result.json`（SHA256前缀 `46a75d65929a`）

<a id="k008"></a>
### K008 · `cross8_original-s20262028-20260906T122521361479Z`

- **方法**：{"model":"QFormerCross8","query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross8","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":10}
- **原假设**：Round-1 architecture and matched-budget baseline comparison
- **状态／预算**：原记录 `keep`；{"epochs":1,"seeds":[20262028],"optimizer_steps":1240}。
- **结果**：logloss=0.189720537；AUC=0.869729898；UAUC=0.789052214；gAUC=0.800142646；NDCG(10)=0.405638562。
- **开销**：参数 9231633；训练含验证 450.5999351888895 秒；峰值CUDA 17897472000 bytes；更新 1240。
- **原记录差值**：+0.000000000；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":13.652032375335693,"mean_ms":13.732988023757935,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计持平0.000000000，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 1](#phase-1)。
- **后续假设**：先统一训练/验证Query路由，再研究入口锚定和更新阻尼；这已在K02及后续轮检查。
- **原判读**：reference retained
- **证据**：`qformer-loop/runs/cross8_original-s20262028-20260906T122521361479Z/manifest.json`（SHA256前缀 `cd7c19a1eb9f`）；`qformer-loop/runs/cross8_original-s20262028-20260906T122521361479Z/result.json`（SHA256前缀 `e725c3efe649`）

<a id="k009"></a>
### K009 · `cross8_original-s20262029-20260906T122912894092Z`

- **方法**：{"model":"QFormerCross8","query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross8","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":10}
- **原假设**：Round-1 architecture and matched-budget baseline comparison
- **状态／预算**：原记录 `keep`；{"epochs":1,"seeds":[20262029],"optimizer_steps":1240}。
- **结果**：logloss=0.190073952；AUC=0.868815161；UAUC=0.788430724；gAUC=0.800456185；NDCG(10)=0.408411341。
- **开销**：参数 9231633；训练含验证 446.0670702010393 秒；峰值CUDA 17897472000 bytes；更新 1240。
- **原记录差值**：+0.000000000；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":13.652032375335693,"mean_ms":13.732988023757935,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计持平0.000000000，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 1](#phase-1)。
- **后续假设**：先统一训练/验证Query路由，再研究入口锚定和更新阻尼；这已在K02及后续轮检查。
- **原判读**：reference retained
- **证据**：`qformer-loop/runs/cross8_original-s20262029-20260906T122912894092Z/manifest.json`（SHA256前缀 `4b6a1232a10f`）；`qformer-loop/runs/cross8_original-s20262029-20260906T122912894092Z/result.json`（SHA256前缀 `1f5d46bb4db8`）

<a id="k010"></a>
### K010 · `deep12-s20262027-20260906T112348364703Z`

- **方法**：{"model":"QFormerCross11","query_fusion_fixed_scales":[1,1,1,0,1,1,1,1]}；设置 {"model":"QFormerCross11","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":12,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Round-1 architecture and matched-budget baseline comparison
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262027],"optimizer_steps":1240}。
- **结果**：logloss=0.191885543；AUC=0.865379701；UAUC=0.781013201；gAUC=0.792283860；NDCG(10)=0.394827118。
- **开销**：参数 22077969；训练含验证 1113.873885653913 秒；峰值CUDA 54316738048 bytes；更新 1240。
- **原记录差值**：-0.004167319；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":38.43681526184082,"mean_ms":38.494651222229,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.004167319，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 1](#phase-1)。
- **后续假设**：先统一训练/验证Query路由，再研究入口锚定和更新阻尼；这已在K02及后续轮检查。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/deep12-s20262027-20260906T112348364703Z/manifest.json`（SHA256前缀 `0fca90250053`）；`qformer-loop/runs/deep12-s20262027-20260906T112348364703Z/result.json`（SHA256前缀 `4d1f37a0fe50`）

<a id="k011"></a>
### K011 · `deep6-s20262027-20260906T112348285703Z`

- **方法**：{"model":"QFormerCross11","query_fusion_fixed_scales":[1,1,1,0,1,1,1,1]}；设置 {"model":"QFormerCross11","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":6,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Round-1 architecture and matched-budget baseline comparison
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262027],"optimizer_steps":1240}。
- **结果**：logloss=0.190490455；AUC=0.869055687；UAUC=0.790172032；gAUC=0.801957600；NDCG(10)=0.409284035。
- **开销**：参数 14964753；训练含验证 699.8459215387702 秒；峰值CUDA 31622484480 bytes；更新 1240。
- **原记录差值**：-0.000491333；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":22.836992263793945,"mean_ms":22.91574320793152,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000491333，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 1](#phase-1)。
- **后续假设**：先统一训练/验证Query路由，再研究入口锚定和更新阻尼；这已在K02及后续轮检查。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/deep6-s20262027-20260906T112348285703Z/manifest.json`（SHA256前缀 `ecf774304782`）；`qformer-loop/runs/deep6-s20262027-20260906T112348285703Z/result.json`（SHA256前缀 `aa182e220568`）

<a id="k012"></a>
### K012 · `loop2-s20262027-20260906T112348372552Z`

- **方法**：{"model":"QFormerCross11Loop","sequence_loop_steps":2,"query_fusion_fixed_scales":[1,1,1,0,1,1,1,1]}；设置 {"model":"QFormerCross11Loop","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Round-1 architecture and matched-budget baseline comparison
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262027],"optimizer_steps":1240}。
- **结果**：logloss=0.195399142；AUC=0.868538871；UAUC=0.787819065；gAUC=0.799870352；NDCG(10)=0.403001330。
- **开销**：参数 11408145；训练含验证 694.6017894670367 秒；峰值CUDA 31574131712 bytes；更新 1240。
- **原记录差值**：-0.001008149；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":22.828495979309082,"mean_ms":22.861299180984496,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.001008149，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 1](#phase-1)。
- **后续假设**：先统一训练/验证Query路由，再研究入口锚定和更新阻尼；这已在K02及后续轮检查。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/loop2-s20262027-20260906T112348372552Z/manifest.json`（SHA256前缀 `5a2e03b9ac5f`）；`qformer-loop/runs/loop2-s20262027-20260906T112348372552Z/result.json`（SHA256前缀 `d0755876c177`）

<a id="k013"></a>
### K013 · `loop2_anchor-s20262027-20260906T112348364406Z`

- **方法**：{"model":"QFormerCross11Loop","sequence_loop_steps":2,"sequence_loop_anchor_mix":0.1,"query_fusion_fixed_scales":[1,1,1,0,1,1,1,1]}；设置 {"model":"QFormerCross11Loop","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Round-1 architecture and matched-budget baseline comparison
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262027],"optimizer_steps":1240}。
- **结果**：logloss=0.193953653；AUC=0.868645661；UAUC=0.785751043；gAUC=0.797463134；NDCG(10)=0.405422677。
- **开销**：参数 11408145；训练含验证 696.7231854647398 秒；峰值CUDA 31574131712 bytes；更新 1240。
- **原记录差值**：-0.000901359；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":22.941039085388184,"mean_ms":22.931752729415894,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000901359，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 1](#phase-1)。
- **后续假设**：先统一训练/验证Query路由，再研究入口锚定和更新阻尼；这已在K02及后续轮检查。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/loop2_anchor-s20262027-20260906T112348364406Z/manifest.json`（SHA256前缀 `3ecaa0aadc1c`）；`qformer-loop/runs/loop2_anchor-s20262027-20260906T112348364406Z/result.json`（SHA256前缀 `d1ed692f48ab`）

<a id="k014"></a>
### K014 · `loop2_anchor-s20262028-20260906T121647646187Z`

- **方法**：{"model":"QFormerCross11Loop","sequence_loop_steps":2,"sequence_loop_anchor_mix":0.1,"query_fusion_fixed_scales":[1,1,1,0,1,1,1,1]}；设置 {"model":"QFormerCross11Loop","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Round-1 architecture and matched-budget baseline comparison
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262028],"optimizer_steps":1240}。
- **结果**：logloss=0.190540323；AUC=0.867580324；UAUC=0.788533978；gAUC=0.799446628；NDCG(10)=0.408376022。
- **开销**：参数 11408145；训练含验证 701.1636211574078 秒；峰值CUDA 31574131712 bytes；更新 1240。
- **原记录差值**：-0.002149574；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":22.941039085388184,"mean_ms":22.931752729415894,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.002149574，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 1](#phase-1)。
- **后续假设**：先统一训练/验证Query路由，再研究入口锚定和更新阻尼；这已在K02及后续轮检查。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/loop2_anchor-s20262028-20260906T121647646187Z/manifest.json`（SHA256前缀 `33e4afc1db28`）；`qformer-loop/runs/loop2_anchor-s20262028-20260906T121647646187Z/result.json`（SHA256前缀 `11cc9d6210e4`）

<a id="k015"></a>
### K015 · `loop2_anchor-s20262029-20260906T121647627037Z`

- **方法**：{"model":"QFormerCross11Loop","sequence_loop_steps":2,"sequence_loop_anchor_mix":0.1,"query_fusion_fixed_scales":[1,1,1,0,1,1,1,1]}；设置 {"model":"QFormerCross11Loop","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Round-1 architecture and matched-budget baseline comparison
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262029],"optimizer_steps":1240}。
- **结果**：logloss=0.191504828；AUC=0.867967785；UAUC=0.788602667；gAUC=0.800481236；NDCG(10)=0.411116746。
- **开销**：参数 11408145；训练含验证 699.5171698108315 秒；峰值CUDA 31574131712 bytes；更新 1240。
- **原记录差值**：-0.000847376；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":22.941039085388184,"mean_ms":22.931752729415894,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000847376，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 1](#phase-1)。
- **后续假设**：先统一训练/验证Query路由，再研究入口锚定和更新阻尼；这已在K02及后续轮检查。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/loop2_anchor-s20262029-20260906T121647627037Z/manifest.json`（SHA256前缀 `522479c93114`）；`qformer-loop/runs/loop2_anchor-s20262029-20260906T121647627037Z/result.json`（SHA256前缀 `fe22f8b79133`）

<a id="k016"></a>
### K016 · `loop4-s20262027-20260906T112348386578Z`

- **方法**：{"model":"QFormerCross11Loop","sequence_loop_steps":4,"query_fusion_fixed_scales":[1,1,1,0,1,1,1,1]}；设置 {"model":"QFormerCross11Loop","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Round-1 architecture and matched-budget baseline comparison
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262027],"optimizer_steps":1240}。
- **结果**：logloss=0.193353334；AUC=0.864914148；UAUC=0.777092201；gAUC=0.789934034；NDCG(10)=0.391308396。
- **开销**：参数 11408145；训练含验证 1109.3504163324833 秒；峰值CUDA 54168787968 bytes；更新 1240。
- **原记录差值**：-0.004632872；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":38.52089500427246,"mean_ms":38.79087934494019,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.004632872，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 1](#phase-1)。
- **后续假设**：先统一训练/验证Query路由，再研究入口锚定和更新阻尼；这已在K02及后续轮检查。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/loop4-s20262027-20260906T112348386578Z/manifest.json`（SHA256前缀 `4ba454a1f4f9`）；`qformer-loop/runs/loop4-s20262027-20260906T112348386578Z/result.json`（SHA256前缀 `66da816d2f5b`）

<a id="k017"></a>
### K017 · `middle2-s20262027-20260906T121257263362Z`

- **方法**：{"model":"QFormerCross11Loop","sequence_loop_scope":"middle","sequence_loop_steps":2,"query_fusion_fixed_scales":[1,1,1,0,1,1,1,1]}；设置 {"model":"QFormerCross11Loop","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Round-1 architecture and matched-budget baseline comparison
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262027],"optimizer_steps":1240}。
- **结果**：logloss=0.193070110；AUC=0.869713080；UAUC=0.790945531；gAUC=0.802205465；NDCG(10)=0.405441370。
- **开销**：参数 11408145；训练含验证 562.8916749954224 秒；峰值CUDA 24108657664 bytes；更新 1240。
- **原记录差值**：+0.000166060；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":17.5567684173584,"mean_ms":17.631023168563843,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计提高0.000166060，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 1](#phase-1)。
- **后续假设**：先统一训练/验证Query路由，再研究入口锚定和更新阻尼；这已在K02及后续轮检查。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/middle2-s20262027-20260906T121257263362Z/manifest.json`（SHA256前缀 `22a03ab63ea5`）；`qformer-loop/runs/middle2-s20262027-20260906T121257263362Z/result.json`（SHA256前缀 `ba4a9104d156`）

<a id="k018"></a>
### K018 · `middle2-s20262028-20260906T121647640962Z`

- **方法**：{"model":"QFormerCross11Loop","sequence_loop_scope":"middle","sequence_loop_steps":2,"query_fusion_fixed_scales":[1,1,1,0,1,1,1,1]}；设置 {"model":"QFormerCross11Loop","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Round-1 architecture and matched-budget baseline comparison
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262028],"optimizer_steps":1240}。
- **结果**：logloss=0.189874409；AUC=0.868530766；UAUC=0.788707210；gAUC=0.800687641；NDCG(10)=0.406463094。
- **开销**：参数 11408145；训练含验证 564.4322580620646 秒；峰值CUDA 24108657664 bytes；更新 1240。
- **原记录差值**：-0.001199132；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":17.5567684173584,"mean_ms":17.631023168563843,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.001199132，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 1](#phase-1)。
- **后续假设**：先统一训练/验证Query路由，再研究入口锚定和更新阻尼；这已在K02及后续轮检查。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/middle2-s20262028-20260906T121647640962Z/manifest.json`（SHA256前缀 `6b123f0e4c0c`）；`qformer-loop/runs/middle2-s20262028-20260906T121647640962Z/result.json`（SHA256前缀 `f997146911b5`）

<a id="k019"></a>
### K019 · `middle2-s20262029-20260906T121647530746Z`

- **方法**：{"model":"QFormerCross11Loop","sequence_loop_scope":"middle","sequence_loop_steps":2,"query_fusion_fixed_scales":[1,1,1,0,1,1,1,1]}；设置 {"model":"QFormerCross11Loop","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Round-1 architecture and matched-budget baseline comparison
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262029],"optimizer_steps":1240}。
- **结果**：logloss=0.197915044；AUC=0.868726425；UAUC=0.788598485；gAUC=0.800814551；NDCG(10)=0.402185072。
- **开销**：参数 11408145；训练含验证 561.3717121258378 秒；峰值CUDA 24108657664 bytes；更新 1240。
- **原记录差值**：-0.000088736；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":17.5567684173584,"mean_ms":17.631023168563843,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000088736，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 1](#phase-1)。
- **后续假设**：先统一训练/验证Query路由，再研究入口锚定和更新阻尼；这已在K02及后续轮检查。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/middle2-s20262029-20260906T121647530746Z/manifest.json`（SHA256前缀 `938c7b6776f7`）；`qformer-loop/runs/middle2-s20262029-20260906T121647530746Z/result.json`（SHA256前缀 `6d98ddc1a136`）

<a id="k020"></a>
### K020 · `middle4-s20262027-20260906T121257319387Z`

- **方法**：{"model":"QFormerCross11Loop","sequence_loop_scope":"middle","sequence_loop_steps":4,"query_fusion_fixed_scales":[1,1,1,0,1,1,1,1]}；设置 {"model":"QFormerCross11Loop","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Round-1 architecture and matched-budget baseline comparison
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262027],"optimizer_steps":1240}。
- **结果**：logloss=0.192570373；AUC=0.868280287；UAUC=0.789410994；gAUC=0.801094528；NDCG(10)=0.405171848。
- **开销**：参数 11408145；训练含验证 697.4066553637385 秒；峰值CUDA 31639794688 bytes；更新 1240。
- **原记录差值**：-0.001266733；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":22.736063957214355,"mean_ms":22.867293643951417,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.001266733，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 1](#phase-1)。
- **后续假设**：先统一训练/验证Query路由，再研究入口锚定和更新阻尼；这已在K02及后续轮检查。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/middle4-s20262027-20260906T121257319387Z/manifest.json`（SHA256前缀 `e35bae5b7cbf`）；`qformer-loop/runs/middle4-s20262027-20260906T121257319387Z/result.json`（SHA256前缀 `c332d319c9da`）

<a id="k021"></a>
### K021 · `middle4-s20262028-20260906T121647620952Z`

- **方法**：{"model":"QFormerCross11Loop","sequence_loop_scope":"middle","sequence_loop_steps":4,"query_fusion_fixed_scales":[1,1,1,0,1,1,1,1]}；设置 {"model":"QFormerCross11Loop","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Round-1 architecture and matched-budget baseline comparison
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262028],"optimizer_steps":1240}。
- **结果**：logloss=0.191673248；AUC=0.868456321；UAUC=0.788316118；gAUC=0.799902210；NDCG(10)=0.406560580。
- **开销**：参数 11408145；训练含验证 696.4703677818179 秒；峰值CUDA 31639794688 bytes；更新 1240。
- **原记录差值**：-0.001273577；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":22.736063957214355,"mean_ms":22.867293643951417,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.001273577，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 1](#phase-1)。
- **后续假设**：先统一训练/验证Query路由，再研究入口锚定和更新阻尼；这已在K02及后续轮检查。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/middle4-s20262028-20260906T121647620952Z/manifest.json`（SHA256前缀 `6b2a27134331`）；`qformer-loop/runs/middle4-s20262028-20260906T121647620952Z/result.json`（SHA256前缀 `09987ed820c6`）

<a id="k022"></a>
### K022 · `middle4-s20262029-20260906T121647685217Z`

- **方法**：{"model":"QFormerCross11Loop","sequence_loop_scope":"middle","sequence_loop_steps":4,"query_fusion_fixed_scales":[1,1,1,0,1,1,1,1]}；设置 {"model":"QFormerCross11Loop","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Round-1 architecture and matched-budget baseline comparison
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262029],"optimizer_steps":1240}。
- **结果**：logloss=0.196167532；AUC=0.866927680；UAUC=0.787403662；gAUC=0.799036669；NDCG(10)=0.406778942。
- **开销**：参数 11408145；训练含验证 699.6453966870904 秒；峰值CUDA 31639794688 bytes；更新 1240。
- **原记录差值**：-0.001887481；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":22.736063957214355,"mean_ms":22.867293643951417,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.001887481，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 1](#phase-1)。
- **后续假设**：先统一训练/验证Query路由，再研究入口锚定和更新阻尼；这已在K02及后续轮检查。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/middle4-s20262029-20260906T121647685217Z/manifest.json`（SHA256前缀 `4118395edb9c`）；`qformer-loop/runs/middle4-s20262029-20260906T121647685217Z/result.json`（SHA256前缀 `2c0bd64d9226`）

<a id="phase-2"></a>
## 阶段 2：Query路由与阻尼

**方法**：固定checkpoint恢复全Query，另训练入口混合/半幅更新。

**归因与边界**：恢复训练时的全Query改善平均AUC和Logloss；被固定置零Query的零梯度不是冗余证据。

**历史后续假设**：改变循环位置和字段/历史交互来源，而非只加次数；见K03。

<a id="k023"></a>
### K023 · `routes-baseline-s20262027`

- **方法**：Q3 eval scale 0 -> 0.5 / 1, no weight update
- **原假设**：Removing train/eval query mismatch improves validation quality
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"seeds":[20262027],"evaluation_routes":3}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "primary": {
    "name": "validation_AUC_all_queries",
    "value": 0.8699734166101216,
    "delta": 0.0004903207626300032,
    "reference": "same checkpoint native route"
  },
  "guardrails": {
    "logloss": 0.1894329521538222,
    "native_logloss": 0.1900833987939914
  }
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 2](#phase-2)。
- **后续假设**：改变循环位置和字段/历史交互来源，而非只加次数；见K03。
- **证据**：`qformer-loop/reports/round2/routes/baseline-s20262027/result.json`（SHA256前缀 `3de9e3864978`）

<a id="k024"></a>
### K024 · `routes-baseline-s20262028`

- **方法**：Q3 eval scale 0 -> 0.5 / 1, no weight update
- **原假设**：Removing train/eval query mismatch improves validation quality
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"seeds":[20262028],"evaluation_routes":3}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "primary": {
    "name": "validation_AUC_all_queries",
    "value": 0.8708279342804846,
    "delta": 0.0002877513818125843,
    "reference": "same checkpoint native route"
  },
  "guardrails": {
    "logloss": 0.18874293464019073,
    "native_logloss": 0.18915727542425867
  }
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 2](#phase-2)。
- **后续假设**：改变循环位置和字段/历史交互来源，而非只加次数；见K03。
- **证据**：`qformer-loop/reports/round2/routes/baseline-s20262028/result.json`（SHA256前缀 `e1da911ca43e`）

<a id="k025"></a>
### K025 · `routes-baseline-s20262029`

- **方法**：Q3 eval scale 0 -> 0.5 / 1, no weight update
- **原假设**：Removing train/eval query mismatch improves validation quality
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"seeds":[20262029],"evaluation_routes":3}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "primary": {
    "name": "validation_AUC_all_queries",
    "value": 0.86828197053844,
    "delta": 0.0003781723090752198,
    "reference": "same checkpoint native route"
  },
  "guardrails": {
    "logloss": 0.19027100510930803,
    "native_logloss": 0.19450827060502757
  }
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 2](#phase-2)。
- **后续假设**：改变循环位置和字段/历史交互来源，而非只加次数；见K03。
- **证据**：`qformer-loop/reports/round2/routes/baseline-s20262029/result.json`（SHA256前缀 `ba9611ddc31b`）

<a id="k026"></a>
### K026 · `routes-middle2-s20262027`

- **方法**：Q3 eval scale 0 -> 0.5 / 1, no weight update
- **原假设**：Removing train/eval query mismatch improves validation quality
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"seeds":[20262027],"evaluation_routes":3}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "primary": {
    "name": "validation_AUC_all_queries",
    "value": 0.8699935258802891,
    "delta": 0.000280445868362289,
    "reference": "same checkpoint native route"
  },
  "guardrails": {
    "logloss": 0.1895778943355105,
    "native_logloss": 0.1930701100529127
  }
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 2](#phase-2)。
- **后续假设**：改变循环位置和字段/历史交互来源，而非只加次数；见K03。
- **证据**：`qformer-loop/reports/round2/routes/middle2-s20262027/result.json`（SHA256前缀 `d0ba096a0896`）

<a id="k027"></a>
### K027 · `routes-middle2-s20262028`

- **方法**：Q3 eval scale 0 -> 0.5 / 1, no weight update
- **原假设**：Removing train/eval query mismatch improves validation quality
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"seeds":[20262028],"evaluation_routes":3}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "primary": {
    "name": "validation_AUC_all_queries",
    "value": 0.8692897691125018,
    "delta": 0.0007590030082039734,
    "reference": "same checkpoint native route"
  },
  "guardrails": {
    "logloss": 0.18956021900561082,
    "native_logloss": 0.18987440936561303
  }
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 2](#phase-2)。
- **后续假设**：改变循环位置和字段/历史交互来源，而非只加次数；见K03。
- **证据**：`qformer-loop/reports/round2/routes/middle2-s20262028/result.json`（SHA256前缀 `6a433373f497`）

<a id="k028"></a>
### K028 · `routes-middle2-s20262029`

- **方法**：Q3 eval scale 0 -> 0.5 / 1, no weight update
- **原假设**：Removing train/eval query mismatch improves validation quality
- **状态／预算**：原记录 `provisional`；{"training_updates":0,"seeds":[20262029],"evaluation_routes":3}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "primary": {
    "name": "validation_AUC_all_queries",
    "value": 0.8686460110468807,
    "delta": -8.041393531799113e-05,
    "reference": "same checkpoint native route"
  },
  "guardrails": {
    "logloss": 0.18956905312026695,
    "native_logloss": 0.19791504445468114
  }
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 2](#phase-2)。
- **后续假设**：改变循环位置和字段/历史交互来源，而非只加次数；见K03。
- **证据**：`qformer-loop/reports/round2/routes/middle2-s20262029/result.json`（SHA256前缀 `11329bfd69e3`）

<a id="k029"></a>
### K029 · `diagnostics-round2-cross8`

- **方法**：same-weight query Zero/Shuffle, depth perturbation and hook export
- **原假设**：Measure query reliance, finite gradients and recurrence geometry before structural edits
- **状态／预算**：原记录 `provisional`；{}。
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
      "AUC": 0.8603539198817866,
      "logloss": 0.2033490389585495
    },
    "all_queries": {
      "AUC": 0.8603539198817866,
      "logloss": 0.2033490389585495
    },
    "q3_half": {
      "AUC": 0.8604143051838613,
      "logloss": 0.20321542024612427
    },
    "q0_zero": {
      "AUC": 0.8600874536315227,
      "logloss": 0.2044866383075714
    },
    "q0_shuffle": {
      "AUC": 0.856785057708326,
      "logloss": 0.20667436718940735
    },
    "q1_zero": {
      "AUC": 0.8566582506860818,
      "logloss": 0.20575854182243347
    },
    "q1_shuffle": {
      "AUC": 0.851592821489912,
      "logloss": 0.21086491644382477
    },
    "q2_zero": {
      "AUC": 0.8599887292597292,
      "logloss": 0.20446033775806427
    },
    "q2_shuffle": {
      "AUC": 0.8582394923069083,
      "logloss": 0.20641355216503143
    },
    "q3_zero": {
      "AUC": 0.8596458531083405,
      "logloss": 0.20414601266384125
    },
    "q3_shuffle": {
      "AUC": 0.857364110301757,
      "logloss": 0.20666155219078064
    },
    "q4_zero": {
      "AUC": 0.8597952956423838,
      "logloss": 0.20357711613178253
    },
    "q4_shuffle": {
      "AUC": 0.8564590024224749,
      "logloss": 0.206950381398201
    },
    "q5_zero": {
      "AUC": 0.8596897259133316,
      "logloss": 0.20718878507614136
    },
    "q5_shuffle": {
      "AUC": 0.8536180672181725,
      "logloss": 0.20852252840995789
    },
    "q6_zero": {
      "AUC": 0.8588383881991499,
      "logloss": 0.20687530934810638
    },
    "q6_shuffle": {
      "AUC": 0.8526816303712209,
      "logloss": 0.21119429171085358
    },
    "q7_zero": {
      "AUC": 0.8575813326444355,
      "logloss": 0.20583581924438477
    },
    "q7_shuffle": {
      "AUC": 0.8551588998988348,
      "logloss": 0.20819737017154694
    }
  },
  "checkpoint_unchanged": true,
  "note": "subset diagnostics; within-block shuffle; blockwise loss means in toolkit differ from pooled full-validation metrics"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 2](#phase-2)。
- **后续假设**：改变循环位置和字段/历史交互来源，而非只加次数；见K03。
- **证据**：`qformer-loop/reports/round2/cross8/summary.json`（SHA256前缀 `7b2fdade2f39`）；`qformer-loop/reports/round2/cross8/toolkit-hooks/hook_report.json`（SHA256前缀 `4e007c73958c`）；`qformer-loop/reports/round2/cross8/toolkit-perturbations/perturbation_report.json`（SHA256前缀 `441147810c96`）

<a id="k030"></a>
### K030 · `diagnostics-round2-cross11`

- **方法**：same-weight query Zero/Shuffle, depth perturbation and hook export
- **原假设**：Measure query reliance, finite gradients and recurrence geometry before structural edits
- **状态／预算**：原记录 `provisional`；{}。
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
      "AUC": 0.8596503350114885,
      "logloss": 0.20410896837711334
    },
    "all_queries": {
      "AUC": 0.8597460221652203,
      "logloss": 0.20424902439117432
    },
    "q3_half": {
      "AUC": 0.8597136899440185,
      "logloss": 0.2033531367778778
    },
    "q0_zero": {
      "AUC": 0.8587282753156813,
      "logloss": 0.20884588360786438
    },
    "q0_shuffle": {
      "AUC": 0.8403825911358043,
      "logloss": 0.21471631526947021
    },
    "q1_zero": {
      "AUC": 0.8587013944573567,
      "logloss": 0.21388423442840576
    },
    "q1_shuffle": {
      "AUC": 0.853613215695302,
      "logloss": 0.20842865109443665
    },
    "q2_zero": {
      "AUC": 0.8598443367871502,
      "logloss": 0.22771163284778595
    },
    "q2_shuffle": {
      "AUC": 0.8589082146449907,
      "logloss": 0.20476317405700684
    },
    "q3_zero": {
      "AUC": 0.8596503350114885,
      "logloss": 0.20410896837711334
    },
    "q3_shuffle": {
      "AUC": 0.8596503350114885,
      "logloss": 0.20410896837711334
    },
    "q4_zero": {
      "AUC": 0.857911310123569,
      "logloss": 0.22315764427185059
    },
    "q4_shuffle": {
      "AUC": 0.8563618938168966,
      "logloss": 0.20653551816940308
    },
    "q5_zero": {
      "AUC": 0.8596300376084477,
      "logloss": 0.21226246654987335
    },
    "q5_shuffle": {
      "AUC": 0.8566197510957979,
      "logloss": 0.2065700888633728
    },
    "q6_zero": {
      "AUC": 0.8590987229862476,
      "logloss": 0.20950283110141754
    },
    "q6_shuffle": {
      "AUC": 0.8440738354554038,
      "logloss": 0.2127789705991745
    },
    "q7_zero": {
      "AUC": 0.8597212428590314,
      "logloss": 0.20633462071418762
    },
    "q7_shuffle": {
      "AUC": 0.8561343622521326,
      "logloss": 0.2062728852033615
    }
  },
  "checkpoint_unchanged": true,
  "note": "subset diagnostics; within-block shuffle; blockwise loss means in toolkit differ from pooled full-validation metrics"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 2](#phase-2)。
- **后续假设**：改变循环位置和字段/历史交互来源，而非只加次数；见K03。
- **证据**：`qformer-loop/reports/round2/cross11/summary.json`（SHA256前缀 `6c3ad34a98f2`）；`qformer-loop/reports/round2/cross11/toolkit-hooks/hook_report.json`（SHA256前缀 `b690029e32e0`）；`qformer-loop/reports/round2/cross11/toolkit-perturbations/perturbation_report.json`（SHA256前缀 `0d23ef479921`）

<a id="k031"></a>
### K031 · `diagnostics-round2-middle2-v2`

- **方法**：same-weight query Zero/Shuffle, depth perturbation and hook export
- **原假设**：Measure query reliance, finite gradients and recurrence geometry before structural edits
- **状态／预算**：原记录 `provisional`；{}。
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
      "AUC": 0.8599578586205152,
      "logloss": 0.20728984475135803
    },
    "all_queries": {
      "AUC": 0.8606606070941845,
      "logloss": 0.2041746824979782
    },
    "q3_half": {
      "AUC": 0.8605086067916623,
      "logloss": 0.2045028954744339
    },
    "q0_zero": {
      "AUC": 0.8598831785396914,
      "logloss": 0.22060993313789368
    },
    "q0_shuffle": {
      "AUC": 0.858559416129593,
      "logloss": 0.20906700193881989
    },
    "q1_zero": {
      "AUC": 0.8593591507063039,
      "logloss": 0.216674342751503
    },
    "q1_shuffle": {
      "AUC": 0.8466911443913236,
      "logloss": 0.2158718854188919
    },
    "q2_zero": {
      "AUC": 0.8596479398756874,
      "logloss": 0.21228788793087006
    },
    "q2_shuffle": {
      "AUC": 0.8512296479354825,
      "logloss": 0.2137618362903595
    },
    "q3_zero": {
      "AUC": 0.8599578586205152,
      "logloss": 0.20728984475135803
    },
    "q3_shuffle": {
      "AUC": 0.8599578586205152,
      "logloss": 0.20728984475135803
    },
    "q4_zero": {
      "AUC": 0.8579802706032031,
      "logloss": 0.2161305546760559
    },
    "q4_shuffle": {
      "AUC": 0.8484761795422859,
      "logloss": 0.21418295800685883
    },
    "q5_zero": {
      "AUC": 0.8601521286344898,
      "logloss": 0.221412792801857
    },
    "q5_shuffle": {
      "AUC": 0.8510356398234827,
      "logloss": 0.21447968482971191
    },
    "q6_zero": {
      "AUC": 0.860814254844612,
      "logloss": 0.21960705518722534
    },
    "q6_shuffle": {
      "AUC": 0.8596295729436538,
      "logloss": 0.20842038094997406
    },
    "q7_zero": {
      "AUC": 0.8595939458266411,
      "logloss": 0.21294201910495758
    },
    "q7_shuffle": {
      "AUC": 0.8535959639587737,
      "logloss": 0.21234937012195587
    },
    "inference_loops1": {
      "AUC": 0.8553992709966982,
      "logloss": 0.20967337489128113
    },
    "inference_loops3": {
      "AUC": 0.8589178310941108,
      "logloss": 0.2128170132637024
    },
    "inference_loops4": {
      "AUC": 0.8447280412429037,
      "logloss": 0.22673816978931427
    }
  },
  "checkpoint_unchanged": true,
  "note": "subset diagnostics; within-block shuffle; blockwise loss means in toolkit differ from pooled full-validation metrics"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 2](#phase-2)。
- **后续假设**：改变循环位置和字段/历史交互来源，而非只加次数；见K03。
- **证据**：`qformer-loop/reports/round2/middle2-v2/summary.json`（SHA256前缀 `31738add1b01`）；`qformer-loop/reports/round2/middle2-v2/toolkit-hooks/hook_report.json`（SHA256前缀 `740ea13ccb8e`）；`qformer-loop/reports/round2/middle2-v2/toolkit-perturbations/perturbation_report.json`（SHA256前缀 `e190fff12119`）

<a id="k032"></a>
### K032 · `round2-diagnostic-bce-autocast`

- **方法**：initial export attempt
- **原假设**：Export native-route activation gradients
- **状态／预算**：原记录 `crash`；{}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "decision": "crash",
  "validity": {
    "valid": false,
    "violations": [
      "BCE executed inside autocast"
    ],
    "notes": [
      "fixed by FP32 BCE outside autocast; successful rerun is diagnostics-round2-middle2-v2"
    ]
  },
  "falsification_result": null,
  "note": "本条未保存独立性能指标；保留运行状态与证据位置，不补造数值。"
}
```

</details>

- **归因**：这是执行或有效性问题；本条不能用作结构性能反证。修复后完整训练需用独立运行ID核对。 本阶段证据解释见[阶段 2](#phase-2)。
- **后续假设**：改变循环位置和字段/历史交互来源，而非只加次数；见K03。
- **失败后续门槛**：先复核同配置/初始函数/有效梯度及完整预算，再判断科学假设；不把同一次修复重复算成独立支持。
- **证据**：`qformer-loop/reports/round2-middle2.log`

<a id="k033"></a>
### K033 · `middle2_anchor50-s20262027-20260906T132615122887Z`

- **方法**：{"model":"QFormerCross11Loop","sequence_loop_scope":"middle","sequence_loop_steps":2,"sequence_loop_anchor_mix":0.5,"query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Loop","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Reinjecting half the pre-loop state preserves useful query information
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262027],"optimizer_steps":1240}。
- **结果**：logloss=0.190598255；AUC=0.869011225；UAUC=0.790243994；gAUC=0.801888662；NDCG(10)=0.411813784。
- **开销**：参数 11408145；训练含验证 558.1768635511398 秒；峰值CUDA 24175766528 bytes；更新 1240。
- **原记录差值**：-0.000535795；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":17.566895484924316,"mean_ms":17.619572782516478,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000535795，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 2](#phase-2)。
- **后续假设**：改变循环位置和字段/历史交互来源，而非只加次数；见K03。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/middle2_anchor50-s20262027-20260906T132615122887Z/manifest.json`（SHA256前缀 `8a656ebd78b8`）；`qformer-loop/runs/middle2_anchor50-s20262027-20260906T132615122887Z/result.json`（SHA256前缀 `3aa087bb6758`）

<a id="k034"></a>
### K034 · `middle2_anchor50-s20262028-20260906T132615031643Z`

- **方法**：{"model":"QFormerCross11Loop","sequence_loop_scope":"middle","sequence_loop_steps":2,"sequence_loop_anchor_mix":0.5,"query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Loop","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Reinjecting half the pre-loop state preserves useful query information
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262028],"optimizer_steps":1240}。
- **结果**：logloss=0.188839200；AUC=0.870245906；UAUC=0.788830565；gAUC=0.801282666；NDCG(10)=0.409127885。
- **开销**：参数 11408145；训练含验证 568.4847017601132 秒；峰值CUDA 24175766528 bytes；更新 1240。
- **原记录差值**：+0.000516008；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":17.566895484924316,"mean_ms":17.619572782516478,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计提高0.000516008，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 2](#phase-2)。
- **后续假设**：改变循环位置和字段/历史交互来源，而非只加次数；见K03。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/middle2_anchor50-s20262028-20260906T132615031643Z/manifest.json`（SHA256前缀 `ef093cec595e`）；`qformer-loop/runs/middle2_anchor50-s20262028-20260906T132615031643Z/result.json`（SHA256前缀 `912dfec61eef`）

<a id="k035"></a>
### K035 · `middle2_anchor50-s20262029-20260906T132614984678Z`

- **方法**：{"model":"QFormerCross11Loop","sequence_loop_scope":"middle","sequence_loop_steps":2,"sequence_loop_anchor_mix":0.5,"query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Loop","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Reinjecting half the pre-loop state preserves useful query information
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262029],"optimizer_steps":1240}。
- **结果**：logloss=0.189267811；AUC=0.869885554；UAUC=0.790514261；gAUC=0.802327688；NDCG(10)=0.409080068。
- **开销**：参数 11408145；训练含验证 562.2734424993396 秒；峰值CUDA 24175766528 bytes；更新 1240。
- **原记录差值**：+0.001070393；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":17.566895484924316,"mean_ms":17.619572782516478,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计提高0.001070393，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 2](#phase-2)。
- **后续假设**：改变循环位置和字段/历史交互来源，而非只加次数；见K03。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/middle2_anchor50-s20262029-20260906T132614984678Z/manifest.json`（SHA256前缀 `fb820eed5d3a`）；`qformer-loop/runs/middle2_anchor50-s20262029-20260906T132614984678Z/result.json`（SHA256前缀 `64e9578e7c68`）

<a id="k036"></a>
### K036 · `middle2_damped50-s20262027-20260906T132615128131Z`

- **方法**：{"model":"QFormerCross11Loop","sequence_loop_scope":"middle","sequence_loop_steps":2,"sequence_loop_update_scale":0.5,"query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Loop","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Half-scale extra recurrent update reduces state drift while preserving validation quality
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262027],"optimizer_steps":1240}。
- **结果**：logloss=0.189605043；AUC=0.869303744；UAUC=0.791462014；gAUC=0.803145434；NDCG(10)=0.412055299。
- **开销**：参数 11408145；训练含验证 562.682440161705 秒；峰值CUDA 24175766528 bytes；更新 1240。
- **原记录差值**：-0.000243276；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":17.575119972229004,"mean_ms":17.636474466323854,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000243276，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 2](#phase-2)。
- **后续假设**：改变循环位置和字段/历史交互来源，而非只加次数；见K03。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/middle2_damped50-s20262027-20260906T132615128131Z/manifest.json`（SHA256前缀 `7f98367a47e6`）；`qformer-loop/runs/middle2_damped50-s20262027-20260906T132615128131Z/result.json`（SHA256前缀 `98e45bd74055`）

<a id="k037"></a>
### K037 · `middle2_damped50-s20262028-20260906T132615043003Z`

- **方法**：{"model":"QFormerCross11Loop","sequence_loop_scope":"middle","sequence_loop_steps":2,"sequence_loop_update_scale":0.5,"query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Loop","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Half-scale extra recurrent update reduces state drift while preserving validation quality
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262028],"optimizer_steps":1240}。
- **结果**：logloss=0.188769934；AUC=0.870381761；UAUC=0.790617457；gAUC=0.802604363；NDCG(10)=0.407053068。
- **开销**：参数 11408145；训练含验证 562.7107034176588 秒；峰值CUDA 24175766528 bytes；更新 1240。
- **原记录差值**：+0.000651862；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":17.575119972229004,"mean_ms":17.636474466323854,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计提高0.000651862，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 2](#phase-2)。
- **后续假设**：改变循环位置和字段/历史交互来源，而非只加次数；见K03。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/middle2_damped50-s20262028-20260906T132615043003Z/manifest.json`（SHA256前缀 `fb3e7a98493b`）；`qformer-loop/runs/middle2_damped50-s20262028-20260906T132615043003Z/result.json`（SHA256前缀 `21a4be11d550`）

<a id="k038"></a>
### K038 · `middle2_damped50-s20262029-20260906T132615116024Z`

- **方法**：{"model":"QFormerCross11Loop","sequence_loop_scope":"middle","sequence_loop_steps":2,"sequence_loop_update_scale":0.5,"query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Loop","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Half-scale extra recurrent update reduces state drift while preserving validation quality
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262029],"optimizer_steps":1240}。
- **结果**：logloss=0.189140160；AUC=0.869501714；UAUC=0.791125752；gAUC=0.802897316；NDCG(10)=0.410099339。
- **开销**：参数 11408145；训练含验证 562.4441971257329 秒；峰值CUDA 24175766528 bytes；更新 1240。
- **原记录差值**：+0.000686553；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":17.575119972229004,"mean_ms":17.636474466323854,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计提高0.000686553，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 2](#phase-2)。
- **后续假设**：改变循环位置和字段/历史交互来源，而非只加次数；见K03。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/middle2_damped50-s20262029-20260906T132615116024Z/manifest.json`（SHA256前缀 `153050cab6e7`）；`qformer-loop/runs/middle2_damped50-s20262029-20260906T132615116024Z/result.json`（SHA256前缀 `a9c17d0ef3df`）

<a id="k039"></a>
### K039 · `diagnostics-round2-middle2-damped50`

- **方法**：same-weight query Zero/Shuffle, depth perturbation and hook export
- **原假设**：Measure query reliance, finite gradients and recurrence geometry before structural edits
- **状态／预算**：原记录 `provisional`；{}。
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
      "AUC": 0.8607282517276069,
      "logloss": 0.20368030667304993
    },
    "all_queries": {
      "AUC": 0.8607282517276069,
      "logloss": 0.20368030667304993
    },
    "q3_half": {
      "AUC": 0.8606146264007193,
      "logloss": 0.20478254556655884
    },
    "q0_zero": {
      "AUC": 0.8602563381629634,
      "logloss": 0.20824359357357025
    },
    "q0_shuffle": {
      "AUC": 0.8597690758753338,
      "logloss": 0.20479924976825714
    },
    "q1_zero": {
      "AUC": 0.861073863064941,
      "logloss": 0.20494244992733002
    },
    "q1_shuffle": {
      "AUC": 0.852522919886653,
      "logloss": 0.21206465363502502
    },
    "q2_zero": {
      "AUC": 0.8599774463536892,
      "logloss": 0.2049076408147812
    },
    "q2_shuffle": {
      "AUC": 0.8555028722367144,
      "logloss": 0.20978187024593353
    },
    "q3_zero": {
      "AUC": 0.8603507158068217,
      "logloss": 0.20664677023887634
    },
    "q3_shuffle": {
      "AUC": 0.8583944326703079,
      "logloss": 0.20645053684711456
    },
    "q4_zero": {
      "AUC": 0.8598963454502592,
      "logloss": 0.2066020369529724
    },
    "q4_shuffle": {
      "AUC": 0.855532980403244,
      "logloss": 0.21048106253147125
    },
    "q5_zero": {
      "AUC": 0.8594684652111722,
      "logloss": 0.20553086698055267
    },
    "q5_shuffle": {
      "AUC": 0.8554355908866764,
      "logloss": 0.20934072136878967
    },
    "q6_zero": {
      "AUC": 0.8603295524375741,
      "logloss": 0.20454730093479156
    },
    "q6_shuffle": {
      "AUC": 0.8570025694104441,
      "logloss": 0.2076481282711029
    },
    "q7_zero": {
      "AUC": 0.861110121703651,
      "logloss": 0.20586808025836945
    },
    "q7_shuffle": {
      "AUC": 0.8584472249272284,
      "logloss": 0.20785574615001678
    },
    "inference_loops1": {
      "AUC": 0.8585281378526282,
      "logloss": 0.204511359333992
    },
    "inference_loops3": {
      "AUC": 0.8607611246496596,
      "logloss": 0.20411373674869537
    },
    "inference_loops4": {
      "AUC": 0.8591132817790843,
      "logloss": 0.2055419236421585
    }
  },
  "checkpoint_unchanged": true,
  "note": "subset diagnostics; within-block shuffle; blockwise loss means in toolkit differ from pooled full-validation metrics"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 2](#phase-2)。
- **后续假设**：改变循环位置和字段/历史交互来源，而非只加次数；见K03。
- **证据**：`qformer-loop/reports/round2/middle2-damped50/summary.json`（SHA256前缀 `382a5f71923f`）；`qformer-loop/reports/round2/middle2-damped50/toolkit-hooks/hook_report.json`（SHA256前缀 `d649dc69d24c`）；`qformer-loop/reports/round2/middle2-damped50/toolkit-perturbations/perturbation_report.json`（SHA256前缀 `845f4db11816`）

<a id="k040"></a>
### K040 · `round2-premature-diagnostic-analysis`

- **方法**：toolkit invoked before exporter completed
- **原假设**：Analyze newly exported checkpoint arrays
- **状态／预算**：原记录 `invalid`；{}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "decision": "invalid",
  "validity": {
    "valid": false,
    "violations": [
      "exporter still writing arrays; required hooks.json/states.npz not present"
    ],
    "notes": [
      "does not invalidate the model or completed rerun"
    ]
  },
  "falsification_result": null,
  "note": "本条未保存独立性能指标；保留运行状态与证据位置，不补造数值。"
}
```

</details>

- **归因**：这是执行或有效性问题；本条不能用作结构性能反证。修复后完整训练需用独立运行ID核对。 本阶段证据解释见[阶段 2](#phase-2)。
- **后续假设**：改变循环位置和字段/历史交互来源，而非只加次数；见K03。
- **失败后续门槛**：先复核同配置/初始函数/有效梯度及完整预算，再判断科学假设；不把同一次修复重复算成独立支持。
- **证据**：`qformer-loop/reports/round2/toolkit-post-training-premature-status.json`（SHA256前缀 `8ba547d7d1fa`）

<a id="phase-3"></a>
## 阶段 3：循环位置

**方法**：字段末层、历史首/尾层、交替及历史后字段回读。

**归因与边界**：位置和读取来源比一味增加深度更有研究价值；字段回读保留成本取舍，字段循环有明显seed波动。

**历史后续假设**：分开用户上下文、目标物品及全部字段回读；见K04。

<a id="k041"></a>
### K041 · `place_alternating2-s20262027-20260906T135252356123Z`

- **方法**：{"model":"QFormerCross11Placement","loop_placement":"alternating","placement_steps":2,"query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Placement","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Alternate recurrent field and history reads with persistent latent queries
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262027],"optimizer_steps":1240}。
- **结果**：logloss=0.189498062；AUC=0.869032758；UAUC=0.787721888；gAUC=0.799982865；NDCG(10)=0.406526883。
- **开销**：参数 11408145；训练含验证 604.9235116317868 秒；峰值CUDA 26204303360 bytes；更新 1240。
- **原记录差值**：-0.000514262；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":18.95358371734619,"mean_ms":18.967292833328248,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000514262，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 3](#phase-3)。
- **后续假设**：分开用户上下文、目标物品及全部字段回读；见K04。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/place_alternating2-s20262027-20260906T135252356123Z/manifest.json`（SHA256前缀 `1ccf56c5cac7`）；`qformer-loop/runs/place_alternating2-s20262027-20260906T135252356123Z/result.json`（SHA256前缀 `48f522b78bf8`）

<a id="k042"></a>
### K042 · `place_feedback2-s20262027-20260906T135252383734Z`

- **方法**：{"model":"QFormerCross11Placement","loop_placement":"field_feedback","placement_steps":2,"query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Placement","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Let history-conditioned queries return to fields before the final history reads
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262027],"optimizer_steps":1240}。
- **结果**：logloss=0.189726938；AUC=0.869487709；UAUC=0.789530606；gAUC=0.802280128；NDCG(10)=0.407811574。
- **开销**：参数 11408145；训练含验证 542.1615200191736 秒；峰值CUDA 22438734848 bytes；更新 1240。
- **原记录差值**：-0.000059311；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":16.34940814971924,"mean_ms":16.35241847038269,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000059311，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 3](#phase-3)。
- **后续假设**：分开用户上下文、目标物品及全部字段回读；见K04。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/place_feedback2-s20262027-20260906T135252383734Z/manifest.json`（SHA256前缀 `98d7b4fdb839`）；`qformer-loop/runs/place_feedback2-s20262027-20260906T135252383734Z/result.json`（SHA256前缀 `9128b6dacc4b`）

<a id="k043"></a>
### K043 · `place_feedback2-s20262028-20260906T140336021412Z`

- **方法**：{"model":"QFormerCross11Placement","loop_placement":"field_feedback","placement_steps":2,"query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Placement","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Let history-conditioned queries return to fields before the final history reads
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262028],"optimizer_steps":1240}。
- **结果**：logloss=0.189292885；AUC=0.869569032；UAUC=0.788755741；gAUC=0.801075108；NDCG(10)=0.412255570。
- **开销**：参数 11408145；训练含验证 538.1555475667119 秒；峰值CUDA 22438734848 bytes；更新 1240。
- **原记录差值**：-0.000160867；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":16.34940814971924,"mean_ms":16.35241847038269,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000160867，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 3](#phase-3)。
- **后续假设**：分开用户上下文、目标物品及全部字段回读；见K04。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/place_feedback2-s20262028-20260906T140336021412Z/manifest.json`（SHA256前缀 `10715dac5efa`）；`qformer-loop/runs/place_feedback2-s20262028-20260906T140336021412Z/result.json`（SHA256前缀 `690f2c05c5f8`）

<a id="k044"></a>
### K044 · `place_feedback2-s20262029-20260906T140336014811Z`

- **方法**：{"model":"QFormerCross11Placement","loop_placement":"field_feedback","placement_steps":2,"query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Placement","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Let history-conditioned queries return to fields before the final history reads
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262029],"optimizer_steps":1240}。
- **结果**：logloss=0.189558398；AUC=0.868719841；UAUC=0.791121895；gAUC=0.802726657；NDCG(10)=0.411445935。
- **开销**：参数 11408145；训练含验证 533.523654088378 秒；峰值CUDA 22438734848 bytes；更新 1240。
- **原记录差值**：-0.000095320；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":16.34940814971924,"mean_ms":16.35241847038269,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000095320，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 3](#phase-3)。
- **后续假设**：分开用户上下文、目标物品及全部字段回读；见K04。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/place_feedback2-s20262029-20260906T140336014811Z/manifest.json`（SHA256前缀 `022b16057587`）；`qformer-loop/runs/place_feedback2-s20262029-20260906T140336014811Z/result.json`（SHA256前缀 `02c9faea3f03`）

<a id="k045"></a>
### K045 · `place_field2-s20262027-20260906T135252567700Z`

- **方法**：{"model":"QFormerCross11Placement","loop_placement":"field","placement_steps":2,"query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Placement","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Move recurrence from history processing to field/target conditioning
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262027],"optimizer_steps":1240}。
- **结果**：logloss=0.189385440；AUC=0.869121904；UAUC=0.789897643；gAUC=0.801777174；NDCG(10)=0.407990891。
- **开销**：参数 11408145；训练含验证 535.7194380611181 秒；峰值CUDA 22371101696 bytes；更新 1240。
- **原记录差值**：-0.000425116；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":16.336112022399902,"mean_ms":16.387300729751587,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000425116，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 3](#phase-3)。
- **后续假设**：分开用户上下文、目标物品及全部字段回读；见K04。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/place_field2-s20262027-20260906T135252567700Z/manifest.json`（SHA256前缀 `001b819699f3`）；`qformer-loop/runs/place_field2-s20262027-20260906T135252567700Z/result.json`（SHA256前缀 `3cb48b19703f`）

<a id="k046"></a>
### K046 · `place_field2-s20262028-20260906T140336061987Z`

- **方法**：{"model":"QFormerCross11Placement","loop_placement":"field","placement_steps":2,"query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Placement","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Move recurrence from history processing to field/target conditioning
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262028],"optimizer_steps":1240}。
- **结果**：logloss=0.189158310；AUC=0.870338002；UAUC=0.788384804；gAUC=0.800042313；NDCG(10)=0.410425041。
- **开销**：参数 11408145；训练含验证 534.4157861769199 秒；峰值CUDA 22371101696 bytes；更新 1240。
- **原记录差值**：+0.000608103；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":16.336112022399902,"mean_ms":16.387300729751587,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计提高0.000608103，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 3](#phase-3)。
- **后续假设**：分开用户上下文、目标物品及全部字段回读；见K04。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/place_field2-s20262028-20260906T140336061987Z/manifest.json`（SHA256前缀 `9f2ebbbf7972`）；`qformer-loop/runs/place_field2-s20262028-20260906T140336061987Z/result.json`（SHA256前缀 `5a60c751a68d`）

<a id="k047"></a>
### K047 · `place_field2-s20262029-20260906T140336018898Z`

- **方法**：{"model":"QFormerCross11Placement","loop_placement":"field","placement_steps":2,"query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Placement","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Move recurrence from history processing to field/target conditioning
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262029],"optimizer_steps":1240}。
- **结果**：logloss=0.191276264；AUC=0.866293456；UAUC=0.789354788；gAUC=0.801867128；NDCG(10)=0.408920697。
- **开销**：参数 11408145；训练含验证 535.9344513341784 秒；峰值CUDA 22371101696 bytes；更新 1240。
- **原记录差值**：-0.002521705；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":16.336112022399902,"mean_ms":16.387300729751587,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.002521705，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 3](#phase-3)。
- **后续假设**：分开用户上下文、目标物品及全部字段回读；见K04。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/place_field2-s20262029-20260906T140336018898Z/manifest.json`（SHA256前缀 `b2093837fd2e`）；`qformer-loop/runs/place_field2-s20262029-20260906T140336018898Z/result.json`（SHA256前缀 `2447ca6bc754`）

<a id="k048"></a>
### K048 · `place_front2-s20262027-20260906T135252489600Z`

- **方法**：{"model":"QFormerCross11Placement","loop_placement":"sequence_front","placement_steps":2,"query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Placement","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Repeat the first history-read block instead of the middle history block
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262027],"optimizer_steps":1240}。
- **结果**：logloss=0.189919426；AUC=0.868282150；UAUC=0.790133682；gAUC=0.802469295；NDCG(10)=0.413736518。
- **开销**：参数 11408145；训练含验证 560.1093784198165 秒；峰值CUDA 23974112256 bytes；更新 1240。
- **原记录差值**：-0.001264870；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":17.36412811279297,"mean_ms":17.428822374343873,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.001264870，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 3](#phase-3)。
- **后续假设**：分开用户上下文、目标物品及全部字段回读；见K04。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/place_front2-s20262027-20260906T135252489600Z/manifest.json`（SHA256前缀 `dce64d2f6499`）；`qformer-loop/runs/place_front2-s20262027-20260906T135252489600Z/result.json`（SHA256前缀 `a8c9a47b2b87`）

<a id="k049"></a>
### K049 · `place_tail2-s20262027-20260906T135252567701Z`

- **方法**：{"model":"QFormerCross11Placement","loop_placement":"sequence_tail","placement_steps":2,"query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Placement","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Repeat the final history-read block instead of the middle history block
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262027],"optimizer_steps":1240}。
- **结果**：logloss=0.189850328；AUC=0.868925880；UAUC=0.789247769；gAUC=0.801465565；NDCG(10)=0.409006315。
- **开销**：参数 11408145；训练含验证 561.9283286854625 秒；峰值CUDA 23975558144 bytes；更新 1240。
- **原记录差值**：-0.000621140；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":17.373119354248047,"mean_ms":17.39921669960022,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000621140，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 3](#phase-3)。
- **后续假设**：分开用户上下文、目标物品及全部字段回读；见K04。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/place_tail2-s20262027-20260906T135252567701Z/manifest.json`（SHA256前缀 `97474bd3ca48`）；`qformer-loop/runs/place_tail2-s20262027-20260906T135252567701Z/result.json`（SHA256前缀 `b173fc8a4f9f`）

<a id="k050"></a>
### K050 · `diagnostics-round3-place_feedback2`

- **方法**：export final norm and individual field/sequence block hooks; paired equal-width Query Zero/Shuffle
- **原假设**：Check the new field/history call graph with shared existing probes
- **状态／预算**：原记录 `provisional`；{}。
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
      "AUC": 0.8599354596653387,
      "logloss": 0.2040272355079651
    },
    "all_queries": {
      "AUC": 0.8599354596653387,
      "logloss": 0.2040272355079651
    },
    "q3_half": {
      "AUC": 0.859886012994934,
      "logloss": 0.20399592816829681
    },
    "q0_zero": {
      "AUC": 0.859302168017788,
      "logloss": 0.20598402619361877
    },
    "q0_shuffle": {
      "AUC": 0.8542837058717139,
      "logloss": 0.2107304334640503
    },
    "q1_zero": {
      "AUC": 0.859673213416248,
      "logloss": 0.2062813937664032
    },
    "q1_shuffle": {
      "AUC": 0.8536465913001774,
      "logloss": 0.2117195576429367
    },
    "q2_zero": {
      "AUC": 0.8590072051429843,
      "logloss": 0.2054300606250763
    },
    "q2_shuffle": {
      "AUC": 0.8589041826218475,
      "logloss": 0.2053479254245758
    },
    "q3_zero": {
      "AUC": 0.8597539742695334,
      "logloss": 0.20947718620300293
    },
    "q3_shuffle": {
      "AUC": 0.8571520013839238,
      "logloss": 0.20857591927051544
    },
    "q4_zero": {
      "AUC": 0.8594259927369006,
      "logloss": 0.20641815662384033
    },
    "q4_shuffle": {
      "AUC": 0.8545387561528378,
      "logloss": 0.21033695340156555
    },
    "q5_zero": {
      "AUC": 0.8599102621608352,
      "logloss": 0.20581990480422974
    },
    "q5_shuffle": {
      "AUC": 0.8577912153954834,
      "logloss": 0.2077895849943161
    },
    "q6_zero": {
      "AUC": 0.8592402303128792,
      "logloss": 0.20531515777111053
    },
    "q6_shuffle": {
      "AUC": 0.8557914332979265,
      "logloss": 0.2088811993598938
    },
    "q7_zero": {
      "AUC": 0.859629135736325,
      "logloss": 0.205266073346138
    },
    "q7_shuffle": {
      "AUC": 0.8588720003606476,
      "logloss": 0.20547936856746674
    },
    "inference_placement_steps1": {
      "AUC": 0.8567510379090773,
      "logloss": 0.21228425204753876
    },
    "inference_placement_steps3": {
      "AUC": 0.8570853304344558,
      "logloss": 0.20551763474941254
    }
  },
  "checkpoint_unchanged": true,
  "note": "subset diagnostics; within-block shuffle; blockwise loss means in toolkit differ from pooled full-validation metrics"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 3](#phase-3)。
- **后续假设**：分开用户上下文、目标物品及全部字段回读；见K04。
- **证据**：`qformer-loop/reports/round3/place_feedback2/summary.json`（SHA256前缀 `ed56b6f5c622`）；`qformer-loop/reports/round3/place_feedback2/toolkit-hooks/hook_report.json`（SHA256前缀 `8c8cd7d44593`）；`qformer-loop/reports/round3/place_feedback2/toolkit-perturbations/perturbation_report.json`（SHA256前缀 `e6b22b09ab92`）

<a id="k051"></a>
### K051 · `diagnostics-round3-place_field2`

- **方法**：export final norm and individual field/sequence block hooks; paired equal-width Query Zero/Shuffle
- **原假设**：Check the new field/history call graph with shared existing probes
- **状态／预算**：原记录 `provisional`；{}。
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
      "AUC": 0.8606116757792783,
      "logloss": 0.20346152782440186
    },
    "all_queries": {
      "AUC": 0.8606116757792783,
      "logloss": 0.20346152782440186
    },
    "q3_half": {
      "AUC": 0.8606803468994726,
      "logloss": 0.2035728245973587
    },
    "q0_zero": {
      "AUC": 0.8606075022445845,
      "logloss": 0.205962136387825
    },
    "q0_shuffle": {
      "AUC": 0.8577133734819485,
      "logloss": 0.20798979699611664
    },
    "q1_zero": {
      "AUC": 0.8608681770818267,
      "logloss": 0.20505791902542114
    },
    "q1_shuffle": {
      "AUC": 0.8593003178070634,
      "logloss": 0.20591703057289124
    },
    "q2_zero": {
      "AUC": 0.8605719934058828,
      "logloss": 0.20448148250579834
    },
    "q2_shuffle": {
      "AUC": 0.8577347037080993,
      "logloss": 0.20753346383571625
    },
    "q3_zero": {
      "AUC": 0.8605858763226557,
      "logloss": 0.20780198276042938
    },
    "q3_shuffle": {
      "AUC": 0.8570416202621426,
      "logloss": 0.2086593359708786
    },
    "q4_zero": {
      "AUC": 0.858881944187236,
      "logloss": 0.20860156416893005
    },
    "q4_shuffle": {
      "AUC": 0.8536083029211634,
      "logloss": 0.21099580824375153
    },
    "q5_zero": {
      "AUC": 0.8595379558310938,
      "logloss": 0.20650514960289001
    },
    "q5_shuffle": {
      "AUC": 0.856116633178134,
      "logloss": 0.20897236466407776
    },
    "q6_zero": {
      "AUC": 0.8605339436956038,
      "logloss": 0.20377492904663086
    },
    "q6_shuffle": {
      "AUC": 0.8590388868334747,
      "logloss": 0.20554400980472565
    },
    "q7_zero": {
      "AUC": 0.8602221409462482,
      "logloss": 0.20470768213272095
    },
    "q7_shuffle": {
      "AUC": 0.8566507083316323,
      "logloss": 0.20755529403686523
    },
    "inference_placement_steps1": {
      "AUC": 0.8592072285519518,
      "logloss": 0.20694272220134735
    },
    "inference_placement_steps3": {
      "AUC": 0.8584168379618228,
      "logloss": 0.2037179321050644
    }
  },
  "checkpoint_unchanged": true,
  "note": "subset diagnostics; within-block shuffle; blockwise loss means in toolkit differ from pooled full-validation metrics"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 3](#phase-3)。
- **后续假设**：分开用户上下文、目标物品及全部字段回读；见K04。
- **证据**：`qformer-loop/reports/round3/place_field2/summary.json`（SHA256前缀 `c4eda1970110`）；`qformer-loop/reports/round3/place_field2/toolkit-hooks/hook_report.json`（SHA256前缀 `0241bd236dc4`）；`qformer-loop/reports/round3/place_field2/toolkit-perturbations/perturbation_report.json`（SHA256前缀 `a63649d65ae4`）

<a id="phase-4"></a>
## 阶段 4：上下文回读来源

**方法**：历史后回读context、target或all，并控制归一化。

**归因与边界**：只回读用户/上下文的三seed平均表现较好，但源token数量和计算量也改变；不是跨数据集通用结论。

**历史后续假设**：减少独立block并检查额外attention与保留深度的交互；见K05。

<a id="k052"></a>
### K052 · `feedback_context-s20262027-20260906T143634603091Z`

- **方法**：{"model":"QFormerCross11Feedback","loop_placement":"field_feedback","placement_steps":2,"feedback_source":"context","query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Feedback","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Restrict the additional history-conditioned field read to user/context K/V
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262027],"optimizer_steps":1240}。
- **结果**：logloss=0.189167991；AUC=0.869753845；UAUC=0.789646966；gAUC=0.802264072；NDCG(10)=0.404876776。
- **开销**：参数 11408145；训练含验证 531.3591391444206 秒；峰值CUDA 22261132288 bytes；更新 1240。
- **原记录差值**：+0.000206825；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":16.164608001708984,"mean_ms":16.25972318649292,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计提高0.000206825，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 4](#phase-4)。
- **后续假设**：减少独立block并检查额外attention与保留深度的交互；见K05。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/feedback_context-s20262027-20260906T143634603091Z/manifest.json`（SHA256前缀 `560015ffb43f`）；`qformer-loop/runs/feedback_context-s20262027-20260906T143634603091Z/result.json`（SHA256前缀 `22fd089d8cc7`）

<a id="k053"></a>
### K053 · `feedback_context-s20262028-20260906T143634525420Z`

- **方法**：{"model":"QFormerCross11Feedback","loop_placement":"field_feedback","placement_steps":2,"feedback_source":"context","query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Feedback","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Restrict the additional history-conditioned field read to user/context K/V
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262028],"optimizer_steps":1240}。
- **结果**：logloss=0.188276417；AUC=0.871183452；UAUC=0.792218717；gAUC=0.803561726；NDCG(10)=0.408801941。
- **开销**：参数 11408145；训练含验证 540.0088440701365 秒；峰值CUDA 22261132288 bytes；更新 1240。
- **原记录差值**：+0.001453554；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":16.164608001708984,"mean_ms":16.25972318649292,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计提高0.001453554，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 4](#phase-4)。
- **后续假设**：减少独立block并检查额外attention与保留深度的交互；见K05。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/feedback_context-s20262028-20260906T143634525420Z/manifest.json`（SHA256前缀 `de75c78601a9`）；`qformer-loop/runs/feedback_context-s20262028-20260906T143634525420Z/result.json`（SHA256前缀 `c8e547efd432`）

<a id="k054"></a>
### K054 · `feedback_context-s20262029-20260906T143634581333Z`

- **方法**：{"model":"QFormerCross11Feedback","loop_placement":"field_feedback","placement_steps":2,"feedback_source":"context","query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Feedback","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Restrict the additional history-conditioned field read to user/context K/V
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262029],"optimizer_steps":1240}。
- **结果**：logloss=0.189718533；AUC=0.868723365；UAUC=0.790709927；gAUC=0.802620954；NDCG(10)=0.407892394。
- **开销**：参数 11408145；训练含验证 536.9447413310409 秒；峰值CUDA 22261132288 bytes；更新 1240。
- **原记录差值**：-0.000091796；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":16.164608001708984,"mean_ms":16.25972318649292,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000091796，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 4](#phase-4)。
- **后续假设**：减少独立block并检查额外attention与保留深度的交互；见K05。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/feedback_context-s20262029-20260906T143634581333Z/manifest.json`（SHA256前缀 `0da9edca11fd`）；`qformer-loop/runs/feedback_context-s20262029-20260906T143634581333Z/result.json`（SHA256前缀 `5658252bf27b`）

<a id="k055"></a>
### K055 · `feedback_no_norm-s20262027-20260906T143634406327Z`

- **方法**：{"model":"QFormerCross11Feedback","loop_placement":"field_feedback","placement_steps":2,"feedback_final_norm":false,"query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Feedback","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Remove only the additional field final norm after the feedback read
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262027],"optimizer_steps":1240}。
- **结果**：logloss=0.189986565；AUC=0.867742141；UAUC=0.789221680；gAUC=0.801914752；NDCG(10)=0.406378335。
- **开销**：参数 11408145；训练含验证 539.4663129150867 秒；峰值CUDA 22371101696 bytes；更新 1240。
- **原记录差值**：-0.001804879；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":16.214624404907227,"mean_ms":16.240484094619752,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.001804879，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 4](#phase-4)。
- **后续假设**：减少独立block并检查额外attention与保留深度的交互；见K05。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/feedback_no_norm-s20262027-20260906T143634406327Z/manifest.json`（SHA256前缀 `0d69678d1fdc`）；`qformer-loop/runs/feedback_no_norm-s20262027-20260906T143634406327Z/result.json`（SHA256前缀 `43a6ff31ff7a`）

<a id="k056"></a>
### K056 · `feedback_no_norm-s20262028-20260906T143634491754Z`

- **方法**：{"model":"QFormerCross11Feedback","loop_placement":"field_feedback","placement_steps":2,"feedback_final_norm":false,"query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Feedback","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Remove only the additional field final norm after the feedback read
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262028],"optimizer_steps":1240}。
- **结果**：logloss=0.189210838；AUC=0.869317214；UAUC=0.789100036；gAUC=0.801445186；NDCG(10)=0.410471955。
- **开销**：参数 11408145；训练含验证 538.0602175816894 秒；峰值CUDA 22371101696 bytes；更新 1240。
- **原记录差值**：-0.000412684；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":16.214624404907227,"mean_ms":16.240484094619752,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000412684，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 4](#phase-4)。
- **后续假设**：减少独立block并检查额外attention与保留深度的交互；见K05。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/feedback_no_norm-s20262028-20260906T143634491754Z/manifest.json`（SHA256前缀 `7a6dafa51d65`）；`qformer-loop/runs/feedback_no_norm-s20262028-20260906T143634491754Z/result.json`（SHA256前缀 `20e959f56842`）

<a id="k057"></a>
### K057 · `feedback_target-s20262027-20260906T143634564447Z`

- **方法**：{"model":"QFormerCross11Feedback","loop_placement":"field_feedback","placement_steps":2,"feedback_source":"target","query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Feedback","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Restrict the additional history-conditioned field read to target-item K/V
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262027],"optimizer_steps":1240}。
- **结果**：logloss=0.189640823；AUC=0.868861815；UAUC=0.789162856；gAUC=0.800758792；NDCG(10)=0.406331941。
- **开销**：参数 11408145；训练含验证 527.1802174001932 秒；峰值CUDA 21675371520 bytes；更新 1240。
- **原记录差值**：-0.000685205；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":15.75871992111206,"mean_ms":15.762597680091858,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000685205，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 4](#phase-4)。
- **后续假设**：减少独立block并检查额外attention与保留深度的交互；见K05。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/feedback_target-s20262027-20260906T143634564447Z/manifest.json`（SHA256前缀 `88d9919aef97`）；`qformer-loop/runs/feedback_target-s20262027-20260906T143634564447Z/result.json`（SHA256前缀 `5cf8e675ccec`）

<a id="k058"></a>
### K058 · `feedback_target-s20262028-20260906T143634606134Z`

- **方法**：{"model":"QFormerCross11Feedback","loop_placement":"field_feedback","placement_steps":2,"feedback_source":"target","query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Feedback","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Restrict the additional history-conditioned field read to target-item K/V
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262028],"optimizer_steps":1240}。
- **结果**：logloss=0.190222942；AUC=0.867314949；UAUC=0.791276331；gAUC=0.802643136；NDCG(10)=0.412009202。
- **开销**：参数 11408145；训练含验证 525.4551426917315 秒；峰值CUDA 21675371520 bytes；更新 1240。
- **原记录差值**：-0.002414949；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":15.75871992111206,"mean_ms":15.762597680091858,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.002414949，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 4](#phase-4)。
- **后续假设**：减少独立block并检查额外attention与保留深度的交互；见K05。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/feedback_target-s20262028-20260906T143634606134Z/manifest.json`（SHA256前缀 `40da38f7605b`）；`qformer-loop/runs/feedback_target-s20262028-20260906T143634606134Z/result.json`（SHA256前缀 `40a997fe0a56`）

<a id="k059"></a>
### K059 · `feedback_target-s20262029-20260906T143634564353Z`

- **方法**：{"model":"QFormerCross11Feedback","loop_placement":"field_feedback","placement_steps":2,"feedback_source":"target","query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Feedback","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Restrict the additional history-conditioned field read to target-item K/V
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262029],"optimizer_steps":1240}。
- **结果**：logloss=0.189365692；AUC=0.869289277；UAUC=0.790954333；gAUC=0.802164120；NDCG(10)=0.409963753。
- **开销**：参数 11408145；训练含验证 525.1321871057153 秒；峰值CUDA 21675371520 bytes；更新 1240。
- **原记录差值**：+0.000474116；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":15.75871992111206,"mean_ms":15.762597680091858,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计提高0.000474116，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 4](#phase-4)。
- **后续假设**：减少独立block并检查额外attention与保留深度的交互；见K05。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/feedback_target-s20262029-20260906T143634564353Z/manifest.json`（SHA256前缀 `f3c9c91882e5`）；`qformer-loop/runs/feedback_target-s20262029-20260906T143634564353Z/result.json`（SHA256前缀 `507c83ff08ff`）

<a id="k060"></a>
### K060 · `feedback_no_norm-s20262029-20260906T144556381002Z`

- **方法**：{"model":"QFormerCross11Feedback","loop_placement":"field_feedback","placement_steps":2,"feedback_final_norm":false,"query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Feedback","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Remove only the additional field final norm after the feedback read
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262029],"optimizer_steps":1240}。
- **结果**：logloss=0.189838037；AUC=0.868563636；UAUC=0.788734006；gAUC=0.800862971；NDCG(10)=0.406967760。
- **开销**：参数 11408145；训练含验证 535.9332623332739 秒；峰值CUDA 22371101696 bytes；更新 1240。
- **原记录差值**：-0.000251525；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":16.214624404907227,"mean_ms":16.240484094619752,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000251525，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 4](#phase-4)。
- **后续假设**：减少独立block并检查额外attention与保留深度的交互；见K05。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/feedback_no_norm-s20262029-20260906T144556381002Z/manifest.json`（SHA256前缀 `424d42d21b93`）；`qformer-loop/runs/feedback_no_norm-s20262029-20260906T144556381002Z/result.json`（SHA256前缀 `bc9d61668de9`）

<a id="k061"></a>
### K061 · `field_norm_matched-s20262027-20260906T144556294569Z`

- **方法**：{"model":"QFormerCross11Feedback","loop_placement":"field","placement_steps":2,"field_norm_each_pass":true,"query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Feedback","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Normalize each pre-history field read to match the feedback interface norm count
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262027],"optimizer_steps":1240}。
- **结果**：logloss=0.189766540；AUC=0.868846324；UAUC=0.791665089；gAUC=0.803478763；NDCG(10)=0.411774662。
- **开销**：参数 11408145；训练含验证 540.8656810298562 秒；峰值CUDA 22438734848 bytes；更新 1240。
- **原记录差值**：-0.000700697；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":16.241040229797363,"mean_ms":16.26253538131714,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000700697，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 4](#phase-4)。
- **后续假设**：减少独立block并检查额外attention与保留深度的交互；见K05。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/field_norm_matched-s20262027-20260906T144556294569Z/manifest.json`（SHA256前缀 `3cf5d998156c`）；`qformer-loop/runs/field_norm_matched-s20262027-20260906T144556294569Z/result.json`（SHA256前缀 `a7154e874359`）

<a id="k062"></a>
### K062 · `field_norm_matched-s20262028-20260906T144556444817Z`

- **方法**：{"model":"QFormerCross11Feedback","loop_placement":"field","placement_steps":2,"field_norm_each_pass":true,"query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Feedback","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Normalize each pre-history field read to match the feedback interface norm count
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262028],"optimizer_steps":1240}。
- **结果**：logloss=0.189397685；AUC=0.869677975；UAUC=0.789939069；gAUC=0.801765908；NDCG(10)=0.412502534。
- **开销**：参数 11408145；训练含验证 539.3790066242218 秒；峰值CUDA 22438734848 bytes；更新 1240。
- **原记录差值**：-0.000051923；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":16.241040229797363,"mean_ms":16.26253538131714,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000051923，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 4](#phase-4)。
- **后续假设**：减少独立block并检查额外attention与保留深度的交互；见K05。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/field_norm_matched-s20262028-20260906T144556444817Z/manifest.json`（SHA256前缀 `c9ce5ac16840`）；`qformer-loop/runs/field_norm_matched-s20262028-20260906T144556444817Z/result.json`（SHA256前缀 `d9d8d38c8bd4`）

<a id="k063"></a>
### K063 · `field_norm_matched-s20262029-20260906T144601305121Z`

- **方法**：{"model":"QFormerCross11Feedback","loop_placement":"field","placement_steps":2,"field_norm_each_pass":true,"query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Feedback","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Normalize each pre-history field read to match the feedback interface norm count
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262029],"optimizer_steps":1240}。
- **结果**：logloss=0.190046090；AUC=0.867269779；UAUC=0.788922698；gAUC=0.801454978；NDCG(10)=0.408180639。
- **开销**：参数 11408145；训练含验证 536.5238933637738 秒；峰值CUDA 22438734848 bytes；更新 1240。
- **原记录差值**：-0.001545381；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":16.241040229797363,"mean_ms":16.26253538131714,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.001545381，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 4](#phase-4)。
- **后续假设**：减少独立block并检查额外attention与保留深度的交互；见K05。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/field_norm_matched-s20262029-20260906T144601305121Z/manifest.json`（SHA256前缀 `a8511983c5e5`）；`qformer-loop/runs/field_norm_matched-s20262029-20260906T144601305121Z/result.json`（SHA256前缀 `7f940e0f7e64`）

<a id="k064"></a>
### K064 · `diagnostics-round4-source-probe`

- **方法**：frozen checkpoint intervention using existing rec-harness probes
- **原假设**：Check feedback source dependence and recurrent query use
- **状态／预算**：原记录 `provisional`；{}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "rows": 57408,
  "users": 7176,
  "positives": 4472,
  "precision": "bf16",
  "scores": {
    "all": {
      "AUC": 0.8599354596653387,
      "logloss": 0.20402722992247616
    },
    "target": {
      "AUC": 0.8536651166406635,
      "logloss": 0.2094827721055603
    },
    "context": {
      "AUC": 0.8598917051386586,
      "logloss": 0.20397213408146567
    },
    "no_extra_norm": {
      "AUC": 0.8599555352965461,
      "logloss": 0.2040186157904741
    }
  },
  "note": "source widths are 7/30/37, not equal-cost controls; subset results are diagnostic, not model-selection scores"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 4](#phase-4)。
- **后续假设**：减少独立block并检查额外attention与保留深度的交互；见K05。
- **证据**：`qformer-loop/reports/round4/source-probe/summary.json`（SHA256前缀 `e9e487dabdc0`）

<a id="k065"></a>
### K065 · `diagnostics-round4-feedback_context`

- **方法**：frozen checkpoint intervention using existing rec-harness probes
- **原假设**：Check feedback source dependence and recurrent query use
- **状态／预算**：原记录 `provisional`；{}。
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
      "AUC": 0.8609137522496535,
      "logloss": 0.20329073071479797
    },
    "all_queries": {
      "AUC": 0.8609137522496535,
      "logloss": 0.20329073071479797
    },
    "q3_half": {
      "AUC": 0.8610682913116403,
      "logloss": 0.20333124697208405
    },
    "q0_zero": {
      "AUC": 0.860653113318327,
      "logloss": 0.20364908874034882
    },
    "q0_shuffle": {
      "AUC": 0.8585074856146538,
      "logloss": 0.20676882565021515
    },
    "q1_zero": {
      "AUC": 0.8605750855388746,
      "logloss": 0.2041037529706955
    },
    "q1_shuffle": {
      "AUC": 0.8575151263597613,
      "logloss": 0.20750540494918823
    },
    "q2_zero": {
      "AUC": 0.8602230681637232,
      "logloss": 0.20705154538154602
    },
    "q2_shuffle": {
      "AUC": 0.8581751024391556,
      "logloss": 0.20751148462295532
    },
    "q3_zero": {
      "AUC": 0.8609714488322618,
      "logloss": 0.20398524403572083
    },
    "q3_shuffle": {
      "AUC": 0.857630654700191,
      "logloss": 0.20799806714057922
    },
    "q4_zero": {
      "AUC": 0.85957497060615,
      "logloss": 0.20587661862373352
    },
    "q4_shuffle": {
      "AUC": 0.8558553373797582,
      "logloss": 0.20821338891983032
    },
    "q5_zero": {
      "AUC": 0.8606274194673393,
      "logloss": 0.20704063773155212
    },
    "q5_shuffle": {
      "AUC": 0.8583941644319952,
      "logloss": 0.20604844391345978
    },
    "q6_zero": {
      "AUC": 0.8605063088130454,
      "logloss": 0.20530273020267487
    },
    "q6_shuffle": {
      "AUC": 0.85510651105544,
      "logloss": 0.2095365673303604
    },
    "q7_zero": {
      "AUC": 0.8604054448710875,
      "logloss": 0.20401215553283691
    },
    "q7_shuffle": {
      "AUC": 0.8538955713694032,
      "logloss": 0.21046210825443268
    },
    "inference_placement_steps1": {
      "AUC": 0.8562765919213076,
      "logloss": 0.20807570219039917
    },
    "inference_placement_steps3": {
      "AUC": 0.857244095833954,
      "logloss": 0.2059764862060547
    }
  },
  "checkpoint_unchanged": true,
  "note": "subset diagnostics; within-block shuffle; blockwise loss means in toolkit differ from pooled full-validation metrics"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 4](#phase-4)。
- **后续假设**：减少独立block并检查额外attention与保留深度的交互；见K05。
- **证据**：`qformer-loop/reports/round4/feedback_context/summary.json`（SHA256前缀 `613c51c2a393`）

<a id="phase-5"></a>
## 阶段 5：紧凑共享循环与额外attention

**方法**：减少历史/字段块；有无额外共享Query self-attention对照。

**归因与边界**：compact_seq1三seed平均AUC=0.870296，attn_full=0.870442；attention收益依赖底层深度，单历史块时也可能退化。

**历史后续假设**：保持少量共享轮次，移动回读/attention位置并进一步压缩；见K06。

<a id="k066"></a>
### K066 · `compact_seq1-s20262027-20260906T150445272533Z`

- **方法**：{"model":"QFormerCross11Compact","loop_placement":"field_feedback","placement_steps":2,"num_ns_layers":2,"num_unified_layers":1,"extra_self_attention":false,"feedback_source":"context","query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Compact","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Reduce to one recurrent history block without a separate coda
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262027],"optimizer_steps":1240}。
- **结果**：logloss=0.188829919；AUC=0.870197440；UAUC=0.790913911；gAUC=0.803285376；NDCG(10)=0.409850314。
- **开销**：参数 9037073；训练含验证 464.7255721464753 秒；峰值CUDA 18463417344 bytes；更新 1240。
- **原记录差值**：+0.000650420；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":13.755311965942383,"mean_ms":13.763571906089783,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计提高0.000650420，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 5](#phase-5)。
- **后续假设**：保持少量共享轮次，移动回读/attention位置并进一步压缩；见K06。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/compact_seq1-s20262027-20260906T150445272533Z/manifest.json`（SHA256前缀 `a1962ebb42d7`）；`qformer-loop/runs/compact_seq1-s20262027-20260906T150445272533Z/result.json`（SHA256前缀 `4971ec6bfbde`）

<a id="k067"></a>
### K067 · `compact_seq1-s20262028-20260906T150445070621Z`

- **方法**：{"model":"QFormerCross11Compact","loop_placement":"field_feedback","placement_steps":2,"num_ns_layers":2,"num_unified_layers":1,"extra_self_attention":false,"feedback_source":"context","query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Compact","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Reduce to one recurrent history block without a separate coda
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262028],"optimizer_steps":1240}。
- **结果**：logloss=0.188293597；AUC=0.870753503；UAUC=0.790237096；gAUC=0.802978688；NDCG(10)=0.409335018。
- **开销**：参数 9037073；训练含验证 464.6954624503851 秒；峰值CUDA 18463417344 bytes；更新 1240。
- **原记录差值**：+0.001023605；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":13.755311965942383,"mean_ms":13.763571906089783,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计提高0.001023605，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 5](#phase-5)。
- **后续假设**：保持少量共享轮次，移动回读/attention位置并进一步压缩；见K06。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/compact_seq1-s20262028-20260906T150445070621Z/manifest.json`（SHA256前缀 `ba56a4d7189a`）；`qformer-loop/runs/compact_seq1-s20262028-20260906T150445070621Z/result.json`（SHA256前缀 `024298576f1b`）

<a id="k068"></a>
### K068 · `compact_seq1-s20262029-20260906T150444924864Z`

- **方法**：{"model":"QFormerCross11Compact","loop_placement":"field_feedback","placement_steps":2,"num_ns_layers":2,"num_unified_layers":1,"extra_self_attention":false,"feedback_source":"context","query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Compact","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Reduce to one recurrent history block without a separate coda
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262029],"optimizer_steps":1240}。
- **结果**：logloss=0.189084579；AUC=0.869937355；UAUC=0.790091521；gAUC=0.802190754；NDCG(10)=0.406481663。
- **开销**：参数 9037073；训练含验证 465.1099829599261 秒；峰值CUDA 18463417344 bytes；更新 1240。
- **原记录差值**：+0.001122194；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":13.755311965942383,"mean_ms":13.763571906089783,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计提高0.001122194，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 5](#phase-5)。
- **后续假设**：保持少量共享轮次，移动回读/attention位置并进一步压缩；见K06。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/compact_seq1-s20262029-20260906T150444924864Z/manifest.json`（SHA256前缀 `dabd694f686e`）；`qformer-loop/runs/compact_seq1-s20262029-20260906T150444924864Z/result.json`（SHA256前缀 `80907d02ce27`）

<a id="k069"></a>
### K069 · `attn_full-s20262027-20260906T150445276311Z`

- **方法**：{"model":"QFormerCross11Compact","loop_placement":"field_feedback","placement_steps":2,"num_ns_layers":2,"num_unified_layers":3,"extra_self_attention":true,"feedback_source":"context","query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Compact","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Add shared Query self-attention before each recurrent field read
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262027],"optimizer_steps":1240}。
- **结果**：logloss=0.189551239；AUC=0.869118227；UAUC=0.789450847；gAUC=0.801644525；NDCG(10)=0.407239370。
- **开销**：参数 11672081；训练含验证 565.6397142484784 秒；峰值CUDA 23215889408 bytes；更新 1240。
- **原记录差值**：-0.000428793；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":17.28372859954834,"mean_ms":17.309410429000856,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000428793，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 5](#phase-5)。
- **后续假设**：保持少量共享轮次，移动回读/attention位置并进一步压缩；见K06。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/attn_full-s20262027-20260906T150445276311Z/manifest.json`（SHA256前缀 `9d5639a187e4`）；`qformer-loop/runs/attn_full-s20262027-20260906T150445276311Z/result.json`（SHA256前缀 `e05692592497`）

<a id="k070"></a>
### K070 · `attn_full-s20262028-20260906T150445272581Z`

- **方法**：{"model":"QFormerCross11Compact","loop_placement":"field_feedback","placement_steps":2,"num_ns_layers":2,"num_unified_layers":3,"extra_self_attention":true,"feedback_source":"context","query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Compact","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Add shared Query self-attention before each recurrent field read
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262028],"optimizer_steps":1240}。
- **结果**：logloss=0.188511913；AUC=0.870901360；UAUC=0.789498858；gAUC=0.801935943；NDCG(10)=0.411350216。
- **开销**：参数 11672081；训练含验证 565.4069591835141 秒；峰值CUDA 23215889408 bytes；更新 1240。
- **原记录差值**：+0.001171462；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":17.28372859954834,"mean_ms":17.309410429000856,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计提高0.001171462，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 5](#phase-5)。
- **后续假设**：保持少量共享轮次，移动回读/attention位置并进一步压缩；见K06。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/attn_full-s20262028-20260906T150445272581Z/manifest.json`（SHA256前缀 `147c055e29ef`）；`qformer-loop/runs/attn_full-s20262028-20260906T150445272581Z/result.json`（SHA256前缀 `ca2272f7c0ef`）

<a id="k071"></a>
### K071 · `attn_full-s20262029-20260906T151301200510Z`

- **方法**：{"model":"QFormerCross11Compact","loop_placement":"field_feedback","placement_steps":2,"num_ns_layers":2,"num_unified_layers":3,"extra_self_attention":true,"feedback_source":"context","query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Compact","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":3,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Add shared Query self-attention before each recurrent field read
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262029],"optimizer_steps":1240}。
- **结果**：logloss=0.188414442；AUC=0.871305700；UAUC=0.789684338；gAUC=0.801397659；NDCG(10)=0.408568041。
- **开销**：参数 11672081；训练含验证 570.3779691159725 秒；峰值CUDA 23215889408 bytes；更新 1240。
- **原记录差值**：+0.002490539；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":17.28372859954834,"mean_ms":17.309410429000856,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计提高0.002490539，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 5](#phase-5)。
- **后续假设**：保持少量共享轮次，移动回读/attention位置并进一步压缩；见K06。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/attn_full-s20262029-20260906T151301200510Z/manifest.json`（SHA256前缀 `65b03b270e26`）；`qformer-loop/runs/attn_full-s20262029-20260906T151301200510Z/result.json`（SHA256前缀 `1f89a02ab083`）

<a id="k072"></a>
### K072 · `attn_seq1-s20262027-20260906T151417256474Z`

- **方法**：{"model":"QFormerCross11Compact","loop_placement":"field_feedback","placement_steps":2,"num_ns_layers":2,"num_unified_layers":1,"extra_self_attention":true,"feedback_source":"context","query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Compact","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Combine extra shared attention with one recurrent history block
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262027],"optimizer_steps":1240}。
- **结果**：logloss=0.189559933；AUC=0.869009500；UAUC=0.788941296；gAUC=0.801142267；NDCG(10)=0.407748230。
- **开销**：参数 9301009；训练含验证 502.0937755405903 秒；峰值CUDA 19418174464 bytes；更新 1240。
- **原记录差值**：-0.000537520；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":14.693471908569336,"mean_ms":14.705157613754272,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000537520，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 5](#phase-5)。
- **后续假设**：保持少量共享轮次，移动回读/attention位置并进一步压缩；见K06。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/attn_seq1-s20262027-20260906T151417256474Z/manifest.json`（SHA256前缀 `406325d93425`）；`qformer-loop/runs/attn_seq1-s20262027-20260906T151417256474Z/result.json`（SHA256前缀 `a52a09a2b677`）

<a id="k073"></a>
### K073 · `attn_seq1-s20262028-20260906T151417316199Z`

- **方法**：{"model":"QFormerCross11Compact","loop_placement":"field_feedback","placement_steps":2,"num_ns_layers":2,"num_unified_layers":1,"extra_self_attention":true,"feedback_source":"context","query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Compact","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Combine extra shared attention with one recurrent history block
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262028],"optimizer_steps":1240}。
- **结果**：logloss=0.189651921；AUC=0.869364766；UAUC=0.789540279；gAUC=0.802105416；NDCG(10)=0.404735076。
- **开销**：参数 9301009；训练含验证 494.6134649515152 秒；峰值CUDA 19418174464 bytes；更新 1240。
- **原记录差值**：-0.000365132；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":14.693471908569336,"mean_ms":14.705157613754272,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000365132，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 5](#phase-5)。
- **后续假设**：保持少量共享轮次，移动回读/attention位置并进一步压缩；见K06。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/attn_seq1-s20262028-20260906T151417316199Z/manifest.json`（SHA256前缀 `617c761bf679`）；`qformer-loop/runs/attn_seq1-s20262028-20260906T151417316199Z/result.json`（SHA256前缀 `28de0b013f73`）

<a id="k074"></a>
### K074 · `attn_seq1-s20262029-20260906T151447655718Z`

- **方法**：{"model":"QFormerCross11Compact","loop_placement":"field_feedback","placement_steps":2,"num_ns_layers":2,"num_unified_layers":1,"extra_self_attention":true,"feedback_source":"context","query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Compact","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Combine extra shared attention with one recurrent history block
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262029],"optimizer_steps":1240}。
- **结果**：logloss=0.188918660；AUC=0.870924950；UAUC=0.790545547；gAUC=0.802205281；NDCG(10)=0.410191544。
- **开销**：参数 9301009；训练含验证 499.7980373799801 秒；峰值CUDA 19418174464 bytes；更新 1240。
- **原记录差值**：+0.002109789；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":14.693471908569336,"mean_ms":14.705157613754272,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计提高0.002109789，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 5](#phase-5)。
- **后续假设**：保持少量共享轮次，移动回读/attention位置并进一步压缩；见K06。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/attn_seq1-s20262029-20260906T151447655718Z/manifest.json`（SHA256前缀 `d91a76f47977`）；`qformer-loop/runs/attn_seq1-s20262029-20260906T151447655718Z/result.json`（SHA256前缀 `ee3ef27356ba`）

<a id="k075"></a>
### K075 · `attn_seq2-s20262027-20260906T151301214301Z`

- **方法**：{"model":"QFormerCross11Compact","loop_placement":"field_feedback","placement_steps":2,"num_ns_layers":2,"num_unified_layers":2,"extra_self_attention":true,"feedback_source":"context","query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Compact","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":2,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Combine extra shared attention with two independent history blocks
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262027],"optimizer_steps":1240}。
- **结果**：logloss=0.189231553；AUC=0.869618238；UAUC=0.790526485；gAUC=0.802495821；NDCG(10)=0.411744026。
- **开销**：参数 10486545；训练含验证 563.9541566595435 秒；峰值CUDA 23199289856 bytes；更新 1240。
- **原记录差值**：+0.000071218；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":17.295936584472656,"mean_ms":17.302246427536012,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计提高0.000071218，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 5](#phase-5)。
- **后续假设**：保持少量共享轮次，移动回读/attention位置并进一步压缩；见K06。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/attn_seq2-s20262027-20260906T151301214301Z/manifest.json`（SHA256前缀 `8665b793fa7e`）；`qformer-loop/runs/attn_seq2-s20262027-20260906T151301214301Z/result.json`（SHA256前缀 `550358c04123`）

<a id="k076"></a>
### K076 · `attn_seq2-s20262028-20260906T151301220984Z`

- **方法**：{"model":"QFormerCross11Compact","loop_placement":"field_feedback","placement_steps":2,"num_ns_layers":2,"num_unified_layers":2,"extra_self_attention":true,"feedback_source":"context","query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Compact","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":2,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Combine extra shared attention with two independent history blocks
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262028],"optimizer_steps":1240}。
- **结果**：logloss=0.189578748；AUC=0.869215896；UAUC=0.789929357；gAUC=0.802214338；NDCG(10)=0.406759532。
- **开销**：参数 10486545；训练含验证 570.0389040261507 秒；峰值CUDA 23199289856 bytes；更新 1240。
- **原记录差值**：-0.000514002；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":17.295936584472656,"mean_ms":17.302246427536012,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000514002，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 5](#phase-5)。
- **后续假设**：保持少量共享轮次，移动回读/attention位置并进一步压缩；见K06。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/attn_seq2-s20262028-20260906T151301220984Z/manifest.json`（SHA256前缀 `f0ffccc42185`）；`qformer-loop/runs/attn_seq2-s20262028-20260906T151301220984Z/result.json`（SHA256前缀 `751b2c0131ff`）

<a id="k077"></a>
### K077 · `attn_seq2-s20262029-20260906T151412118962Z`

- **方法**：{"model":"QFormerCross11Compact","loop_placement":"field_feedback","placement_steps":2,"num_ns_layers":2,"num_unified_layers":2,"extra_self_attention":true,"feedback_source":"context","query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Compact","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":2,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Combine extra shared attention with two independent history blocks
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262029],"optimizer_steps":1240}。
- **结果**：logloss=0.190039875；AUC=0.869180171；UAUC=0.790418586；gAUC=0.802339364；NDCG(10)=0.409298440。
- **开销**：参数 10486545；训练含验证 566.1920337826014 秒；峰值CUDA 23199289856 bytes；更新 1240。
- **原记录差值**：+0.000365010；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":17.295936584472656,"mean_ms":17.302246427536012,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计提高0.000365010，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 5](#phase-5)。
- **后续假设**：保持少量共享轮次，移动回读/attention位置并进一步压缩；见K06。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/attn_seq2-s20262029-20260906T151412118962Z/manifest.json`（SHA256前缀 `d1db6f54afc7`）；`qformer-loop/runs/attn_seq2-s20262029-20260906T151412118962Z/result.json`（SHA256前缀 `173a54cde845`）

<a id="k078"></a>
### K078 · `attn_small-s20262027-20260906T151447646450Z`

- **方法**：{"model":"QFormerCross11Compact","loop_placement":"field_feedback","placement_steps":2,"num_ns_layers":1,"num_unified_layers":1,"extra_self_attention":true,"feedback_source":"context","query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Compact","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Use one field and one history block plus shared extra attention in the recurrent core
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262027],"optimizer_steps":1240}。
- **结果**：logloss=0.189229006；AUC=0.869647548；UAUC=0.789756534；gAUC=0.801531417；NDCG(10)=0.409940261。
- **开销**：参数 8115473；训练含验证 457.06169555336237 秒；峰值CUDA 17238033920 bytes；更新 1240。
- **原记录差值**：+0.000100528；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":13.124783992767334,"mean_ms":13.171357607841491,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计提高0.000100528，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 5](#phase-5)。
- **后续假设**：保持少量共享轮次，移动回读/attention位置并进一步压缩；见K06。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/attn_small-s20262027-20260906T151447646450Z/manifest.json`（SHA256前缀 `bcf214df3b18`）；`qformer-loop/runs/attn_small-s20262027-20260906T151447646450Z/result.json`（SHA256前缀 `9ad8e7bc425e`）

<a id="k079"></a>
### K079 · `attn_small-s20262028-20260906T152254190441Z`

- **方法**：{"model":"QFormerCross11Compact","loop_placement":"field_feedback","placement_steps":2,"num_ns_layers":1,"num_unified_layers":1,"extra_self_attention":true,"feedback_source":"context","query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Compact","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Use one field and one history block plus shared extra attention in the recurrent core
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262028],"optimizer_steps":1240}。
- **结果**：logloss=0.189947407；AUC=0.868235420；UAUC=0.790406892；gAUC=0.802461143；NDCG(10)=0.407417134。
- **开销**：参数 8115473；训练含验证 451.95081778615713 秒；峰值CUDA 17238033920 bytes；更新 1240。
- **原记录差值**：-0.001494479；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":13.124783992767334,"mean_ms":13.171357607841491,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.001494479，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 5](#phase-5)。
- **后续假设**：保持少量共享轮次，移动回读/attention位置并进一步压缩；见K06。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/attn_small-s20262028-20260906T152254190441Z/manifest.json`（SHA256前缀 `4415a572828a`）；`qformer-loop/runs/attn_small-s20262028-20260906T152254190441Z/result.json`（SHA256前缀 `a4a371cf699d`）

<a id="k080"></a>
### K080 · `attn_small-s20262029-20260906T152259162068Z`

- **方法**：{"model":"QFormerCross11Compact","loop_placement":"field_feedback","placement_steps":2,"num_ns_layers":1,"num_unified_layers":1,"extra_self_attention":true,"feedback_source":"context","query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Compact","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Use one field and one history block plus shared extra attention in the recurrent core
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262029],"optimizer_steps":1240}。
- **结果**：logloss=0.190308177；AUC=0.867576095；UAUC=0.788774705；gAUC=0.800665092；NDCG(10)=0.408581009。
- **开销**：参数 8115473；训练含验证 457.37276799976826 秒；峰值CUDA 17238033920 bytes；更新 1240。
- **原记录差值**：-0.001239066；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":13.124783992767334,"mean_ms":13.171357607841491,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.001239066，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 5](#phase-5)。
- **后续假设**：保持少量共享轮次，移动回读/attention位置并进一步压缩；见K06。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/attn_small-s20262029-20260906T152259162068Z/manifest.json`（SHA256前缀 `7b0f6948da04`）；`qformer-loop/runs/attn_small-s20262029-20260906T152259162068Z/result.json`（SHA256前缀 `b0f6a6aee61c`）

<a id="k081"></a>
### K081 · `compact_seq2-s20262027-20260906T150445270379Z`

- **方法**：{"model":"QFormerCross11Compact","loop_placement":"field_feedback","placement_steps":2,"num_ns_layers":2,"num_unified_layers":2,"extra_self_attention":false,"feedback_source":"context","query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Compact","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":2,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Reuse the first history block with two independent history blocks
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262027],"optimizer_steps":1240}。
- **结果**：logloss=0.188851900；AUC=0.869600800；UAUC=0.789331875；gAUC=0.801643611；NDCG(10)=0.410399396。
- **开销**：参数 10222609；训练含验证 532.5960987135768 秒；峰值CUDA 22244532736 bytes；更新 1240。
- **原记录差值**：+0.000053780；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":16.32686424255371,"mean_ms":16.38263292312622,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计提高0.000053780，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 5](#phase-5)。
- **后续假设**：保持少量共享轮次，移动回读/attention位置并进一步压缩；见K06。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/compact_seq2-s20262027-20260906T150445270379Z/manifest.json`（SHA256前缀 `a9bba015f861`）；`qformer-loop/runs/compact_seq2-s20262027-20260906T150445270379Z/result.json`（SHA256前缀 `9a7973652e10`）

<a id="k082"></a>
### K082 · `compact_seq2-s20262028-20260906T150445279440Z`

- **方法**：{"model":"QFormerCross11Compact","loop_placement":"field_feedback","placement_steps":2,"num_ns_layers":2,"num_unified_layers":2,"extra_self_attention":false,"feedback_source":"context","query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Compact","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":2,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Reuse the first history block with two independent history blocks
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262028],"optimizer_steps":1240}。
- **结果**：logloss=0.190133412；AUC=0.868166373；UAUC=0.788391659；gAUC=0.801609803；NDCG(10)=0.406562710。
- **开销**：参数 10222609；训练含验证 534.5550039634109 秒；峰值CUDA 22244532736 bytes；更新 1240。
- **原记录差值**：-0.001563525；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":16.32686424255371,"mean_ms":16.38263292312622,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.001563525，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 5](#phase-5)。
- **后续假设**：保持少量共享轮次，移动回读/attention位置并进一步压缩；见K06。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/compact_seq2-s20262028-20260906T150445279440Z/manifest.json`（SHA256前缀 `825c4a17589f`）；`qformer-loop/runs/compact_seq2-s20262028-20260906T150445279440Z/result.json`（SHA256前缀 `9c02a12efbcc`）

<a id="k083"></a>
### K083 · `compact_seq2-s20262029-20260906T150444934869Z`

- **方法**：{"model":"QFormerCross11Compact","loop_placement":"field_feedback","placement_steps":2,"num_ns_layers":2,"num_unified_layers":2,"extra_self_attention":false,"feedback_source":"context","query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Compact","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":2,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Reuse the first history block with two independent history blocks
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262029],"optimizer_steps":1240}。
- **结果**：logloss=0.189838008；AUC=0.869263314；UAUC=0.788948738；gAUC=0.801610407；NDCG(10)=0.408351416。
- **开销**：参数 10222609；训练含验证 538.4941755607724 秒；峰值CUDA 22244532736 bytes；更新 1240。
- **原记录差值**：+0.000448153；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":16.32686424255371,"mean_ms":16.38263292312622,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计提高0.000448153，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 5](#phase-5)。
- **后续假设**：保持少量共享轮次，移动回读/attention位置并进一步压缩；见K06。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/compact_seq2-s20262029-20260906T150444934869Z/manifest.json`（SHA256前缀 `f8c16ac4b1ec`）；`qformer-loop/runs/compact_seq2-s20262029-20260906T150444934869Z/result.json`（SHA256前缀 `f2c93d0d480b`）

<a id="k084"></a>
### K084 · `diagnostics-round5-compact_seq1`

- **方法**：same checkpoint depth and Query interventions; extra attention off where present; shared mixer state hooks
- **原假设**：Check recurrence and extra attention reliance after compact/attention retraining
- **状态／预算**：原记录 `provisional`；{}。
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
      "AUC": 0.860906828744225,
      "logloss": 0.2030494660139084
    },
    "all_queries": {
      "AUC": 0.860906828744225,
      "logloss": 0.2030494660139084
    },
    "q3_half": {
      "AUC": 0.8604740124977595,
      "logloss": 0.20291730761528015
    },
    "q0_zero": {
      "AUC": 0.8606949944010427,
      "logloss": 0.2100018560886383
    },
    "q0_shuffle": {
      "AUC": 0.8580165017844481,
      "logloss": 0.20688511431217194
    },
    "q1_zero": {
      "AUC": 0.8605048683521844,
      "logloss": 0.2066923975944519
    },
    "q1_shuffle": {
      "AUC": 0.8528034190136914,
      "logloss": 0.21196772158145905
    },
    "q2_zero": {
      "AUC": 0.8605661766475088,
      "logloss": 0.2042122185230255
    },
    "q2_shuffle": {
      "AUC": 0.8581403771942655,
      "logloss": 0.20631448924541473
    },
    "q3_zero": {
      "AUC": 0.8594991267512286,
      "logloss": 0.20371031761169434
    },
    "q3_shuffle": {
      "AUC": 0.857301914919099,
      "logloss": 0.20623667538166046
    },
    "q4_zero": {
      "AUC": 0.8605144657922903,
      "logloss": 0.2042846381664276
    },
    "q4_shuffle": {
      "AUC": 0.8579697924121017,
      "logloss": 0.20722706615924835
    },
    "q5_zero": {
      "AUC": 0.8603631181325924,
      "logloss": 0.20464874804019928
    },
    "q5_shuffle": {
      "AUC": 0.8533956300692394,
      "logloss": 0.21140408515930176
    },
    "q6_zero": {
      "AUC": 0.8604956363075755,
      "logloss": 0.2043534815311432
    },
    "q6_shuffle": {
      "AUC": 0.8567381603579494,
      "logloss": 0.20774973928928375
    },
    "q7_zero": {
      "AUC": 0.8603212687315672,
      "logloss": 0.20619787275791168
    },
    "q7_shuffle": {
      "AUC": 0.858232940533315,
      "logloss": 0.20685479044914246
    },
    "inference_placement_steps1": {
      "AUC": 0.8367469186134376,
      "logloss": 0.21420589089393616
    },
    "inference_placement_steps3": {
      "AUC": 0.8396931616448173,
      "logloss": 0.21720102429389954
    }
  },
  "checkpoint_unchanged": true,
  "note": "subset diagnostics; within-block shuffle; blockwise loss means in toolkit differ from pooled full-validation metrics"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 5](#phase-5)。
- **后续假设**：保持少量共享轮次，移动回读/attention位置并进一步压缩；见K06。
- **证据**：`qformer-loop/reports/round5/compact_seq1/summary.json`（SHA256前缀 `5a1ab51961b3`）；`qformer-loop/reports/round5/compact_seq1/toolkit-perturbations/perturbation_report.json`（SHA256前缀 `01cca7f9324d`）；`qformer-loop/reports/round5/compact_seq1/toolkit-hooks/hook_report.json`（SHA256前缀 `d556c6279c13`）

<a id="k085"></a>
### K085 · `diagnostics-round5-attn_full`

- **方法**：same checkpoint depth and Query interventions; extra attention off where present; shared mixer state hooks
- **原假设**：Check recurrence and extra attention reliance after compact/attention retraining
- **状态／预算**：原记录 `provisional`；{}。
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
      "AUC": 0.8604958369582819,
      "logloss": 0.2032255232334137
    },
    "all_queries": {
      "AUC": 0.8604958369582819,
      "logloss": 0.2032255232334137
    },
    "q3_half": {
      "AUC": 0.8603251402341452,
      "logloss": 0.20369887351989746
    },
    "q0_zero": {
      "AUC": 0.8601115380526333,
      "logloss": 0.2047366499900818
    },
    "q0_shuffle": {
      "AUC": 0.8562714404784338,
      "logloss": 0.20823292434215546
    },
    "q1_zero": {
      "AUC": 0.8600773894145103,
      "logloss": 0.20497484505176544
    },
    "q1_shuffle": {
      "AUC": 0.8563678077324548,
      "logloss": 0.20795077085494995
    },
    "q2_zero": {
      "AUC": 0.8604072950818121,
      "logloss": 0.20382380485534668
    },
    "q2_shuffle": {
      "AUC": 0.8596372525854288,
      "logloss": 0.20489521324634552
    },
    "q3_zero": {
      "AUC": 0.8598764240032788,
      "logloss": 0.20942650735378265
    },
    "q3_shuffle": {
      "AUC": 0.8584039773075963,
      "logloss": 0.20595328509807587
    },
    "q4_zero": {
      "AUC": 0.8603637369816133,
      "logloss": 0.20608489215373993
    },
    "q4_shuffle": {
      "AUC": 0.8548138884015072,
      "logloss": 0.20991180837154388
    },
    "q5_zero": {
      "AUC": 0.8601246901784123,
      "logloss": 0.21933957934379578
    },
    "q5_shuffle": {
      "AUC": 0.8573280882196694,
      "logloss": 0.20738330483436584
    },
    "q6_zero": {
      "AUC": 0.860583198163753,
      "logloss": 0.20320594310760498
    },
    "q6_shuffle": {
      "AUC": 0.8560092766017384,
      "logloss": 0.20854943990707397
    },
    "q7_zero": {
      "AUC": 0.8598689175547453,
      "logloss": 0.2050502598285675
    },
    "q7_shuffle": {
      "AUC": 0.8576520377291593,
      "logloss": 0.20644867420196533
    },
    "inference_placement_steps1": {
      "AUC": 0.8432580171404872,
      "logloss": 0.21197260916233063
    },
    "inference_placement_steps3": {
      "AUC": 0.8461112279437986,
      "logloss": 0.21655941009521484
    },
    "extra_attention_off": {
      "AUC": 0.8572056173647971,
      "logloss": 0.2074534296989441
    }
  },
  "checkpoint_unchanged": true,
  "note": "subset diagnostics; within-block shuffle; blockwise loss means in toolkit differ from pooled full-validation metrics"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 5](#phase-5)。
- **后续假设**：保持少量共享轮次，移动回读/attention位置并进一步压缩；见K06。
- **证据**：`qformer-loop/reports/round5/attn_full/summary.json`（SHA256前缀 `38d622ae0ab7`）；`qformer-loop/reports/round5/attn_full/toolkit-perturbations/perturbation_report.json`（SHA256前缀 `084d28045662`）；`qformer-loop/reports/round5/attn_full/toolkit-hooks/hook_report.json`（SHA256前缀 `9fdcaa90ade3`）

<a id="phase-6"></a>
## 阶段 6：调度与小模型

**方法**：12个单seed结构：无循环、单字段块、末尾回读、history-only、三轮、独立权重及组合。

**归因与边界**：r6_field_after偏AUC，r6_small_attention_after偏压缩；收益不能直接相加；无循环和独立权重对照支持当前两轮的局部价值。

**历史后续假设**：在小模型中只保留首次或末次额外attention，并检查memory接口；见K07。

<a id="k086"></a>
### K086 · `r6_attention_after-s20262027-20260906T154409005638Z`

- **方法**：{"model":"QFormerCross11Schedule","loop_placement":"field_feedback","placement_steps":2,"attention_position":"after_history","num_ns_layers":2,"num_unified_layers":1,"extra_self_attention":true,"feedback_source":"context","query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Schedule","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Move the extra shared self-attention after each history read
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262027],"optimizer_steps":1240}。
- **结果**：logloss=0.189557436；AUC=0.869786821；UAUC=0.791967860；gAUC=0.803652713；NDCG(10)=0.412304558。
- **开销**：参数 9301009；训练含验证 493.8430597037077 秒；峰值CUDA 18942644224 bytes；更新 1240。
- **原记录差值**：+0.000239801；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":14.704192161560059,"mean_ms":14.746914339065551,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计提高0.000239801，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 6](#phase-6)。
- **后续假设**：在小模型中只保留首次或末次额外attention，并检查memory接口；见K07。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/r6_attention_after-s20262027-20260906T154409005638Z/manifest.json`（SHA256前缀 `76b8d66857b4`）；`qformer-loop/runs/r6_attention_after-s20262027-20260906T154409005638Z/result.json`（SHA256前缀 `f197dfc59af9`）

<a id="k087"></a>
### K087 · `r6_field_after-s20262027-20260906T154408812865Z`

- **方法**：{"model":"QFormerCross11Schedule","loop_placement":"field_feedback","placement_steps":2,"compact_schedule":"field_after","num_ns_layers":2,"num_unified_layers":1,"extra_self_attention":false,"feedback_source":"context","query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Schedule","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Read context again after both recurrent history reads
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262027],"optimizer_steps":1240}。
- **结果**：logloss=0.189043428；AUC=0.870420419；UAUC=0.790519470；gAUC=0.803063619；NDCG(10)=0.409432475。
- **开销**：参数 9037073；训练含验证 468.52347303181887 秒；峰值CUDA 17442170880 bytes；更新 1240。
- **原记录差值**：+0.000873399；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":13.66052770614624,"mean_ms":13.713779282569885,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计提高0.000873399，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 6](#phase-6)。
- **后续假设**：在小模型中只保留首次或末次额外attention，并检查memory接口；见K07。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/r6_field_after-s20262027-20260906T154408812865Z/manifest.json`（SHA256前缀 `acff67d9ebed`）；`qformer-loop/runs/r6_field_after-s20262027-20260906T154408812865Z/result.json`（SHA256前缀 `63208e3e5665`）

<a id="k088"></a>
### K088 · `r6_field_before-s20262027-20260906T154409034007Z`

- **方法**：{"model":"QFormerCross11Schedule","loop_placement":"field_feedback","placement_steps":2,"compact_schedule":"field_before","num_ns_layers":2,"num_unified_layers":1,"extra_self_attention":false,"feedback_source":"context","query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Schedule","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Read context again before both recurrent history reads
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262027],"optimizer_steps":1240}。
- **结果**：logloss=0.189606574；AUC=0.868442402；UAUC=0.790629622；gAUC=0.802437554；NDCG(10)=0.408094066。
- **开销**：参数 9037073；训练含验证 466.0376594811678 秒；峰值CUDA 18463417344 bytes；更新 1240。
- **原记录差值**：-0.001104618；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":13.681903839111328,"mean_ms":13.711269640922547,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.001104618，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 6](#phase-6)。
- **后续假设**：在小模型中只保留首次或末次额外attention，并检查memory接口；见K07。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/r6_field_before-s20262027-20260906T154409034007Z/manifest.json`（SHA256前缀 `155c6ef9f4a0`）；`qformer-loop/runs/r6_field_before-s20262027-20260906T154409034007Z/result.json`（SHA256前缀 `f855cd90d446`）

<a id="k089"></a>
### K089 · `r6_history_only-s20262027-20260906T154409087527Z`

- **方法**：{"model":"QFormerCross11Schedule","loop_placement":"field_feedback","placement_steps":2,"compact_schedule":"history_only","num_ns_layers":2,"num_unified_layers":1,"extra_self_attention":false,"feedback_source":"context","query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Schedule","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Allocate the same five block calls to one field stage and three history reads
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262027],"optimizer_steps":1240}。
- **结果**：logloss=0.189984123；AUC=0.868178603；UAUC=0.789600704；gAUC=0.801792136；NDCG(10)=0.408226347。
- **开销**：参数 9037073；训练含验证 492.4893796220422 秒；峰值CUDA 20176790528 bytes；更新 1240。
- **原记录差值**：-0.001368417；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":14.730384349822998,"mean_ms":14.781688809394836,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.001368417，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 6](#phase-6)。
- **后续假设**：在小模型中只保留首次或末次额外attention，并检查memory接口；见K07。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/r6_history_only-s20262027-20260906T154409087527Z/manifest.json`（SHA256前缀 `cc0814b2bce2`）；`qformer-loop/runs/r6_history_only-s20262027-20260906T154409087527Z/result.json`（SHA256前缀 `2a875d505fe1`）

<a id="k090"></a>
### K090 · `r6_loop3-s20262027-20260906T154408977323Z`

- **方法**：{"model":"QFormerCross11Schedule","loop_placement":"field_feedback","placement_steps":3,"num_ns_layers":2,"num_unified_layers":1,"extra_self_attention":false,"feedback_source":"context","query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Schedule","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Train three compact alternating cycles instead of changing depth only at inference
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262027],"optimizer_steps":1240}。
- **结果**：logloss=0.189424294；AUC=0.868594590；UAUC=0.790704844；gAUC=0.802810873；NDCG(10)=0.408113892。
- **开销**：参数 9037073；训练含验证 570.6745585277677 秒；峰值CUDA 24282426368 bytes；更新 1240。
- **原记录差值**：-0.000952430；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":17.824015617370605,"mean_ms":17.855112838745118,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000952430，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 6](#phase-6)。
- **后续假设**：在小模型中只保留首次或末次额外attention，并检查memory接口；见K07。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/r6_loop3-s20262027-20260906T154408977323Z/manifest.json`（SHA256前缀 `1d0768a2eeae`）；`qformer-loop/runs/r6_loop3-s20262027-20260906T154408977323Z/result.json`（SHA256前缀 `a44ea520d817`）

<a id="k091"></a>
### K091 · `r6_shallow-s20262027-20260906T154408839146Z`

- **方法**：{"model":"QFormerCross11Schedule","loop_placement":"field_feedback","placement_steps":1,"num_ns_layers":2,"num_unified_layers":1,"extra_self_attention":false,"feedback_source":"context","query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Schedule","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Train the shallow no-loop control from scratch on the existing seed
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262027],"optimizer_steps":1240}。
- **结果**：logloss=0.188992004；AUC=0.869363698；UAUC=0.789577803；gAUC=0.801564424；NDCG(10)=0.405964227。
- **开销**：参数 9037073；训练含验证 360.23789162933826 秒；峰值CUDA 12642700288 bytes；更新 1240。
- **原记录差值**：-0.000183322；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":9.551408290863037,"mean_ms":9.597433614730836,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000183322，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 6](#phase-6)。
- **后续假设**：在小模型中只保留首次或末次额外attention，并检查memory接口；见K07。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/r6_shallow-s20262027-20260906T154408839146Z/manifest.json`（SHA256前缀 `8b6a7b96e4ae`）；`qformer-loop/runs/r6_shallow-s20262027-20260906T154408839146Z/result.json`（SHA256前缀 `df25a3357b4b`）

<a id="k092"></a>
### K092 · `r6_small-s20262027-20260906T154409093884Z`

- **方法**：{"model":"QFormerCross11Schedule","loop_placement":"field_feedback","placement_steps":2,"num_ns_layers":1,"num_unified_layers":1,"extra_self_attention":false,"feedback_source":"context","query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Schedule","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Remove the independent field prelude without extra attention
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262027],"optimizer_steps":1240}。
- **结果**：logloss=0.189212116；AUC=0.869914941；UAUC=0.790276759；gAUC=0.802925524；NDCG(10)=0.406754926。
- **开销**：参数 7851537；训练含验证 431.50614561885595 秒；峰值CUDA 16284259840 bytes；更新 1240。
- **原记录差值**：+0.000367921；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.096400260925293,"mean_ms":12.126486372947692,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计提高0.000367921，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 6](#phase-6)。
- **后续假设**：在小模型中只保留首次或末次额外attention，并检查memory接口；见K07。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/r6_small-s20262027-20260906T154409093884Z/manifest.json`（SHA256前缀 `19568b1005d8`）；`qformer-loop/runs/r6_small-s20262027-20260906T154409093884Z/result.json`（SHA256前缀 `0250f0563c00`）

<a id="k093"></a>
### K093 · `r6_untied-s20262027-20260906T154409087381Z`

- **方法**：{"model":"QFormerCross11Schedule","loop_placement":"field_feedback","placement_steps":2,"compact_schedule":"untied","num_ns_layers":2,"num_unified_layers":2,"extra_self_attention":false,"feedback_source":"context","query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Schedule","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":2,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Use independent history weights at the same five-block-call depth
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262027],"optimizer_steps":1240}。
- **结果**：logloss=0.189260407；AUC=0.869809011；UAUC=0.790037898；gAUC=0.802444297；NDCG(10)=0.409393466。
- **开销**：参数 10222609；训练含验证 462.38438352942467 秒；峰值CUDA 18478571008 bytes；更新 1240。
- **原记录差值**：+0.000261991；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":13.624959945678711,"mean_ms":13.661237597465515,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计提高0.000261991，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 6](#phase-6)。
- **后续假设**：在小模型中只保留首次或末次额外attention，并检查memory接口；见K07。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/r6_untied-s20262027-20260906T154409087381Z/manifest.json`（SHA256前缀 `072401ea85dd`）；`qformer-loop/runs/r6_untied-s20262027-20260906T154409087381Z/result.json`（SHA256前缀 `449e3e172663`）

<a id="k094"></a>
### K094 · `r6_after_all-s20262027-20260906T160133940781Z`

- **方法**：{"model":"QFormerCross11Schedule","loop_placement":"field_feedback","placement_steps":2,"compact_schedule":"field_after","num_ns_layers":2,"num_unified_layers":1,"extra_self_attention":false,"feedback_source":"all","query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Schedule","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Read all fields in the terminal feedback position
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262027],"optimizer_steps":1240}。
- **结果**：logloss=0.189803776；AUC=0.868398496；UAUC=0.791361111；gAUC=0.803123556；NDCG(10)=0.412986728。
- **开销**：参数 9037073；训练含验证 471.97756879776716 秒；峰值CUDA 17618987008 bytes；更新 1240。
- **原记录差值**：-0.001148524；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":13.715375900268555,"mean_ms":13.78675606250763,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.001148524，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 6](#phase-6)。
- **后续假设**：在小模型中只保留首次或末次额外attention，并检查memory接口；见K07。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/r6_after_all-s20262027-20260906T160133940781Z/manifest.json`（SHA256前缀 `e85a88efec0f`）；`qformer-loop/runs/r6_after_all-s20262027-20260906T160133940781Z/result.json`（SHA256前缀 `770619eeda6e`）

<a id="k095"></a>
### K095 · `r6_after_target-s20262027-20260906T160133985128Z`

- **方法**：{"model":"QFormerCross11Schedule","loop_placement":"field_feedback","placement_steps":2,"compact_schedule":"field_after","num_ns_layers":2,"num_unified_layers":1,"extra_self_attention":false,"feedback_source":"target","query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Schedule","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Read target-only fields in the terminal feedback position
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262027],"optimizer_steps":1240}。
- **结果**：logloss=0.189810061；AUC=0.868483810；UAUC=0.789959386；gAUC=0.802018016；NDCG(10)=0.414381346。
- **开销**：参数 9037073；训练含验证 459.2070964425802 秒；峰值CUDA 16855623680 bytes；更新 1240。
- **原记录差值**：-0.001063210；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":13.218400001525879,"mean_ms":13.221523237228393,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.001063210，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 6](#phase-6)。
- **后续假设**：在小模型中只保留首次或末次额外attention，并检查memory接口；见K07。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/r6_after_target-s20262027-20260906T160133985128Z/manifest.json`（SHA256前缀 `c683efc1ce39`）；`qformer-loop/runs/r6_after_target-s20262027-20260906T160133985128Z/result.json`（SHA256前缀 `94a34d728524`）

<a id="k096"></a>
### K096 · `r6_small_after-s20262027-20260906T160133981082Z`

- **方法**：{"model":"QFormerCross11Schedule","loop_placement":"field_feedback","placement_steps":2,"compact_schedule":"field_after","num_ns_layers":1,"num_unified_layers":1,"extra_self_attention":false,"feedback_source":"context","query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Schedule","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Combine terminal field feedback with removal of the independent field prelude
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262027],"optimizer_steps":1240}。
- **结果**：logloss=0.189580222；AUC=0.868993971；UAUC=0.788462076；gAUC=0.800350080；NDCG(10)=0.406757431。
- **开销**：参数 7851537；训练含验证 419.26593551039696 秒；峰值CUDA 15263078912 bytes；更新 1240。
- **原记录差值**：-0.000553049；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.065743923187256,"mean_ms":12.063038444519043,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000553049，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 6](#phase-6)。
- **后续假设**：在小模型中只保留首次或末次额外attention，并检查memory接口；见K07。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/r6_small_after-s20262027-20260906T160133981082Z/manifest.json`（SHA256前缀 `809e93354a0a`）；`qformer-loop/runs/r6_small_after-s20262027-20260906T160133981082Z/result.json`（SHA256前缀 `d6feb178ef6f`）

<a id="k097"></a>
### K097 · `r6_small_attention_after-s20262027-20260906T160133918683Z`

- **方法**：{"model":"QFormerCross11Schedule","loop_placement":"field_feedback","placement_steps":2,"attention_position":"after_history","num_ns_layers":1,"num_unified_layers":1,"extra_self_attention":true,"feedback_source":"context","query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Schedule","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Add shared post-history attention to the one-field one-history compact model
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262027],"optimizer_steps":1240}。
- **结果**：logloss=0.188566548；AUC=0.870152667；UAUC=0.788919847；gAUC=0.801209437；NDCG(10)=0.410035849。
- **开销**：参数 8115473；训练含验证 455.5708587914705 秒；峰值CUDA 16763355648 bytes；更新 1240。
- **原记录差值**：+0.000605647；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":13.008975982666016,"mean_ms":13.008667159080506,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计提高0.000605647，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 6](#phase-6)。
- **后续假设**：在小模型中只保留首次或末次额外attention，并检查memory接口；见K07。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/r6_small_attention_after-s20262027-20260906T160133918683Z/manifest.json`（SHA256前缀 `7460608991c9`）；`qformer-loop/runs/r6_small_attention_after-s20262027-20260906T160133918683Z/result.json`（SHA256前缀 `5069f997e25a`）

<a id="k098"></a>
### K098 · `diagnostics-round6-r6_field_after`

- **方法**：frozen checkpoint Query/depth interventions; attention off where present
- **原假设**：Check trained terminal feedback and compact post-history attention reliance
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
      "AUC": 0.8612251494733709,
      "logloss": 0.2034590244293213
    },
    "all_queries": {
      "AUC": 0.8612251494733709,
      "logloss": 0.2034590244293213
    },
    "q3_half": {
      "AUC": 0.8612742814389833,
      "logloss": 0.20242054760456085
    },
    "q0_zero": {
      "AUC": 0.8608289678216758,
      "logloss": 0.20595785975456238
    },
    "q0_shuffle": {
      "AUC": 0.8543813826356085,
      "logloss": 0.2109258621931076
    },
    "q1_zero": {
      "AUC": 0.8601885646906664,
      "logloss": 0.204035684466362
    },
    "q1_shuffle": {
      "AUC": 0.8585862969879178,
      "logloss": 0.2056545466184616
    },
    "q2_zero": {
      "AUC": 0.8611121810135329,
      "logloss": 0.2027694433927536
    },
    "q2_shuffle": {
      "AUC": 0.859612093098954,
      "logloss": 0.20585928857326508
    },
    "q3_zero": {
      "AUC": 0.8610118957904547,
      "logloss": 0.20644529163837433
    },
    "q3_shuffle": {
      "AUC": 0.8605840345603819,
      "logloss": 0.20359733700752258
    },
    "q4_zero": {
      "AUC": 0.8608115196586663,
      "logloss": 0.20544253289699554
    },
    "q4_shuffle": {
      "AUC": 0.8558479323126342,
      "logloss": 0.20979799330234528
    },
    "q5_zero": {
      "AUC": 0.8601057952182039,
      "logloss": 0.2052803337574005
    },
    "q5_shuffle": {
      "AUC": 0.8570492449889872,
      "logloss": 0.2080642133951187
    },
    "q6_zero": {
      "AUC": 0.8608595828952531,
      "logloss": 0.20448751747608185
    },
    "q6_shuffle": {
      "AUC": 0.8565614272157178,
      "logloss": 0.2092379927635193
    },
    "q7_zero": {
      "AUC": 0.8610085861098548,
      "logloss": 0.20351466536521912
    },
    "q7_shuffle": {
      "AUC": 0.8554429537115464,
      "logloss": 0.20978933572769165
    },
    "inference_placement_steps1": {
      "AUC": 0.8328175082416327,
      "logloss": 0.2551407217979431
    },
    "inference_placement_steps3": {
      "AUC": 0.8563795743123028,
      "logloss": 0.21089953184127808
    }
  },
  "checkpoint_unchanged": true,
  "note": "subset diagnostics; within-block shuffle; blockwise loss means in toolkit differ from pooled full-validation metrics"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 6](#phase-6)。
- **后续假设**：在小模型中只保留首次或末次额外attention，并检查memory接口；见K07。
- **证据**：`qformer-loop/reports/round6/r6_field_after/summary.json`（SHA256前缀 `b43c269787a0`）；`qformer-loop/reports/round6/r6_field_after/toolkit-perturbations/perturbation_report.json`（SHA256前缀 `dbde33b14f5a`）；`qformer-loop/reports/round6/r6_field_after/toolkit-hooks/hook_report.json`（SHA256前缀 `7d334f1aaae4`）

<a id="k099"></a>
### K099 · `diagnostics-round6-r6_small_attention_after`

- **方法**：frozen checkpoint Query/depth interventions; attention off where present
- **原假设**：Check trained terminal feedback and compact post-history attention reliance
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
      "AUC": 0.8600300992111716,
      "logloss": 0.20309753715991974
    },
    "all_queries": {
      "AUC": 0.8600300992111716,
      "logloss": 0.20309753715991974
    },
    "q3_half": {
      "AUC": 0.8598591321366091,
      "logloss": 0.20323516428470612
    },
    "q0_zero": {
      "AUC": 0.8585110762062428,
      "logloss": 0.20762629806995392
    },
    "q0_shuffle": {
      "AUC": 0.8561724922226942,
      "logloss": 0.20715183019638062
    },
    "q1_zero": {
      "AUC": 0.8587776966407337,
      "logloss": 0.204011470079422
    },
    "q1_shuffle": {
      "AUC": 0.8570279781262174,
      "logloss": 0.2060786932706833
    },
    "q2_zero": {
      "AUC": 0.8593634889857885,
      "logloss": 0.20711572468280792
    },
    "q2_shuffle": {
      "AUC": 0.8564459516781054,
      "logloss": 0.20707741379737854
    },
    "q3_zero": {
      "AUC": 0.8592942095771368,
      "logloss": 0.20444054901599884
    },
    "q3_shuffle": {
      "AUC": 0.8559966081497675,
      "logloss": 0.20681607723236084
    },
    "q4_zero": {
      "AUC": 0.8597707993592966,
      "logloss": 0.2060043215751648
    },
    "q4_shuffle": {
      "AUC": 0.8570725648253009,
      "logloss": 0.2062799334526062
    },
    "q5_zero": {
      "AUC": 0.8600780441694471,
      "logloss": 0.20321987569332123
    },
    "q5_shuffle": {
      "AUC": 0.8583460040382244,
      "logloss": 0.2054452747106552
    },
    "q6_zero": {
      "AUC": 0.8591720470907186,
      "logloss": 0.2033652514219284
    },
    "q6_shuffle": {
      "AUC": 0.8555462043408546,
      "logloss": 0.2076447606086731
    },
    "q7_zero": {
      "AUC": 0.8598978260412614,
      "logloss": 0.20425091683864594
    },
    "q7_shuffle": {
      "AUC": 0.8558679340198971,
      "logloss": 0.20784872770309448
    },
    "inference_placement_steps1": {
      "AUC": 0.8117712640916781,
      "logloss": 0.30256301164627075
    },
    "inference_placement_steps3": {
      "AUC": 0.816501678419926,
      "logloss": 0.229020893573761
    },
    "extra_attention_off": {
      "AUC": 0.852196119025019,
      "logloss": 0.21969152987003326
    }
  },
  "checkpoint_unchanged": true,
  "note": "subset diagnostics; within-block shuffle; blockwise loss means in toolkit differ from pooled full-validation metrics"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 6](#phase-6)。
- **后续假设**：在小模型中只保留首次或末次额外attention，并检查memory接口；见K07。
- **证据**：`qformer-loop/reports/round6/r6_small_attention_after/summary.json`（SHA256前缀 `2554c2a945a0`）；`qformer-loop/reports/round6/r6_small_attention_after/toolkit-perturbations/perturbation_report.json`（SHA256前缀 `61c9ec8e0fc5`）；`qformer-loop/reports/round6/r6_small_attention_after/toolkit-hooks/hook_report.json`（SHA256前缀 `eb5dd5d477e6`）

<a id="phase-7"></a>
## 阶段 7：首次attention与memory接口

**方法**：单字段/单历史共享两轮，仅首次额外attention；另测末次attention、norm位置和附加K/V。

**归因与边界**：r7_small_attention_first AUC=0.871234058；相对每轮额外attention的直接父模型+0.001081。参数8.115M；没有证明所有attention都应只做一次。

**历史后续假设**：检验一/二/三轮及norm/残差组合；见K08。

<a id="k100"></a>
### K100 · `r7_small_attention_first-s20262027-20260906T163655782605Z`

- **方法**：{"model":"QFormerCross11Memory","loop_placement":"field_feedback","placement_steps":2,"attention_position":"after_history","mixer_passes":"first","num_ns_layers":1,"num_unified_layers":1,"extra_self_attention":true,"feedback_source":"context","query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Memory","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Use the extra self-attention only after the first history read
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262027],"optimizer_steps":1240}。
- **结果**：logloss=0.188066858；AUC=0.871234058；UAUC=0.790837013；gAUC=0.803790662；NDCG(10)=0.409856157。
- **开销**：参数 8115473；训练含验证 441.5844851434231 秒；峰值CUDA 16897573376 bytes；更新 1240。
- **原记录差值**：+0.001687038；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r6_small_attention_after`；完整指标差 {"logloss":-0.0004996899819595391,"AUC":0.001081391238758611,"UAUC":0.001917165531842735,"gAUC":0.0025812256474732065,"NDCG(10)":-0.00017969208190776742}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.51796817779541,"mean_ms":12.527073621749878,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计提高0.001687038，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 7](#phase-7)。
- **后续假设**：检验一/二/三轮及norm/残差组合；见K08。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/r7_small_attention_first-s20262027-20260906T163655782605Z/manifest.json`（SHA256前缀 `938371c55169`）；`qformer-loop/runs/r7_small_attention_first-s20262027-20260906T163655782605Z/result.json`（SHA256前缀 `dcad15eaf6b6`）

<a id="k101"></a>
### K101 · `r7_small_attention_last-s20262027-20260906T163655854191Z`

- **方法**：{"model":"QFormerCross11Memory","loop_placement":"field_feedback","placement_steps":2,"attention_position":"after_history","mixer_passes":"last","num_ns_layers":1,"num_unified_layers":1,"extra_self_attention":true,"feedback_source":"context","query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Memory","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Use the extra self-attention only after the last history read
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262027],"optimizer_steps":1240}。
- **结果**：logloss=0.189319175；AUC=0.868642592；UAUC=0.790045987；gAUC=0.801798882；NDCG(10)=0.411488051。
- **开销**：参数 8115473；训练含验证 449.5333648249507 秒；峰值CUDA 16421647872 bytes；更新 1240。
- **原记录差值**：-0.000904429；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r6_small_attention_after`；完整指标差 {"logloss":0.0007526269319351186,"AUC":-0.0015100756874062427,"UAUC":0.0011261396657286094,"gAUC":0.00058944569745778,"NDCG(10)":0.001452202466949215}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.526463985443115,"mean_ms":12.530442428588866,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000904429，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 7](#phase-7)。
- **后续假设**：检验一/二/三轮及norm/残差组合；见K08。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/r7_small_attention_last-s20262027-20260906T163655854191Z/manifest.json`（SHA256前缀 `060294a9809f`）；`qformer-loop/runs/r7_small_attention_last-s20262027-20260906T163655854191Z/result.json`（SHA256前缀 `89fb11de3236`）

<a id="k102"></a>
### K102 · `r7_small_field_memory-s20262027-20260906T163655842646Z`

- **方法**：{"model":"QFormerCross11Memory","loop_placement":"field_feedback","placement_steps":2,"attention_position":"after_history","feedback_memory":"first_field","num_ns_layers":1,"num_unified_layers":1,"extra_self_attention":true,"feedback_source":"context","query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Memory","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Preserve first field encoding in compact recurrent field memory
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262027],"optimizer_steps":1240}。
- **结果**：logloss=0.189543628；AUC=0.868943803；UAUC=0.789600982；gAUC=0.801834502；NDCG(10)=0.408844058。
- **开销**：参数 8115473；训练含验证 463.2584771886468 秒；峰值CUDA 17100276224 bytes；更新 1240。
- **原记录差值**：-0.000603217；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r6_small_attention_after`；完整指标差 {"logloss":0.000977079312106971,"AUC":-0.0012088642033489805,"UAUC":0.000681134302969788,"gAUC":0.0006250655665290816,"NDCG(10)":-0.0011917907706699982}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":13.184912204742432,"mean_ms":13.269463205337525,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000603217，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 7](#phase-7)。
- **后续假设**：检验一/二/三轮及norm/残差组合；见K08。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/r7_small_field_memory-s20262027-20260906T163655842646Z/manifest.json`（SHA256前缀 `e35f96f47d41`）；`qformer-loop/runs/r7_small_field_memory-s20262027-20260906T163655842646Z/result.json`（SHA256前缀 `af6d606f0a4c`）

<a id="k103"></a>
### K103 · `r7_small_midnorm-s20262027-20260906T163655828274Z`

- **方法**：{"model":"QFormerCross11Memory","loop_placement":"field_feedback","placement_steps":2,"attention_position":"after_history","interface_norm":"per_history","num_ns_layers":1,"num_unified_layers":1,"extra_self_attention":true,"feedback_source":"context","query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Memory","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Normalize every history read before extra self-attention
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262027],"optimizer_steps":1240}。
- **结果**：logloss=0.188237207；AUC=0.870780429；UAUC=0.787894161；gAUC=0.800085154；NDCG(10)=0.406470847。
- **开销**：参数 8115473；训练含验证 459.9259078800678 秒；峰值CUDA 16965337600 bytes；更新 1240。
- **原记录差值**：+0.001233409；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r6_small_attention_after`；完整指标差 {"logloss":-0.0003293412072898627,"AUC":0.0006277622588470866,"UAUC":-0.0010256858677493152,"gAUC":-0.0011242827206656703,"NDCG(10)":-0.003565001584841798}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":13.020448207855225,"mean_ms":13.023963952064515,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计提高0.001233409，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 7](#phase-7)。
- **后续假设**：检验一/二/三轮及norm/残差组合；见K08。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/r7_small_midnorm-s20262027-20260906T163655828274Z/manifest.json`（SHA256前缀 `f58acdd92296`）；`qformer-loop/runs/r7_small_midnorm-s20262027-20260906T163655828274Z/result.json`（SHA256前缀 `ac18d6587bf2`）

<a id="k104"></a>
### K104 · `r7_small_query_memory-s20262027-20260906T163655785402Z`

- **方法**：{"model":"QFormerCross11Memory","loop_placement":"field_feedback","placement_steps":2,"attention_position":"after_history","feedback_memory":"initial_query","num_ns_layers":1,"num_unified_layers":1,"extra_self_attention":true,"feedback_source":"context","query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Memory","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Preserve initial query states in compact recurrent field memory
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262027],"optimizer_steps":1240}。
- **结果**：logloss=0.189726735；AUC=0.868739261；UAUC=0.787747107；gAUC=0.800446591；NDCG(10)=0.406020621。
- **开销**：参数 8115473；训练含验证 470.54008904099464 秒；峰值CUDA 17100276224 bytes；更新 1240。
- **原记录差值**：-0.000807759；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r6_small_attention_after`；完整指标差 {"logloss":0.0011601861244721623,"AUC":-0.0014134057898524954,"UAUC":-0.0011727402955763466,"gAUC":-0.0007628458123740689,"NDCG(10)":-0.004015228178416208}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":13.186944007873535,"mean_ms":13.1907888174057,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000807759，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 7](#phase-7)。
- **后续假设**：检验一/二/三轮及norm/残差组合；见K08。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/r7_small_query_memory-s20262027-20260906T163655785402Z/manifest.json`（SHA256前缀 `4853afedee8b`）；`qformer-loop/runs/r7_small_query_memory-s20262027-20260906T163655785402Z/result.json`（SHA256前缀 `dd413955ec94`）

<a id="k105"></a>
### K105 · `r7_terminal_field_memory-s20262027-20260906T163655831635Z`

- **方法**：{"model":"QFormerCross11Memory","loop_placement":"field_feedback","placement_steps":2,"compact_schedule":"field_after","feedback_memory":"first_field","num_ns_layers":2,"num_unified_layers":1,"extra_self_attention":false,"feedback_source":"context","query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Memory","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Append first full-field states to terminal context K/V
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262027],"optimizer_steps":1240}。
- **结果**：logloss=0.189955455；AUC=0.868358350；UAUC=0.790084471；gAUC=0.801672343；NDCG(10)=0.408826828。
- **开销**：参数 9037073；训练含验证 472.2939475700259 秒；峰值CUDA 18087306240 bytes；更新 1240。
- **原记录差值**：-0.001188670；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r6_field_after`；完整指标差 {"logloss":0.0009120273691900216,"AUC":-0.0020620686094707796,"UAUC":-0.0004349987621806717,"gAUC":-0.0013912768514438012,"NDCG(10)":-0.0006056477879216593}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":13.80683183670044,"mean_ms":13.844675970077514,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.001188670，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 7](#phase-7)。
- **后续假设**：检验一/二/三轮及norm/残差组合；见K08。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/r7_terminal_field_memory-s20262027-20260906T163655831635Z/manifest.json`（SHA256前缀 `724acd6ecb1e`）；`qformer-loop/runs/r7_terminal_field_memory-s20262027-20260906T163655831635Z/result.json`（SHA256前缀 `e8195fdcc1d1`）

<a id="k106"></a>
### K106 · `r7_terminal_prenorm-s20262027-20260906T163655868809Z`

- **方法**：{"model":"QFormerCross11Memory","loop_placement":"field_feedback","placement_steps":2,"compact_schedule":"field_after","interface_norm":"before_feedback","num_ns_layers":2,"num_unified_layers":1,"extra_self_attention":false,"feedback_source":"context","query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Memory","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Move sequence normalization before terminal field feedback
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262027],"optimizer_steps":1240}。
- **结果**：logloss=0.189297295；AUC=0.869821016；UAUC=0.789810278；gAUC=0.801312503；NDCG(10)=0.409524880。
- **开销**：参数 9037073；训练含验证 469.6892803013325 秒；峰值CUDA 17631896576 bytes；更新 1240。
- **原记录差值**：+0.000273996；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r6_field_after`；完整指标差 {"logloss":0.0002538670738833626,"AUC":-0.0005994025517856993,"UAUC":-0.0007091920935178297,"gAUC":-0.001751116566812816,"NDCG(10)":9.24042423514071e-05}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":13.62827205657959,"mean_ms":13.664764070510865,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计提高0.000273996，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 7](#phase-7)。
- **后续假设**：检验一/二/三轮及norm/残差组合；见K08。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/r7_terminal_prenorm-s20262027-20260906T163655868809Z/manifest.json`（SHA256前缀 `f3da238a0ee4`）；`qformer-loop/runs/r7_terminal_prenorm-s20262027-20260906T163655868809Z/result.json`（SHA256前缀 `74edfff9563b`）

<a id="k107"></a>
### K107 · `r7_terminal_query_memory-s20262027-20260906T163655724136Z`

- **方法**：{"model":"QFormerCross11Memory","loop_placement":"field_feedback","placement_steps":2,"compact_schedule":"field_after","feedback_memory":"initial_query","num_ns_layers":2,"num_unified_layers":1,"extra_self_attention":false,"feedback_source":"context","query_fusion_fixed_scales":[1,1,1,1,1,1,1,1]}；设置 {"model":"QFormerCross11Memory","token_dim":256,"num_queries":8,"num_ns_layers":2,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Append frozen initial query states to terminal context K/V
- **状态／预算**：原记录 `provisional`；{"epochs":1,"seeds":[20262027],"optimizer_steps":1240}。
- **结果**：logloss=0.188830868；AUC=0.869804939；UAUC=0.790664715；gAUC=0.802536088；NDCG(10)=0.410483216。
- **开销**：参数 9037073；训练含验证 470.8389959037304 秒；峰值CUDA 18087306240 bytes；更新 1240。
- **原记录差值**：+0.000257919；对照 `same-seed Cross8 original`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r6_field_after`；完整指标差 {"logloss":-0.00021256024246488203,"AUC":-0.0006154797841543269,"UAUC":0.0001452451497542251,"gAUC":-0.0005275311097916457,"NDCG(10)":0.001050740576687803}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":13.814303874969482,"mean_ms":13.824019122123719,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计提高0.000257919，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 7](#phase-7)。
- **后续假设**：检验一/二/三轮及norm/残差组合；见K08。
- **原判读**：preserved optimization/mechanism candidate; not a promotion or stable-improvement claim
- **证据**：`qformer-loop/runs/r7_terminal_query_memory-s20262027-20260906T163655724136Z/manifest.json`（SHA256前缀 `04e5cc5f8f1a`）；`qformer-loop/runs/r7_terminal_query_memory-s20262027-20260906T163655724136Z/result.json`（SHA256前缀 `ea84ec2977d1`）

<a id="k108"></a>
### K108 · `diagnostics-round7-r7_small_attention_first`

- **方法**：Frozen checkpoint Query Zero/Shuffle, depth and extra-attention-off probes
- **原假设**：Check first-pass attention and moved norm at the actual final Query output
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
      "AUC": 0.8621990742086234,
      "logloss": 0.20199689269065857
    },
    "all_queries": {
      "AUC": 0.8621990742086234,
      "logloss": 0.20199689269065857
    },
    "q3_half": {
      "AUC": 0.8623839622179873,
      "logloss": 0.20193025469779968
    },
    "q0_zero": {
      "AUC": 0.861934472953873,
      "logloss": 0.20359773933887482
    },
    "q0_shuffle": {
      "AUC": 0.8558868669981343,
      "logloss": 0.20845234394073486
    },
    "q1_zero": {
      "AUC": 0.861926092090682,
      "logloss": 0.2026444375514984
    },
    "q1_shuffle": {
      "AUC": 0.8610456262302635,
      "logloss": 0.20339269936084747
    },
    "q2_zero": {
      "AUC": 0.8602070773584763,
      "logloss": 0.2055021971464157
    },
    "q2_shuffle": {
      "AUC": 0.852982456470878,
      "logloss": 0.210189089179039
    },
    "q3_zero": {
      "AUC": 0.8623216084268769,
      "logloss": 0.20301643013954163
    },
    "q3_shuffle": {
      "AUC": 0.859486865937009,
      "logloss": 0.2060992419719696
    },
    "q4_zero": {
      "AUC": 0.8619555792960778,
      "logloss": 0.20212696492671967
    },
    "q4_shuffle": {
      "AUC": 0.8604238709422768,
      "logloss": 0.2043696641921997
    },
    "q5_zero": {
      "AUC": 0.8617422157833011,
      "logloss": 0.20446760952472687
    },
    "q5_shuffle": {
      "AUC": 0.8600323464990837,
      "logloss": 0.20458997786045074
    },
    "q6_zero": {
      "AUC": 0.8615060266685826,
      "logloss": 0.2061876803636551
    },
    "q6_shuffle": {
      "AUC": 0.8585145126136046,
      "logloss": 0.20536844432353973
    },
    "q7_zero": {
      "AUC": 0.8616109310821343,
      "logloss": 0.20395638048648834
    },
    "q7_shuffle": {
      "AUC": 0.8578936169554866,
      "logloss": 0.2058779001235962
    },
    "inference_placement_steps1": {
      "AUC": 0.8104100708203215,
      "logloss": 0.2489156574010849
    },
    "inference_placement_steps3": {
      "AUC": 0.8284840084681864,
      "logloss": 0.22069233655929565
    },
    "extra_attention_off": {
      "AUC": 0.8584370677772573,
      "logloss": 0.20498324930667877
    }
  },
  "checkpoint_unchanged": true,
  "note": "subset diagnostics; within-block shuffle; blockwise loss means in toolkit differ from pooled full-validation metrics"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 7](#phase-7)。
- **后续假设**：检验一/二/三轮及norm/残差组合；见K08。
- **证据**：`qformer-loop/reports/round7/r7_small_attention_first/summary.json`（SHA256前缀 `7ce485f39d30`）；`qformer-loop/reports/round7/r7_small_attention_first/toolkit-perturbations/perturbation_report.json`（SHA256前缀 `cdf4266fbdd9`）；`qformer-loop/reports/round7/r7_small_attention_first/toolkit-hooks/hook_report.json`（SHA256前缀 `9c0dc48f3cce`）

<a id="k109"></a>
### K109 · `diagnostics-round7-r7_small_midnorm`

- **方法**：Frozen checkpoint Query Zero/Shuffle, depth and extra-attention-off probes
- **原假设**：Check first-pass attention and moved norm at the actual final Query output
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
      "AUC": 0.8608857224020202,
      "logloss": 0.2028093785047531
    },
    "all_queries": {
      "AUC": 0.8608857224020202,
      "logloss": 0.2028093785047531
    },
    "q3_half": {
      "AUC": 0.8604489565048069,
      "logloss": 0.20310597121715546
    },
    "q0_zero": {
      "AUC": 0.8599141273270752,
      "logloss": 0.20314106345176697
    },
    "q0_shuffle": {
      "AUC": 0.857258407509605,
      "logloss": 0.20684710144996643
    },
    "q1_zero": {
      "AUC": 0.8610728978294375,
      "logloss": 0.20340073108673096
    },
    "q1_shuffle": {
      "AUC": 0.8583974720004822,
      "logloss": 0.20548953115940094
    },
    "q2_zero": {
      "AUC": 0.8601198471039927,
      "logloss": 0.2053142935037613
    },
    "q2_shuffle": {
      "AUC": 0.8578740228859746,
      "logloss": 0.20628516376018524
    },
    "q3_zero": {
      "AUC": 0.8598145243163986,
      "logloss": 0.20483438670635223
    },
    "q3_shuffle": {
      "AUC": 0.8566758572575437,
      "logloss": 0.2067047655582428
    },
    "q4_zero": {
      "AUC": 0.860088708226466,
      "logloss": 0.20535553991794586
    },
    "q4_shuffle": {
      "AUC": 0.8555229499800346,
      "logloss": 0.20733195543289185
    },
    "q5_zero": {
      "AUC": 0.8597405264479767,
      "logloss": 0.2039855718612671
    },
    "q5_shuffle": {
      "AUC": 0.8562307675241821,
      "logloss": 0.20853683352470398
    },
    "q6_zero": {
      "AUC": 0.8606808305732808,
      "logloss": 0.20267321169376373
    },
    "q6_shuffle": {
      "AUC": 0.8585832090791513,
      "logloss": 0.2050958275794983
    },
    "q7_zero": {
      "AUC": 0.8611269658024284,
      "logloss": 0.20418915152549744
    },
    "q7_shuffle": {
      "AUC": 0.8585159277291132,
      "logloss": 0.2061009258031845
    },
    "inference_placement_steps1": {
      "AUC": 0.834920350878355,
      "logloss": 0.2765234410762787
    },
    "inference_placement_steps3": {
      "AUC": 0.8343968405125789,
      "logloss": 0.21954821050167084
    },
    "extra_attention_off": {
      "AUC": 0.8496709700146232,
      "logloss": 0.2136465162038803
    }
  },
  "checkpoint_unchanged": true,
  "note": "subset diagnostics; within-block shuffle; blockwise loss means in toolkit differ from pooled full-validation metrics"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 7](#phase-7)。
- **后续假设**：检验一/二/三轮及norm/残差组合；见K08。
- **证据**：`qformer-loop/reports/round7/r7_small_midnorm/summary.json`（SHA256前缀 `c2befdc1da25`）；`qformer-loop/reports/round7/r7_small_midnorm/toolkit-perturbations/perturbation_report.json`（SHA256前缀 `a6bed73b36bc`）；`qformer-loop/reports/round7/r7_small_midnorm/toolkit-hooks/hook_report.json`（SHA256前缀 `40a7160c1ea3`）

<a id="phase-8"></a>
## 阶段 8：循环深度与残差/归一化

**方法**：8个固定seed对照：1/3轮、pre/mid-norm及半幅/双幅更新。

**归因与边界**：同参数完整训练：两轮高于一轮0.001138、三轮低于两轮0.003163；一次循环是低成本排序候选。八个初始YAML失败不算模型效果失败。

**历史后续假设**：分离第二次上下文回读及共享接口的作用；见K09。

<a id="k110"></a>
### K110 · `round8-launch-yaml-r8_first_midnorm`

- **方法**：YAML append reused an existing anchor name
- **原假设**：Launch the frozen structural configuration
- **状态／预算**：原记录 `crash`；{"training_updates":0,"seeds":[20262027]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "decision": "crash",
  "validity": {
    "valid": false,
    "violations": [
      "configuration parser failure before model/data load"
    ],
    "notes": [
      "not scientific counterevidence; no checkpoint or training update"
    ]
  },
  "falsification_result": null,
  "note": "本条未保存独立性能指标；保留运行状态与证据位置，不补造数值。"
}
```

</details>

- **归因**：这是执行或有效性问题；本条不能用作结构性能反证。修复后完整训练需用独立运行ID核对。 本阶段证据解释见[阶段 8](#phase-8)。
- **后续假设**：分离第二次上下文回读及共享接口的作用；见K09。
- **失败后续门槛**：先复核同配置/初始函数/有效梯度及完整预算，再判断科学假设；不把同一次修复重复算成独立支持。
- **证据**：`qformer-loop/reports/round8/training-launch-failed/r8_first_midnorm-s20262027.log`；`qformer-loop/reports/round8/launch-failed-model_config.yaml`（SHA256前缀 `05542615e32c`）

<a id="k111"></a>
### K111 · `round8-launch-yaml-r8_first_half`

- **方法**：YAML append reused an existing anchor name
- **原假设**：Launch the frozen structural configuration
- **状态／预算**：原记录 `crash`；{"training_updates":0,"seeds":[20262027]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "decision": "crash",
  "validity": {
    "valid": false,
    "violations": [
      "configuration parser failure before model/data load"
    ],
    "notes": [
      "not scientific counterevidence; no checkpoint or training update"
    ]
  },
  "falsification_result": null,
  "note": "本条未保存独立性能指标；保留运行状态与证据位置，不补造数值。"
}
```

</details>

- **归因**：这是执行或有效性问题；本条不能用作结构性能反证。修复后完整训练需用独立运行ID核对。 本阶段证据解释见[阶段 8](#phase-8)。
- **后续假设**：分离第二次上下文回读及共享接口的作用；见K09。
- **失败后续门槛**：先复核同配置/初始函数/有效梯度及完整预算，再判断科学假设；不把同一次修复重复算成独立支持。
- **证据**：`qformer-loop/reports/round8/training-launch-failed/r8_first_half-s20262027.log`；`qformer-loop/reports/round8/launch-failed-model_config.yaml`（SHA256前缀 `05542615e32c`）

<a id="k112"></a>
### K112 · `round8-launch-yaml-r8_first_double`

- **方法**：YAML append reused an existing anchor name
- **原假设**：Launch the frozen structural configuration
- **状态／预算**：原记录 `crash`；{"training_updates":0,"seeds":[20262027]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "decision": "crash",
  "validity": {
    "valid": false,
    "violations": [
      "configuration parser failure before model/data load"
    ],
    "notes": [
      "not scientific counterevidence; no checkpoint or training update"
    ]
  },
  "falsification_result": null,
  "note": "本条未保存独立性能指标；保留运行状态与证据位置，不补造数值。"
}
```

</details>

- **归因**：这是执行或有效性问题；本条不能用作结构性能反证。修复后完整训练需用独立运行ID核对。 本阶段证据解释见[阶段 8](#phase-8)。
- **后续假设**：分离第二次上下文回读及共享接口的作用；见K09。
- **失败后续门槛**：先复核同配置/初始函数/有效梯度及完整预算，再判断科学假设；不把同一次修复重复算成独立支持。
- **证据**：`qformer-loop/reports/round8/training-launch-failed/r8_first_double-s20262027.log`；`qformer-loop/reports/round8/launch-failed-model_config.yaml`（SHA256前缀 `05542615e32c`）

<a id="k113"></a>
### K113 · `round8-launch-yaml-r8_first_prenorm`

- **方法**：YAML append reused an existing anchor name
- **原假设**：Launch the frozen structural configuration
- **状态／预算**：原记录 `crash`；{"training_updates":0,"seeds":[20262027]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "decision": "crash",
  "validity": {
    "valid": false,
    "violations": [
      "configuration parser failure before model/data load"
    ],
    "notes": [
      "not scientific counterevidence; no checkpoint or training update"
    ]
  },
  "falsification_result": null,
  "note": "本条未保存独立性能指标；保留运行状态与证据位置，不补造数值。"
}
```

</details>

- **归因**：这是执行或有效性问题；本条不能用作结构性能反证。修复后完整训练需用独立运行ID核对。 本阶段证据解释见[阶段 8](#phase-8)。
- **后续假设**：分离第二次上下文回读及共享接口的作用；见K09。
- **失败后续门槛**：先复核同配置/初始函数/有效梯度及完整预算，再判断科学假设；不把同一次修复重复算成独立支持。
- **证据**：`qformer-loop/reports/round8/training-launch-failed/r8_first_prenorm-s20262027.log`；`qformer-loop/reports/round8/launch-failed-model_config.yaml`（SHA256前缀 `05542615e32c`）

<a id="k114"></a>
### K114 · `round8-launch-yaml-r8_first_prenorm_half`

- **方法**：YAML append reused an existing anchor name
- **原假设**：Launch the frozen structural configuration
- **状态／预算**：原记录 `crash`；{"training_updates":0,"seeds":[20262027]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "decision": "crash",
  "validity": {
    "valid": false,
    "violations": [
      "configuration parser failure before model/data load"
    ],
    "notes": [
      "not scientific counterevidence; no checkpoint or training update"
    ]
  },
  "falsification_result": null,
  "note": "本条未保存独立性能指标；保留运行状态与证据位置，不补造数值。"
}
```

</details>

- **归因**：这是执行或有效性问题；本条不能用作结构性能反证。修复后完整训练需用独立运行ID核对。 本阶段证据解释见[阶段 8](#phase-8)。
- **后续假设**：分离第二次上下文回读及共享接口的作用；见K09。
- **失败后续门槛**：先复核同配置/初始函数/有效梯度及完整预算，再判断科学假设；不把同一次修复重复算成独立支持。
- **证据**：`qformer-loop/reports/round8/training-launch-failed/r8_first_prenorm_half-s20262027.log`；`qformer-loop/reports/round8/launch-failed-model_config.yaml`（SHA256前缀 `05542615e32c`）

<a id="k115"></a>
### K115 · `round8-launch-yaml-r8_first_midnorm_half`

- **方法**：YAML append reused an existing anchor name
- **原假设**：Launch the frozen structural configuration
- **状态／预算**：原记录 `crash`；{"training_updates":0,"seeds":[20262027]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "decision": "crash",
  "validity": {
    "valid": false,
    "violations": [
      "configuration parser failure before model/data load"
    ],
    "notes": [
      "not scientific counterevidence; no checkpoint or training update"
    ]
  },
  "falsification_result": null,
  "note": "本条未保存独立性能指标；保留运行状态与证据位置，不补造数值。"
}
```

</details>

- **归因**：这是执行或有效性问题；本条不能用作结构性能反证。修复后完整训练需用独立运行ID核对。 本阶段证据解释见[阶段 8](#phase-8)。
- **后续假设**：分离第二次上下文回读及共享接口的作用；见K09。
- **失败后续门槛**：先复核同配置/初始函数/有效梯度及完整预算，再判断科学假设；不把同一次修复重复算成独立支持。
- **证据**：`qformer-loop/reports/round8/training-launch-failed/r8_first_midnorm_half-s20262027.log`；`qformer-loop/reports/round8/launch-failed-model_config.yaml`（SHA256前缀 `05542615e32c`）

<a id="k116"></a>
### K116 · `round8-launch-yaml-r8_first_loop1`

- **方法**：YAML append reused an existing anchor name
- **原假设**：Launch the frozen structural configuration
- **状态／预算**：原记录 `crash`；{"training_updates":0,"seeds":[20262027]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "decision": "crash",
  "validity": {
    "valid": false,
    "violations": [
      "configuration parser failure before model/data load"
    ],
    "notes": [
      "not scientific counterevidence; no checkpoint or training update"
    ]
  },
  "falsification_result": null,
  "note": "本条未保存独立性能指标；保留运行状态与证据位置，不补造数值。"
}
```

</details>

- **归因**：这是执行或有效性问题；本条不能用作结构性能反证。修复后完整训练需用独立运行ID核对。 本阶段证据解释见[阶段 8](#phase-8)。
- **后续假设**：分离第二次上下文回读及共享接口的作用；见K09。
- **失败后续门槛**：先复核同配置/初始函数/有效梯度及完整预算，再判断科学假设；不把同一次修复重复算成独立支持。
- **证据**：`qformer-loop/reports/round8/training-launch-failed/r8_first_loop1-s20262027.log`；`qformer-loop/reports/round8/launch-failed-model_config.yaml`（SHA256前缀 `05542615e32c`）

<a id="k117"></a>
### K117 · `round8-launch-yaml-r8_first_loop3`

- **方法**：YAML append reused an existing anchor name
- **原假设**：Launch the frozen structural configuration
- **状态／预算**：原记录 `crash`；{"training_updates":0,"seeds":[20262027]}。
- **结果**：保存的诊断／统计／运行结果如下（子集结果不与完整验证成绩混算）。

<details><summary>展开结果摘要</summary>

```json
{
  "decision": "crash",
  "validity": {
    "valid": false,
    "violations": [
      "configuration parser failure before model/data load"
    ],
    "notes": [
      "not scientific counterevidence; no checkpoint or training update"
    ]
  },
  "falsification_result": null,
  "note": "本条未保存独立性能指标；保留运行状态与证据位置，不补造数值。"
}
```

</details>

- **归因**：这是执行或有效性问题；本条不能用作结构性能反证。修复后完整训练需用独立运行ID核对。 本阶段证据解释见[阶段 8](#phase-8)。
- **后续假设**：分离第二次上下文回读及共享接口的作用；见K09。
- **失败后续门槛**：先复核同配置/初始函数/有效梯度及完整预算，再判断科学假设；不把同一次修复重复算成独立支持。
- **证据**：`qformer-loop/reports/round8/training-launch-failed/r8_first_loop3-s20262027.log`；`qformer-loop/reports/round8/launch-failed-model_config.yaml`（SHA256前缀 `05542615e32c`）

<a id="k118"></a>
### K118 · `r8_first_midnorm-s20262027-20260906T170409368551Z`

- **方法**：{"interface_norm":"per_history"}；设置 {"model":"QFormerCross11Refine","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Refine first-pass attention or test complete-training recurrence depth
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.188735093；AUC=0.870863428；UAUC=0.791509831；gAUC=0.803548575；NDCG(10)=0.410313197。
- **开销**：参数 8115473；训练含验证 441.33458134531975 秒；峰值CUDA 16965337600 bytes；更新 1240。
- **原记录差值**：-0.000370630；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.0006682342933161112,"AUC":-0.00037063030659345575,"UAUC":0.0006728183989679337,"gAUC":-0.00024208717532014923,"NDCG(10)":0.0004570396397259313}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.688592433929443,"mean_ms":12.705947303771973,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000370630，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 8](#phase-8)。
- **后续假设**：分离第二次上下文回读及共享接口的作用；见K09。
- **原判读**：higher UAUC/gAUC/NDCG than parent at similar cost; retained ranking tradeoff
- **证据**：`qformer-loop/runs/r8_first_midnorm-s20262027-20260906T170409368551Z/manifest.json`（SHA256前缀 `ef7b24e3e104`）；`qformer-loop/runs/r8_first_midnorm-s20262027-20260906T170409368551Z/result.json`（SHA256前缀 `edbe53db2839`）；`qformer-loop/runs/r8_first_midnorm-s20262027-20260906T170409368551Z/inference-all_queries.json`（SHA256前缀 `50fc2d2fe1db`）

<a id="k119"></a>
### K119 · `r8_first_half-s20262027-20260906T170409452426Z`

- **方法**：{"mixer_residual_scale":0.5}；设置 {"model":"QFormerCross11Refine","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Refine first-pass attention or test complete-training recurrence depth
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.189195759；AUC=0.869135718；UAUC=0.790244419；gAUC=0.802460197；NDCG(10)=0.408339564。
- **开销**：参数 8115473；训练含验证 440.1495677381754 秒；峰值CUDA 16897573376 bytes；更新 1240。
- **原记录差值**：-0.002098341；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.0011289009378311898,"AUC":-0.002098340512008079,"UAUC":-0.0005925936243496022,"gAUC":-0.001330465627257138,"NDCG(10)":-0.0015165932032016327}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.868607997894287,"mean_ms":12.949709582328797,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.002098341，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 8](#phase-8)。
- **后续假设**：分离第二次上下文回读及共享接口的作用；见K09。
- **原判读**：deprioritized post-norm half-residual control
- **证据**：`qformer-loop/runs/r8_first_half-s20262027-20260906T170409452426Z/manifest.json`（SHA256前缀 `c929960574c3`）；`qformer-loop/runs/r8_first_half-s20262027-20260906T170409452426Z/result.json`（SHA256前缀 `1fe87b186945`）；`qformer-loop/runs/r8_first_half-s20262027-20260906T170409452426Z/inference-all_queries.json`（SHA256前缀 `24556edea9ca`）

<a id="k120"></a>
### K120 · `r8_first_double-s20262027-20260906T170409454885Z`

- **方法**：{"mixer_residual_scale":2.0}；设置 {"model":"QFormerCross11Refine","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Refine first-pass attention or test complete-training recurrence depth
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.188394688；AUC=0.870764323；UAUC=0.790309119；gAUC=0.802158707；NDCG(10)=0.410965797。
- **开销**：参数 8115473；训练含验证 437.73770025372505 秒；峰值CUDA 16897573376 bytes；更新 1240。
- **原记录差值**：-0.000469735；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.00032782904727038376,"AUC":-0.00046973529318794505,"UAUC":-0.0005278937865339506,"gAUC":-0.001631955475246527,"NDCG(10)":0.0011096396904890504}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.801487922668457,"mean_ms":12.95434799194336,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000469735，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 8](#phase-8)。
- **后续假设**：分离第二次上下文回读及共享接口的作用；见K09。
- **原判读**：higher NDCG but lower AUC/UAUC than parent; retained alternative
- **证据**：`qformer-loop/runs/r8_first_double-s20262027-20260906T170409454885Z/manifest.json`（SHA256前缀 `498b55252018`）；`qformer-loop/runs/r8_first_double-s20262027-20260906T170409454885Z/result.json`（SHA256前缀 `704d6a581545`）；`qformer-loop/runs/r8_first_double-s20262027-20260906T170409454885Z/inference-all_queries.json`（SHA256前缀 `62335582201a`）

<a id="k121"></a>
### K121 · `r8_first_prenorm-s20262027-20260906T170409169559Z`

- **方法**：{"mixer_norm_position":"pre"}；设置 {"model":"QFormerCross11Refine","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Refine first-pass attention or test complete-training recurrence depth
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.188400565；AUC=0.870590652；UAUC=0.790212587；gAUC=0.802012936；NDCG(10)=0.410301293。
- **开销**：参数 8115473；训练含验证 441.99012396484613 秒；峰值CUDA 16897573376 bytes；更新 1240。
- **原记录差值**：-0.000643406；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.0003337069604698317,"AUC":-0.0006434061293170146,"UAUC":-0.0006244254675698002,"gAUC":-0.0017777263956155753,"NDCG(10)":0.0004451357645147702}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.68449592590332,"mean_ms":12.718667221069335,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000643406，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 8](#phase-8)。
- **后续假设**：分离第二次上下文回读及共享接口的作用；见K09。
- **原判读**：near-baseline alternative norm position; lower AUC than parent
- **证据**：`qformer-loop/runs/r8_first_prenorm-s20262027-20260906T170409169559Z/manifest.json`（SHA256前缀 `f48073af5862`）；`qformer-loop/runs/r8_first_prenorm-s20262027-20260906T170409169559Z/result.json`（SHA256前缀 `cd3c11ccb66f`）；`qformer-loop/runs/r8_first_prenorm-s20262027-20260906T170409169559Z/inference-all_queries.json`（SHA256前缀 `4dcbd3d65c38`）

<a id="k122"></a>
### K122 · `r8_first_prenorm_half-s20262027-20260906T170409454737Z`

- **方法**：{"mixer_norm_position":"pre","mixer_residual_scale":0.5}；设置 {"model":"QFormerCross11Refine","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Refine first-pass attention or test complete-training recurrence depth
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.188512719；AUC=0.870984498；UAUC=0.788637907；gAUC=0.800616941；NDCG(10)=0.411015743。
- **开销**：参数 8115473；训练含验证 439.76321171969175 秒；峰值CUDA 16897573376 bytes；更新 1240。
- **原记录差值**：-0.000249560；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.00044586014976305544,"AUC":-0.0002495600351387628,"UAUC":-0.0021991055853585184,"gAUC":-0.003173720705181071,"NDCG(10)":0.0011595862944756008}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.702224254608154,"mean_ms":12.712445640563965,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000249560，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 8](#phase-8)。
- **后续假设**：分离第二次上下文回读及共享接口的作用；见K09。
- **原判读**：near-parent AUC and better NDCG, but lower UAUC; retained alternative
- **证据**：`qformer-loop/runs/r8_first_prenorm_half-s20262027-20260906T170409454737Z/manifest.json`（SHA256前缀 `fe829f182dbf`）；`qformer-loop/runs/r8_first_prenorm_half-s20262027-20260906T170409454737Z/result.json`（SHA256前缀 `5d31d34fa227`）；`qformer-loop/runs/r8_first_prenorm_half-s20262027-20260906T170409454737Z/inference-all_queries.json`（SHA256前缀 `a71e43d2d2e0`）

<a id="k123"></a>
### K123 · `r8_first_midnorm_half-s20262027-20260906T170409455169Z`

- **方法**：{"interface_norm":"per_history","mixer_residual_scale":0.5}；设置 {"model":"QFormerCross11Refine","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Refine first-pass attention or test complete-training recurrence depth
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.189429367；AUC=0.869058069；UAUC=0.790112826；gAUC=0.802319554；NDCG(10)=0.408467341。
- **开销**：参数 8115473；训练含验证 440.73837637901306 秒；峰值CUDA 16965337600 bytes；更新 1240。
- **原记录差值**：-0.002175990；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.001362508213749003,"AUC":-0.002175989638866338,"UAUC":-0.0007241866478557846,"gAUC":-0.0014711080827513046,"NDCG(10)":-0.0013888162747088262}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.729840278625488,"mean_ms":12.744553709030152,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.002175990，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 8](#phase-8)。
- **后续假设**：分离第二次上下文回读及共享接口的作用；见K09。
- **原判读**：deprioritized norm and half-residual combination
- **证据**：`qformer-loop/runs/r8_first_midnorm_half-s20262027-20260906T170409455169Z/manifest.json`（SHA256前缀 `3d0a10b506ed`）；`qformer-loop/runs/r8_first_midnorm_half-s20262027-20260906T170409455169Z/result.json`（SHA256前缀 `28d349bae7e8`）；`qformer-loop/runs/r8_first_midnorm_half-s20262027-20260906T170409455169Z/inference-all_queries.json`（SHA256前缀 `3a53f720b64d`）

<a id="k124"></a>
### K124 · `r8_first_loop1-s20262027-20260906T170409366470Z`

- **方法**：{"placement_steps":1}；设置 {"model":"QFormerCross11Refine","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Refine first-pass attention or test complete-training recurrence depth
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.188918736；AUC=0.870095741；UAUC=0.792500634；gAUC=0.804015568；NDCG(10)=0.412649991。
- **开销**：参数 8115473；训练含验证 336.9914402961731 秒；峰值CUDA 10535460352 bytes；更新 1240。
- **原记录差值**：-0.001138318；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.000851877288155195,"AUC":-0.0011383176251943405,"UAUC":0.0016636217051131386,"gAUC":0.00022490596457525047,"NDCG(10)":0.0027938345186523383}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":8.534687995910645,"mean_ms":8.537121629714965,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.001138318，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 8](#phase-8)。
- **后续假设**：分离第二次上下文回读及共享接口的作用；见K09。
- **原判读**：preferred low-cost ranking candidate; lower AUC than two cycles
- **证据**：`qformer-loop/runs/r8_first_loop1-s20262027-20260906T170409366470Z/manifest.json`（SHA256前缀 `5bcb3c712e20`）；`qformer-loop/runs/r8_first_loop1-s20262027-20260906T170409366470Z/result.json`（SHA256前缀 `603926c225f7`）；`qformer-loop/runs/r8_first_loop1-s20262027-20260906T170409366470Z/inference-all_queries.json`（SHA256前缀 `61f28fc8afde`）

<a id="k125"></a>
### K125 · `r8_first_loop3-s20262027-20260906T170444520001Z`

- **方法**：{"placement_steps":3}；设置 {"model":"QFormerCross11Refine","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Refine first-pass attention or test complete-training recurrence depth
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.189753029；AUC=0.868070803；UAUC=0.789770862；gAUC=0.801828258；NDCG(10)=0.408941456。
- **开销**：参数 8115473；训练含验证 761.8276031166315 秒；峰值CUDA 22716254720 bytes；更新 1240。
- **原记录差值**：-0.003163256；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.0016861707892253386,"AUC":-0.003163255566617451,"UAUC":-0.0010661502581322813,"gAUC":-0.0019624036963381553,"NDCG(10)":-0.0009147010696549107}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":16.746928215026855,"mean_ms":16.74810709953308,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.003163256，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 8](#phase-8)。
- **后续假设**：分离第二次上下文回读及共享接口的作用；见K09。
- **原判读**：deprioritized depth control; lower quality and greater forward cost
- **证据**：`qformer-loop/runs/r8_first_loop3-s20262027-20260906T170444520001Z/manifest.json`（SHA256前缀 `7f54228f0357`）；`qformer-loop/runs/r8_first_loop3-s20262027-20260906T170444520001Z/result.json`（SHA256前缀 `a24a428db30e`）；`qformer-loop/runs/r8_first_loop3-s20262027-20260906T170444520001Z/inference-all_queries.json`（SHA256前缀 `e317072db94b`）

<a id="k126"></a>
### K126 · `diagnostics-round8-r8_first_loop1`

- **方法**：Query Zero/Shuffle, extra attention off and inference-depth interventions
- **原假设**：Measure reliance at the actual output and compare depth perturbations with complete depth retraining
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
      "AUC": 0.8623228419006933,
      "logloss": 0.20249946415424347
    },
    "all_queries": {
      "AUC": 0.8623228419006933,
      "logloss": 0.20249946415424347
    },
    "q3_half": {
      "AUC": 0.8623246012905719,
      "logloss": 0.20307470858097076
    },
    "q0_zero": {
      "AUC": 0.8621443197990052,
      "logloss": 0.20774388313293457
    },
    "q0_shuffle": {
      "AUC": 0.8560687896012682,
      "logloss": 0.20827335119247437
    },
    "q1_zero": {
      "AUC": 0.8613285690716951,
      "logloss": 0.20517116785049438
    },
    "q1_shuffle": {
      "AUC": 0.8572989790824468,
      "logloss": 0.20785404741764069
    },
    "q2_zero": {
      "AUC": 0.8616202835171672,
      "logloss": 0.20336946845054626
    },
    "q2_shuffle": {
      "AUC": 0.8592715994106901,
      "logloss": 0.20552541315555573
    },
    "q3_zero": {
      "AUC": 0.8620155083818093,
      "logloss": 0.20421202480793
    },
    "q3_shuffle": {
      "AUC": 0.8605292717023127,
      "logloss": 0.20476703345775604
    },
    "q4_zero": {
      "AUC": 0.8620721214506031,
      "logloss": 0.20660626888275146
    },
    "q4_shuffle": {
      "AUC": 0.8557627022288771,
      "logloss": 0.20873427391052246
    },
    "q5_zero": {
      "AUC": 0.8616009682465315,
      "logloss": 0.20460109412670135
    },
    "q5_shuffle": {
      "AUC": 0.8588975421395209,
      "logloss": 0.20573222637176514
    },
    "q6_zero": {
      "AUC": 0.8618302739859628,
      "logloss": 0.20870964229106903
    },
    "q6_shuffle": {
      "AUC": 0.8598952133578522,
      "logloss": 0.20548118650913239
    },
    "q7_zero": {
      "AUC": 0.8619810788326973,
      "logloss": 0.20321553945541382
    },
    "q7_shuffle": {
      "AUC": 0.8600007788626791,
      "logloss": 0.20555739104747772
    },
    "inference_placement_steps1": {
      "AUC": 0.8623228419006933,
      "logloss": 0.20249946415424347
    },
    "inference_placement_steps3": {
      "AUC": 0.756626909890581,
      "logloss": 0.30736902356147766
    },
    "extra_attention_off": {
      "AUC": 0.8564067022878135,
      "logloss": 0.2064024955034256
    }
  },
  "checkpoint_unchanged": true,
  "note": "subset diagnostics; within-block shuffle; blockwise loss means in toolkit differ from pooled full-validation metrics"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 8](#phase-8)。
- **后续假设**：分离第二次上下文回读及共享接口的作用；见K09。
- **证据**：`qformer-loop/reports/round8/r8_first_loop1/summary.json`（SHA256前缀 `2a12b655dc98`）；`qformer-loop/reports/round8/r8_first_loop1/toolkit-hooks/hook_report.json`（SHA256前缀 `2bcc8ef67ad3`）；`qformer-loop/reports/round8/r8_first_loop1/toolkit-perturbations/perturbation_report.json`（SHA256前缀 `53436d146456`）

<a id="k127"></a>
### K127 · `diagnostics-round8-r8_first_midnorm`

- **方法**：Query Zero/Shuffle, extra attention off and inference-depth interventions
- **原假设**：Measure reliance at the actual output and compare depth perturbations with complete depth retraining
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
      "AUC": 0.8614546558635088,
      "logloss": 0.202816903591156
    },
    "all_queries": {
      "AUC": 0.8614546558635088,
      "logloss": 0.202816903591156
    },
    "q3_half": {
      "AUC": 0.861562584822446,
      "logloss": 0.2027900218963623
    },
    "q0_zero": {
      "AUC": 0.8604114115894631,
      "logloss": 0.20475755631923676
    },
    "q0_shuffle": {
      "AUC": 0.8581635238373377,
      "logloss": 0.20648162066936493
    },
    "q1_zero": {
      "AUC": 0.8605571558141698,
      "logloss": 0.2039012908935547
    },
    "q1_shuffle": {
      "AUC": 0.8596439712159254,
      "logloss": 0.20440377295017242
    },
    "q2_zero": {
      "AUC": 0.8598745146533986,
      "logloss": 0.20593425631523132
    },
    "q2_shuffle": {
      "AUC": 0.854161553945859,
      "logloss": 0.21045979857444763
    },
    "q3_zero": {
      "AUC": 0.8608756011579649,
      "logloss": 0.20534546673297882
    },
    "q3_shuffle": {
      "AUC": 0.8563702134288194,
      "logloss": 0.20854538679122925
    },
    "q4_zero": {
      "AUC": 0.8602069358469254,
      "logloss": 0.20394852757453918
    },
    "q4_shuffle": {
      "AUC": 0.8586533797993621,
      "logloss": 0.20666132867336273
    },
    "q5_zero": {
      "AUC": 0.861413309145306,
      "logloss": 0.20339557528495789
    },
    "q5_shuffle": {
      "AUC": 0.8585653469420528,
      "logloss": 0.20641376078128815
    },
    "q6_zero": {
      "AUC": 0.86110221817793,
      "logloss": 0.20932607352733612
    },
    "q6_shuffle": {
      "AUC": 0.8591118117486455,
      "logloss": 0.205906942486763
    },
    "q7_zero": {
      "AUC": 0.8596905580857351,
      "logloss": 0.20423924922943115
    },
    "q7_shuffle": {
      "AUC": 0.8585519223537358,
      "logloss": 0.2062321901321411
    },
    "inference_placement_steps1": {
      "AUC": 0.811635850210184,
      "logloss": 0.2385670691728592
    },
    "inference_placement_steps3": {
      "AUC": 0.8069621756774914,
      "logloss": 0.23321682214736938
    },
    "extra_attention_off": {
      "AUC": 0.8583642442434959,
      "logloss": 0.20522558689117432
    }
  },
  "checkpoint_unchanged": true,
  "note": "subset diagnostics; within-block shuffle; blockwise loss means in toolkit differ from pooled full-validation metrics"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 8](#phase-8)。
- **后续假设**：分离第二次上下文回读及共享接口的作用；见K09。
- **证据**：`qformer-loop/reports/round8/r8_first_midnorm/summary.json`（SHA256前缀 `1002402a22a1`）；`qformer-loop/reports/round8/r8_first_midnorm/toolkit-hooks/hook_report.json`（SHA256前缀 `e0b5a3fb57ef`）；`qformer-loop/reports/round8/r8_first_midnorm/toolkit-perturbations/perturbation_report.json`（SHA256前缀 `34f5e0a2db6d`）

<a id="phase-9"></a>
## 阶段 9：第二次回读与分轮共享

**方法**：省略/替换/前移/后移上下文回读；克隆独立历史权重；字段/历史分轮norm。

**归因与边界**：省略第二次上下文回读AUC下降0.001522；克隆独立历史权重下降0.000461并增参。结论限定原交替结构和预算。

**历史后续假设**：逐一精简第二轮self-attention、cross和FFN；见K10。

<a id="k128"></a>
### K128 · `r9_feedback_omit-s20262027-20260906T173725470939Z`

- **方法**：{"feedback_schedule":"omit"}；设置 {"model":"QFormerCross11Disentangle","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Disentangle context feedback, shared history and pass-specific interfaces
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.188931910；AUC=0.869711753；UAUC=0.788870839；gAUC=0.801447379；NDCG(10)=0.405604349。
- **开销**：参数 8115473；训练含验证 396.0555749684572 秒；峰值CUDA 14711291392 bytes；更新 1240。
- **原记录差值**：-0.001522305；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.0008650518972736221,"AUC":-0.0015223054810800107,"UAUC":-0.0019661735073414066,"gAUC":-0.0023432833899733607,"NDCG(10)":-0.004251807726169565}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":11.102112293243408,"mean_ms":11.108202528953552,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.001522305，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 9](#phase-9)。
- **后续假设**：逐一精简第二轮self-attention、cross和FFN；见K10。
- **原判读**：low-cost mechanism control; dominated by earlier one-cycle candidate
- **证据**：`qformer-loop/runs/r9_feedback_omit-s20262027-20260906T173725470939Z/manifest.json`（SHA256前缀 `9149b2323707`）；`qformer-loop/runs/r9_feedback_omit-s20262027-20260906T173725470939Z/result.json`（SHA256前缀 `dc4f2f9182af`）；`qformer-loop/runs/r9_feedback_omit-s20262027-20260906T173725470939Z/inference-all_queries.json`（SHA256前缀 `121080b31405`）

<a id="k129"></a>
### K129 · `r9_history_replace-s20262027-20260906T173725574888Z`

- **方法**：{"feedback_schedule":"history_replace"}；设置 {"model":"QFormerCross11Disentangle","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Disentangle context feedback, shared history and pass-specific interfaces
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.188676522；AUC=0.870560993；UAUC=0.788486304；gAUC=0.800643855；NDCG(10)=0.404844131。
- **开销**：参数 8115473；训练含验证 470.5872892290354 秒；峰值CUDA 18477711872 bytes；更新 1240。
- **原记录差值**：-0.000673065；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.0006096639957334227,"AUC":-0.0006730652020393579,"UAUC":-0.0023507085837917163,"gAUC":-0.0031468066719294585,"NDCG(10)":-0.00501202621808311}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":13.72817611694336,"mean_ms":13.790236806869506,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000673065，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 9](#phase-9)。
- **后续假设**：逐一精简第二轮self-attention、cross和FFN；见K10。
- **原判读**：mechanism candidate above Cross8/Cross11 AUC; lower ranking and greater forward cost than parent
- **证据**：`qformer-loop/runs/r9_history_replace-s20262027-20260906T173725574888Z/manifest.json`（SHA256前缀 `c45b83a2b7fb`）；`qformer-loop/runs/r9_history_replace-s20262027-20260906T173725574888Z/result.json`（SHA256前缀 `e8ac1ec2aca3`）；`qformer-loop/runs/r9_history_replace-s20262027-20260906T173725574888Z/inference-all_queries.json`（SHA256前缀 `c6ac457a6368`）

<a id="k130"></a>
### K130 · `r9_feedback_before-s20262027-20260906T173725569782Z`

- **方法**：{"feedback_schedule":"before"}；设置 {"model":"QFormerCross11Disentangle","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Disentangle context feedback, shared history and pass-specific interfaces
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.189374191；AUC=0.869212680；UAUC=0.790496276；gAUC=0.802101829；NDCG(10)=0.407101208。
- **开销**：参数 8115473；训练含验证 439.5578867867589 秒；峰值CUDA 16762634752 bytes；更新 1240。
- **原记录差值**：-0.002021378；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.001307332643144643,"AUC":-0.002021378047961986,"UAUC":-0.00034073682623980694,"gAUC":-0.0016888333579444526,"NDCG(10)":-0.002754949046790578}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.63483190536499,"mean_ms":12.644612002372742,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.002021378，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 9](#phase-9)。
- **后续假设**：逐一精简第二轮self-attention、cross和FFN；见K10。
- **原判读**：near-Cross8 placement control; lower AUC than parent
- **证据**：`qformer-loop/runs/r9_feedback_before-s20262027-20260906T173725569782Z/manifest.json`（SHA256前缀 `1603625d8a25`）；`qformer-loop/runs/r9_feedback_before-s20262027-20260906T173725569782Z/result.json`（SHA256前缀 `17f0284dc17d`）；`qformer-loop/runs/r9_feedback_before-s20262027-20260906T173725569782Z/inference-all_queries.json`（SHA256前缀 `422bae2f0eae`）

<a id="k131"></a>
### K131 · `r9_feedback_after-s20262027-20260906T173725584656Z`

- **方法**：{"feedback_schedule":"after"}；设置 {"model":"QFormerCross11Disentangle","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Disentangle context feedback, shared history and pass-specific interfaces
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.189769540；AUC=0.867980082；UAUC=0.789683554；gAUC=0.801945722；NDCG(10)=0.407134847。
- **开销**：参数 8115473；训练含验证 440.02844016999006 秒；峰值CUDA 15742305792 bytes；更新 1240。
- **原记录差值**：-0.003253977；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.0017026810778469625,"AUC":-0.0032539766018385707,"UAUC":-0.0011534591474087197,"gAUC":-0.0018449397464915807,"NDCG(10)":-0.0027213094880200184}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.634016036987305,"mean_ms":12.691316843032837,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.003253977，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 9](#phase-9)。
- **后续假设**：逐一精简第二轮self-attention、cross和FFN；见K10。
- **原判读**：deprioritized terminal-feedback control
- **证据**：`qformer-loop/runs/r9_feedback_after-s20262027-20260906T173725584656Z/manifest.json`（SHA256前缀 `c21d41df3d02`）；`qformer-loop/runs/r9_feedback_after-s20262027-20260906T173725584656Z/result.json`（SHA256前缀 `1e2d4e27eb94`）；`qformer-loop/runs/r9_feedback_after-s20262027-20260906T173725584656Z/inference-all_queries.json`（SHA256前缀 `e7449c3da6ee`）

<a id="k132"></a>
### K132 · `r9_field_splitnorm-s20262027-20260906T173725574982Z`

- **方法**：{"split_interface_norms":"field"}；设置 {"model":"QFormerCross11Disentangle","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Disentangle context feedback, shared history and pass-specific interfaces
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.188537784；AUC=0.870864134；UAUC=0.790624320；gAUC=0.802984836；NDCG(10)=0.409086828。
- **开销**：参数 8115985；训练含验证 439.78022819012403 秒；峰值CUDA 16829618688 bytes；更新 1240。
- **原记录差值**：-0.000369925；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.00047092534189233093,"AUC":-0.0003699247760086122,"UAUC":-0.00021269301445614452,"gAUC":-0.0008058259997847372,"NDCG(10)":-0.0007693288768780282}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.636367797851562,"mean_ms":12.643381571769714,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000369925，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 9](#phase-9)。
- **后续假设**：逐一精简第二轮self-attention、cross和FFN；见K10。
- **原判读**：near-baseline alternative; lower AUC and guardrails than shared-norm parent
- **证据**：`qformer-loop/runs/r9_field_splitnorm-s20262027-20260906T173725574982Z/manifest.json`（SHA256前缀 `f3c952967d0f`）；`qformer-loop/runs/r9_field_splitnorm-s20262027-20260906T173725574982Z/result.json`（SHA256前缀 `ad5e46e10cb1`）；`qformer-loop/runs/r9_field_splitnorm-s20262027-20260906T173725574982Z/inference-all_queries.json`（SHA256前缀 `2dd35e1a0470`）

<a id="k133"></a>
### K133 · `r9_history_splitnorm-s20262027-20260906T173725520543Z`

- **方法**：{"split_interface_norms":"history"}；设置 {"model":"QFormerCross11Disentangle","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Disentangle context feedback, shared history and pass-specific interfaces
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.188731717；AUC=0.869939441；UAUC=0.790593772；gAUC=0.803118956；NDCG(10)=0.409263669。
- **开销**：参数 8115985；训练含验证 451.2861417531967 秒；峰值CUDA 16898103808 bytes；更新 1240。
- **原记录差值**：-0.000923987；对照 `same-seed r8_first_midnorm`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r8_first_midnorm`；完整指标差 {"logloss":-3.3753398158142822e-06,"AUC":-0.0009239868622251679,"UAUC":-0.0009160589707641886,"gAUC":-0.00042961884843073506,"NDCG(10)":-0.0010495272086587515}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.676192283630371,"mean_ms":12.68407120704651,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000923987，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 9](#phase-9)。
- **后续假设**：逐一精简第二轮self-attention、cross和FFN；见K10。
- **原判读**：near-Cross11 norm-sharing control
- **证据**：`qformer-loop/runs/r9_history_splitnorm-s20262027-20260906T173725520543Z/manifest.json`（SHA256前缀 `5c1c72a1d4aa`）；`qformer-loop/runs/r9_history_splitnorm-s20262027-20260906T173725520543Z/result.json`（SHA256前缀 `0f451b2652a7`）；`qformer-loop/runs/r9_history_splitnorm-s20262027-20260906T173725520543Z/inference-all_queries.json`（SHA256前缀 `298eba9185f3`）

<a id="k134"></a>
### K134 · `r9_both_splitnorm-s20262027-20260906T173725417308Z`

- **方法**：{"split_interface_norms":"both"}；设置 {"model":"QFormerCross11Disentangle","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Disentangle context feedback, shared history and pass-specific interfaces
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.188609575；AUC=0.870748711；UAUC=0.789850915；gAUC=0.802003245；NDCG(10)=0.406917847。
- **开销**：参数 8116497；训练含验证 440.8567224442959 秒；峰值CUDA 16898109952 bytes；更新 1240。
- **原记录差值**：-0.000114717；对照 `same-seed r8_first_midnorm`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r8_first_midnorm`；完整指标差 {"logloss":-0.00012551735867685765,"AUC":-0.0001147174644634541,"UAUC":-0.001658915897346791,"gAUC":-0.0015453304590176131,"NDCG(10)":-0.0033953494489600278}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.660608291625977,"mean_ms":12.664615273475647,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000114717，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 9](#phase-9)。
- **后续假设**：逐一精简第二轮self-attention、cross和FFN；见K10。
- **原判读**：near-baseline alternative; slightly better Logloss but worse AUC/ranking than matched midnorm parent
- **证据**：`qformer-loop/runs/r9_both_splitnorm-s20262027-20260906T173725417308Z/manifest.json`（SHA256前缀 `452d0afd0727`）；`qformer-loop/runs/r9_both_splitnorm-s20262027-20260906T173725417308Z/result.json`（SHA256前缀 `7a4248392b98`）；`qformer-loop/runs/r9_both_splitnorm-s20262027-20260906T173725417308Z/inference-all_queries.json`（SHA256前缀 `3f53f8ef0d15`）

<a id="k135"></a>
### K135 · `r9_untied_clone-s20262027-20260906T174430682740Z`

- **方法**：{"untie_history":true}；设置 {"model":"QFormerCross11Disentangle","token_dim":256,"num_queries":8,"num_ns_layers":1,"num_unified_layers":1,"dense_optimizer":"AdamW","dense_learning_rate":0.0001,"sparse_learning_rate":0.05,"max_gradient_norm":1}
- **原假设**：Disentangle context feedback, shared history and pass-specific interfaces
- **状态／预算**：原记录 `provisional`；{"epochs":1,"optimizer_steps":1240,"seeds":[20262027]}。
- **结果**：logloss=0.188589479；AUC=0.870773431；UAUC=0.790631600；gAUC=0.802791459；NDCG(10)=0.408268059。
- **开销**：参数 9301009；训练含验证 436.63210805505514 秒；峰值CUDA 16844766208 bytes；更新 1240。
- **原记录差值**：-0.000460628；对照 `same-seed r7_small_attention_first`。仅按该对照解释，不能当成相对当前最优的差。
- **结构父对照**：`r7_small_attention_first`；完整指标差 {"logloss":0.0005226207452267151,"AUC":-0.0004606278766611238,"UAUC":-0.00020541270253882082,"gAUC":-0.0009992035456557469,"NDCG(10)":-0.001588097993245663}。这是阶段保存的父对照，可能不同于上方Cross8或全局最佳参考。
- **保存的计时**：{"batch_size":1024,"iterations":40,"precision":"bf16","p50_ms":12.657936096191406,"mean_ms":12.673667192459106,"scope":"model forward only on one repeated real validation batch; excludes IO"}；使用对应报告口径，不跨轮直接计算速度差。
- **归因**：相对上方记录对照的主指标点估计降低0.000460628，仅此不能认定稳健提升或统计等价。这是固定数据、预算及记录seed下的训练结果；结构因果解释仍受对照完整性、优化器交互及连续筛选影响。 本阶段证据解释见[阶段 9](#phase-9)。
- **后续假设**：逐一精简第二轮self-attention、cross和FFN；见K10。
- **原判读**：higher-parameter alternative; lower AUC than shared history at matched initial function
- **证据**：`qformer-loop/runs/r9_untied_clone-s20262027-20260906T174430682740Z/manifest.json`（SHA256前缀 `cf76d8d1f752`）；`qformer-loop/runs/r9_untied_clone-s20262027-20260906T174430682740Z/result.json`（SHA256前缀 `cffbfb6b732b`）；`qformer-loop/runs/r9_untied_clone-s20262027-20260906T174430682740Z/inference-all_queries.json`（SHA256前缀 `45f4e5a2c3f3`）

<a id="k136"></a>
### K136 · `diagnostics-round9-r9_field_splitnorm`

- **方法**：Query Zero/Shuffle, extra attention off and inference-depth interventions
- **原假设**：Measure actual Query output reliance in split-norm and independently trained history controls
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
      "AUC": 0.8617431007585221,
      "logloss": 0.20270468294620514
    },
    "all_queries": {
      "AUC": 0.8617431007585221,
      "logloss": 0.20270468294620514
    },
    "q3_half": {
      "AUC": 0.8616719014394267,
      "logloss": 0.2032647281885147
    },
    "q0_zero": {
      "AUC": 0.8610004903818781,
      "logloss": 0.20584265887737274
    },
    "q0_shuffle": {
      "AUC": 0.8576146258769155,
      "logloss": 0.2075776755809784
    },
    "q1_zero": {
      "AUC": 0.8605854222184253,
      "logloss": 0.20335468649864197
    },
    "q1_shuffle": {
      "AUC": 0.8606488278416601,
      "logloss": 0.2041618376970291
    },
    "q2_zero": {
      "AUC": 0.8598897725555387,
      "logloss": 0.20649434626102448
    },
    "q2_shuffle": {
      "AUC": 0.8565452695535677,
      "logloss": 0.20857860147953033
    },
    "q3_zero": {
      "AUC": 0.8607172602931192,
      "logloss": 0.20598484575748444
    },
    "q3_shuffle": {
      "AUC": 0.8584543849892793,
      "logloss": 0.20627589523792267
    },
    "q4_zero": {
      "AUC": 0.8602754865766958,
      "logloss": 0.20489105582237244
    },
    "q4_shuffle": {
      "AUC": 0.8567335580643776,
      "logloss": 0.20765572786331177
    },
    "q5_zero": {
      "AUC": 0.8606551705160962,
      "logloss": 0.20462515950202942
    },
    "q5_shuffle": {
      "AUC": 0.85856666701249,
      "logloss": 0.20604923367500305
    },
    "q6_zero": {
      "AUC": 0.8607476557069758,
      "logloss": 0.21018962562084198
    },
    "q6_shuffle": {
      "AUC": 0.857684714224731,
      "logloss": 0.20672187209129333
    },
    "q7_zero": {
      "AUC": 0.8607097622930366,
      "logloss": 0.20425909757614136
    },
    "q7_shuffle": {
      "AUC": 0.8594614656696863,
      "logloss": 0.2043038010597229
    },
    "inference_placement_steps1": {
      "AUC": 0.8208936858272574,
      "logloss": 0.2417023479938507
    },
    "inference_placement_steps3": {
      "AUC": 0.7995652739812317,
      "logloss": 0.23525284230709076
    },
    "extra_attention_off": {
      "AUC": 0.8585915878302297,
      "logloss": 0.20489373803138733
    }
  },
  "checkpoint_unchanged": true,
  "note": "subset diagnostics; within-block shuffle; blockwise loss means in toolkit differ from pooled full-validation metrics"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 9](#phase-9)。
- **后续假设**：逐一精简第二轮self-attention、cross和FFN；见K10。
- **证据**：`qformer-loop/reports/round9/r9_field_splitnorm/summary.json`（SHA256前缀 `9d7c09d2699c`）；`qformer-loop/reports/round9/r9_field_splitnorm/toolkit-hooks/hook_report.json`（SHA256前缀 `1330b64e8814`）；`qformer-loop/reports/round9/r9_field_splitnorm/toolkit-perturbations/perturbation_report.json`（SHA256前缀 `0fb9f46f5a38`）

<a id="k137"></a>
### K137 · `diagnostics-round9-r9_untied_clone`

- **方法**：Query Zero/Shuffle, extra attention off and inference-depth interventions
- **原假设**：Measure actual Query output reliance in split-norm and independently trained history controls
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
      "AUC": 0.8621203515440929,
      "logloss": 0.20264722406864166
    },
    "all_queries": {
      "AUC": 0.8621203515440929,
      "logloss": 0.20264722406864166
    },
    "q3_half": {
      "AUC": 0.8618531143726937,
      "logloss": 0.20264644920825958
    },
    "q0_zero": {
      "AUC": 0.8616512682104667,
      "logloss": 0.20396307110786438
    },
    "q0_shuffle": {
      "AUC": 0.8587670832744194,
      "logloss": 0.20646104216575623
    },
    "q1_zero": {
      "AUC": 0.8592181777441853,
      "logloss": 0.20665010809898376
    },
    "q1_shuffle": {
      "AUC": 0.8557254234397333,
      "logloss": 0.2091476172208786
    },
    "q2_zero": {
      "AUC": 0.8611474744167392,
      "logloss": 0.20421579480171204
    },
    "q2_shuffle": {
      "AUC": 0.8577522448040675,
      "logloss": 0.20755067467689514
    },
    "q3_zero": {
      "AUC": 0.8612793568457999,
      "logloss": 0.20494385063648224
    },
    "q3_shuffle": {
      "AUC": 0.8598614786938182,
      "logloss": 0.2050762176513672
    },
    "q4_zero": {
      "AUC": 0.8603677415472912,
      "logloss": 0.20770514011383057
    },
    "q4_shuffle": {
      "AUC": 0.8598393437527287,
      "logloss": 0.20562154054641724
    },
    "q5_zero": {
      "AUC": 0.861517791136318,
      "logloss": 0.20545241236686707
    },
    "q5_shuffle": {
      "AUC": 0.8577677160295905,
      "logloss": 0.2077283263206482
    },
    "q6_zero": {
      "AUC": 0.8612111482782869,
      "logloss": 0.2067703753709793
    },
    "q6_shuffle": {
      "AUC": 0.8599003901460786,
      "logloss": 0.20543065667152405
    },
    "q7_zero": {
      "AUC": 0.8616391552441359,
      "logloss": 0.20292064547538757
    },
    "q7_shuffle": {
      "AUC": 0.8581439170951496,
      "logloss": 0.20637071132659912
    },
    "inference_placement_steps1": {
      "AUC": 0.8372237449522196,
      "logloss": 0.23633012175559998
    },
    "inference_placement_steps3": {
      "AUC": 0.8486790838729753,
      "logloss": 0.21977686882019043
    },
    "extra_attention_off": {
      "AUC": 0.8579399714928995,
      "logloss": 0.20567069947719574
    }
  },
  "checkpoint_unchanged": true,
  "note": "subset diagnostics; within-block shuffle; blockwise loss means in toolkit differ from pooled full-validation metrics"
}
```

</details>

- **归因**：这是固定权重诊断/统计/历史裁决，不能视为新增训练或据此单独宣称结构提升。 本阶段证据解释见[阶段 9](#phase-9)。
- **后续假设**：逐一精简第二轮self-attention、cross和FFN；见K10。
- **证据**：`qformer-loop/reports/round9/r9_untied_clone/summary.json`（SHA256前缀 `4d03e250cce9`）；`qformer-loop/reports/round9/r9_untied_clone/toolkit-hooks/hook_report.json`（SHA256前缀 `b5d6ad67e0bc`）；`qformer-loop/reports/round9/r9_untied_clone/toolkit-perturbations/perturbation_report.json`（SHA256前缀 `af2ded9e9aef`）


继续阅读[阶段1–9](KUAIREC_EXPERIMENTS.md) · [阶段10–18、跨实验归因与工程补充](KUAIREC_ROUNDS10_18.md)。
