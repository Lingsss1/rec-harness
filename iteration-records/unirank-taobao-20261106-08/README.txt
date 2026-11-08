UniRank Taobao：48小时历史迭代记录（2026-11-06至2026-11-08）

这是已完成实验的冻结归档。原实验为四任务训练，但 checkpoint、父模型、课题收益与最终选模均以点击验证AUC为主；不是按四任务平均AUC开展的搜索。后续多任务选模规则修改不属于本归档。

范围：356轮研究、1386次训练、413次Agent调用；4个GPU课题，每完成6轮决策。模型gpt-6.1-sol/high；全部训练仅一个seed20262027。
初始参照DIN；允许整网重写。行业参考为UniRank中的15种对比方法。使用论文官方离线历史反馈口径，长度100。具体偏差与假设保留在protocol/protocol.json。

独立测试共2907238条曝光；选模在测试前冻结。DIN点击AUC0.652227；验证集选定版本点击AUC0.666318。选定版本四任务平均AUC0.782178，低于DIN的0.807348，其他三个反馈任务退步。该结果支持本次点击指标改善，不能解释为整体多任务提升或充分验证的新统一架构。
同轮完整候选测试点击AUC0.666459，仅作为预先冻结的对照展示，没有据此重新选模。最终验证重放和测试在原48h之外补做并单列成本。

文件入口：
  iterations.csv / iterations.json：每次训练的父模型、课题、角色、四任务验证指标、成本与代码路径。
  topics/gpu-N/state.json：原始完整轨迹、课题变化和决策。
  topics/gpu-N/rounds/rXXXX/propose/proposal.json：问题、假设、相近工作、方案、对照、知识采用/拒绝依据。
  topics/gpu-N/trials/TRIAL/model.py、model-config.json、result.json、changes-from-parent.patch：实际执行代码、配置、结果及父子源码差异。
  各轮runtime/task/workspace导出与knowledge-access/usage：当时上下文及知识访问证据。
  calls.* / token-events.json / event-summary.json：调用、Token与原始CLI流的来源SHA；输入包含缓存，不能再次相加。
  evidence/：DIN、成本校准及共享内存故障/修复记录。
  final-assessment/：最终冻结、验证重放、独立测试、四任务取舍和补测成本。
  source-at-freeze/：当时运行器、训练器、任务和通用Agent指令；不是当前主线程代码。
  selected-model/：按原点击验证指标选定的代码和配置，不含权重。

不含原始样本、模型权重、认证文件、虚拟环境或第三方论文全文缓存。没有这些资产时本目录用于审计和重建，不可直接复现已训练权重的推理。原始绝对路径作为来源证据保留，迁移时需重新配置。
copy-sources.json记录原样复制文件；derived-sources.json记录派生差异和事件摘要；manifest.json可逐文件核验。
本归档不会启动训练、调用LLM或重新打开测试集。
