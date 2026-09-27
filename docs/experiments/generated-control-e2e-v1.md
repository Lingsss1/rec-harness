# 自动生成机制对照的端到端验收

按用户最新要求，优先完整实现 autoresearch，最终测试与词表修正暂缓。未提交的词表改动已备份至 `rec-harness-runs/deferred-vocabulary-work-v1/`，未加入当前执行代码；真实最终测试的失败封存保留。

验收目录为 `/home/liuzhishan/lhr/rec-harness-runs/generated-control-e2e-v1/`。使用固定 RNG 生成的 32 个用户、最多 512 个物品的合成训练/验证数据，一个类别、开发 seed 42、条件确认 seeds 17/83。冻结既有 NextItNet incumbent 源码，训练协议为 2 epochs、每 epoch 2 batches；这是工程集成条件，不是推荐效果评价。

配置 `generated_control_count: 1`，由真实 Codex hypothesis/experiment 产生候选及机制对照，使用同一次 experiment 调用生成 `model.py` 和 `control-1.py`。后续经源码准入、隔离 CPU 数值/训练检查、GPU 开发训练、独立 CPU 评分、evaluation/lead/knowledge 及周期归档。正常轮次六次角色调用，调用上限 10；仅 hypothesis 开启 Codex 内置检索。没有向角色提供真实用户或最终测试结果。

GPU 4/5 在启动前确认空闲。控制器 PID 为 4026245，身份保存在 `launch.json`；源码冻结身份位于 `build-identity.json`。当前已启动，尚未完成验收。通过标准包括：真实生成两个不同源码版本、各自准入与评分、完整开发比较及角色周期、所有生成文件的事件/归档绑定，以及终态恢复无重复调用或训练。若开发门槛未通过，正确淘汰也可证明该开发闭环完成，但不证明确认分支或模型收益。

终态独立审计脚本及监视器已准备并启动（PID 4030618）。脚本核对三个不同源码身份、生成文件的完整绑定、全部开发对照、条件确认收据及精确恢复，明确禁止新增 Codex、训练和推理。当前冻结执行代码保持原样。后续实现进一步要求生成文件哈希表完整：删除整个表或删除单个对照条目均在角色复用和审计时拒绝；相关回归及全套 348 项测试通过（331 通过、17 跳过），这不代替正在运行的真实验收。
