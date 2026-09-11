这是 q4_difference 的独立失败定位目录。config-template 中模型、零梯度延后 helper、完整 configs、参数数目与 route 配置均为原 v2 的逐字节副本，SHA 记录在 source-identities.json。

reproduce_psgd_failure.py 默认仅校验身份；--run 才执行 GPU 0、seed 20262027、batch 8192 的原版两步 smoke。每次创建新的 attempt-UTC 目录，全部运行产物和编译缓存写入该目录。--reuse-cache 先独立复制原编译缓存再使用，并在结束时验证原缓存未变，不直接把原路径作为可写缓存。

observations.jsonl 记录 A 的 backward 原始梯度、clip 后延后规则前梯度、延后规则后梯度、B 参数以及失败时 precond_init_scale 的真实 locals 和输入统计。统计包括严格零、torch.allclose 默认零判据、FP32/FP64 四次矩与范数。保持原优化器参数和原首步零梯度跳过规则，不补偿、不强制初始化 scale、不继续正式训练或质量评估。

原 runner 的 smoke preflight 会读取 train/valid 指纹并构造 valid 元数据，但不会迭代 validation 或运行其预测。result.json 明确区分 smoke 尝试步数、完整步骤数、dense 成功步骤数和正式训练更新数 0。失败步骤可能已经部分修改隔离模型的 optimizer state，不会写回原模型或运行目录。
