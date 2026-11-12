# 本地语义种子检索

通用 Research 和 Amazon autoresearch 均可配置本地 Qwen3 嵌入模型。语义相似度决定种子，随后保留反证配额、角色优先级和两跳关系展开。未配置的旧任务继续使用 `role-lexical-two-hop-v2`，不能称为语义检索；配置后报告方法为 `role-neural-two-hop-v1`。

先在运行控制器的 Python 环境安装可选依赖 `pip install '.[semantic]'`，准备完整的本地 Qwen3 embedding 模型目录，然后向原研究配置增加：

```json
{
  "semantic_encoder": {
    "model_path": "${EMBEDDING_MODEL_DIR}",
    "query": "查找影响长程序列推荐的机制、失败解释及对照证据",
    "instruction": "Retrieve research evidence relevant to the query.",
    "max_length": 2048,
    "batch_size": 4,
    "threads": 2
  }
}
```

`model_path` 相对配置文件解析，必须是已有本地目录。加载仅用 safetensors，禁用远程代码和下载，固定 CPU float32、最后一个非 padding token 池化及 L2 归一化。当前实现支持单文件权重的 Qwen3 embedding 模型，不承诺其他编码器兼容。框架仍直接调用 Codex 执行原有角色，检索不增加 Codex 调用。

控制器冻结所有模型文件摘要、编码实现、依赖版本、查询和运行参数。外层项目、工具/Harness 对照套件在创建子研究前固定同一身份；文件或运行环境变化拒绝继续。`semantic_retrieval` 为控制器生成字段，不能手填。大型模型保留在本地依赖目录，不复制进每轮 CAS；恢复时仍须提供原文件。

通用检索先按任务、协议、图版本和已揭示依赖过滤，再将允许的节点编码。收据绑定每个输入节点的内容、查询、余弦分数及编码器；历史前缀重放使用当时的收据，不重新编码当前图。Amazon 对开发记忆编码，先校验开发读数的协议与 seed；选中设计后补齐对应读数，即使读数超出检索窗口，也保留引用闭包。失败记录和丢弃决策优先保留，但不因此升级为机制反证。

收据保存在 campaign 的 `semantic-retrieval` 目录，恢复相同输入复用原始分数和计时；模型和收据仍校验。图/平面对照在复制分支前冻结首次共同排序，以免两次独立计时改变首轮信息。后续轨迹允许分化。收据记录推理墙钟/CPU 时间、输入数量及截断项，未将 CPU 时间折算进原 GPU/角色 BCU；文件哈希核验开销也不包含在推理计时内。不能据此声称已比较全部运行成本或证明语义检索效用。

验收目录 `rec-harness-runs/semantic-retrieval-v1`：

- 七项边界测试覆盖可见范围、反证保留、历史回放、配置冻结、模型/依赖变更、持久缓存、Amazon 读数闭包及非开发 seed 拒绝。
- Qwen3-Embedding-0.6B 实际离线推理对相关/无关描述分别给出 0.600/0.290；这只是加载和排序验收。
- `toy-campaign-v2` 用真实编码器和 CPU 训练走完六角色，生成六份收据，选中 `attempt-1`；推理约 77.25 墙钟秒、157.35 CPU 秒，终态恢复拦截执行后结果与账本一致。角色为脚本夹具，新增 Codex 调用为零，不是在线研究效用实验。
- 首次小例子暴露缓存路径变量复用错误，修复后重跑。失败目录 `toy-campaign` 与 `accept-failed.log` 保留，不覆盖历史成本或把失败记作成功。

源码入口：`semantic.py`、`semantic_encoder.py`、`retrieval.py`、`adapters/amazon_semantic.py`。验收不改写正在使用旧冻结内核的 Amazon v5 历史。
