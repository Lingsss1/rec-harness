# TencentGR auto / enhanced 24 小时对比

运行根目录：`output/taac-auto-vs-enhanced-20261104/`。

当前状态：准备与两种模式的真实 LLM 集成已经通过，39 项测试通过，GPU 3 的完整尺寸 5 步训练校准通过。**24 小时正式研究尚未启动，等待用户选择评测协议。**

完整数据的原协议仅产生 feedback=7、gate=6、final=7 个查询。只读审计的另一个提案 `taac-timecut-nextclick-v1` 产生 feedback=28714、gate=29047、final=28472 个查询；其输入截止于序列 80% 位置的时间点，目标为之后第一次点击，排除未知/并列目标。它改变了原任务的决策时点，不能等同于赛事的目标曝光前预测。提案尚未采用，没有生成或启动该协议的正式研究。见根目录 `protocol-decision.json`、`timecut-proposal-audit.json`。

- 仓库起点：`40b85214dc8db4a7e219852de1e3b17377cf58cb`。
- 编码模型：`gpt-6.1-sol`，`model_reasoning_effort="high"`；每次调用均保存原始 JSONL 与 token 用量。
- auto 为仓库基础循环，enhanced 为原生 `enhanced=true` 的知识检索、反例、diff、诊断及分析循环。
- 真实官方数据固定在 `TAAC2025/TencentGR-1M@4be8735e2a7e9fc11fe7eb2a5e1e06e27c57e453`，下载大约 2.65 GB 所需文件（含语义编码器），LFS 文件逐个核对发布者 SHA256。保留完整 660000 候选及最多 100 个历史事件。
- 语义编码器固定在 `intfloat/multilingual-e5-small@614241f622f53c4eeff9890bdc4f31cfecc418b3`，不退化为纯关键词检索。
- 使用相同原始基线、固定哈希用户切分、3 epochs、batch 2048、完整训练和全候选精确评分。三个配对种子 42/43/44 各自执行独立的研究轨迹，候选从头初始化。
- 服务器检查发现官方发布文件含 null 行为标签，修复了原适配器 `int(None)` 失败。训练保留 null；为避免给未知行为编造曝光/点击标签，留出评价排除含任一 null 行为的用户，并统计事件、用户和逐切分排除数。两组共用该冻结协议，实际覆盖率见 prepared/public/metadata.json。
- 官方 null 多模态向量作为缺失特征处理，保留默认零向量及 false 覆盖标记并单独计数；不把缺失计为已获得特征。维度错误或非有限的已提供数值向量仍拒绝。
- auto 分配 GPU 0/1/2，enhanced 分配 GPU 4/5/6；GPU 3 用于计时校准。
- 每个方法、每个种子最多 86400 秒；数据准备和 GPU 校准在正式计时外。研究、LLM、检索、失败、评分和最终留出评测在预算内。根据实际校准预约完整 trial 和最终评测；预算不足以容纳下一轮时提前结束，不强行启动注定超时的完整训练。
- 总预算是六个并发研究轨迹的约 24 小时墙钟窗口，不是把 24 小时分给两种方法。
- `min_delta=0.0001` 是沿用示例的预注册选择阈值，并未声称已校准为统计显著阈值。
- `max_trials=1000000` 仅满足仓库配置的正整数要求，实际由时间预算控制。
- 最终 gate/final 仅在研究关闭后由宿主执行一次；任何模式的研究 Agent 均不得读取最终分数或另一组的研究内容。执行器为仓库的可信本地模式，无 OS 级恶意代码隔离声明。
- 分数是仓库保守重建的本地留出 `0.31*HitRate@10 + 0.69*NDCG@10`，不是原赛事隐藏榜单分数。

查看进度与比较（仓库根目录）：

```bash
.venv/bin/python scripts/taac24/controller.py status
cat output/taac-auto-vs-enhanced-20261104/suite.json
cat output/taac-auto-vs-enhanced-20261104/comparison.tsv
```

准备阶段及正式计时分别记录于 `suite.json`、`budget.json`。每个运行的进度、结果、设计、源码及模型调用在 `{auto,enhanced}/seed-{42,43,44}/`。最终比较写入 `comparison.json` 和 `comparison.tsv`；只有完成的留出评价才包含 final 指标。三条配对轨迹不自动构成统计显著证据；没有账单时美元费用保留未知。

在当前轮完成后提前结束并执行最终评测：

```bash
touch output/taac-auto-vs-enhanced-20261104/STOP
```

下载进程和总控制器使用独立会话运行，关闭当前聊天不会主动终止它们。服务器重启不在自动恢复保证范围内。若准备失败，查看 `suite.log`、`download.log`、`prepare.log` 或 `calibrate-*.log`；不要把准备失败报告为正式训练结果。
