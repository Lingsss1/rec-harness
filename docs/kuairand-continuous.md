# KuaiRand 持续研究与图谱对照

本次确认的运行合同是：GPU 0–3 先完成指定提交的四模型基线，然后运行有图谱组；GPU 4–7 运行无图谱组。两组均采用完整方法 D 的自适应阶段版本 `D-adaptive-v1`，共享数据、环境、可用信息、完整训练和宿主评分边界。研究仅改变图谱是否可读。无图谱组保留按时间排列的方案、实验、源码与复盘，不接收图节点、边、检索结果或图谱派生摘要。它不是保留关系信息的 `memory_presentation=flat` 对照。

用户已授权不设候选轮数、费用或运行时间上限，按需自主检索、实施对照、修复可恢复故障；无法维持协议或权限时停止并保留原因。也已授权阶段性冻结验证 Pareto 前沿上的候选后执行宿主独立最终测试，最终结果不回流研究。

## 数据与模型合同

- 原始基线来自独立 worktree `artifacts/kuairand-a37797a`，提交 `a37797a3767e622641ef48a52a6276307a4126b3`。原工作区及既有实验不改写。
- 使用现有公共环境 `artifacts/RecHarness-6lyc/.venv`；不要求官方包版本。两组冻结实际 Python、PyTorch、NumPy、pandas 版本和同一协议哈希。
- 正式训练固定 seed 42、101 epochs、历史 1024、维度 128、全局 batch 128、128 个负例及 FP32；每组最多两个双 GPU 训练。OOM 不触发降序列、降 batch 或换精度。
- 原始基线保留提交的 leave-last-out 和末轮评测。四模型 CUDA/DDP 冒烟及完整尺寸前后向均须通过，然后准备一次正式数据并训练。合成验收在 `acceptance/`、`baseline/preflight/`，不得进入正式结果表。
- 两组研究各自从头重训四个研究起点。研究缓存从原始处理 CSV 单独建立，保留 1027 个末尾事件。倒数第二个事件作验证、最后一个作最终测试，训练排除两者。
- 保留参考实现允许的 next-event 时间信息，包括用于对应评分的目标时间；这属于冻结的信息合同。CCFormer 历史子空间只消费历史时间。不能在候选间悄悄改变时间可用性。
- 候选定义 `build(config)`，返回支持 `encode_all(ids,times)` 与 `score_queries(context,ids,rows,positions,candidates)` 的 encoder。DARE、CCFormer 保留候选条件计算，不要求两塔。
- 宿主采样、计算损失、穷举评分并控制配置。已注册 AdamW/Adam/SGD、constant/cosine、sampled softmax/BPR。训练处理变量另存 `training.json`；新增注册能力需另行受检，不能让候选修改宿主。

## 完整方法 D 的自适应阶段

保留[原方法 D](../staged-autoresearch/EXPERIMENT_DESIGN.md)的事前条件方案、证据复盘、新上下文和按需归因。按本次用户要求取消固定 10 次阶段、固定第 8/10 次复盘和最多两次诊断：

1. LLM 申请 `requested_rounds`，给出相同数量的未来步骤，自己选择 `review_after`，包括阶段末尾。每步给出竞争解释、条件、干预、预期观测及必要对照。这里一轮是一个受控实验组，实际完整训练次数为该组变体数量。
2. 实现角色根据冻结方案和当时已知的观测实施条件分支。候选及全部对照由独立审稿者按源码与哈希审核；宿主执行有限值、梯度、因果性、物理前缀、用户独立性及候选分块检查。
3. 单因素用配对、替换或容量匹配；组合使用 none/only_a/only_b/both，并追加交互差值。共享同名同形状的未训练参数，固定 sampler 随机顺序；新参数与结构特有的 dropout 消耗仍是混杂并记录。
4. 每次复盘用新的临时 Codex 上下文。诊断必须针对会改变方向决策的歧义，指定精确父版本和不同的支持/反证后续决策。诊断次数由 LLM 决定；诊断后再次复盘，允许继续申请。切换方向时只取消尚未执行的步骤，不伪造已完成实验。
5. 实验事实追加到 SQLite 事件图与 CAS。解释保留 `provisional`，单种子不晋升为稳定规律；修订使用新节点和 SUPERSEDES，旧结论与反证保留。实现拒绝、失败、零收益及负收益均保留，不当作机制已被证伪。

正常调用使用 `gpt-6.1-sol/high`。检索源内容、版本 URL、时间、正文哈希、请求定位和实现位置一起归档；获取内容与验证论文主张分开。未获取或定位未独立核验时明确保留状态。

## 成本与最终测试

前沿比较全量验证 NDCG@10（高）、完整训练成功 epochs 的 GPU 秒（低）和固定前 128 位验证用户面板的全目录评分时间（低）。后者用相同 batch/chunk，完整预热一遍后测三遍取中位数，包含统一评分路径的有限值检查，是本机成本，不代表生产延迟。

失败、中断、重试、检查与 checkpoint 等额外占用由每次 job 的 allocated GPU seconds 单独保留；不能与成功 epochs 的训练时间重复相加。Codex 每次 request、response、原始 CLI 日志和 input/output/cache tokens 单独保留；没有美元收据时费用未知，不填零。

阶段结束时只根据验证前沿冻结源码、训练配置、checkpoint 和验证结果哈希。最终评分由单独进程写入 `host-only-final/`，不写入研究 records、图谱事实、后续角色输入或选择规则。每候选最终测试只执行一次；状态不明不自动重复。最终评分错误也只进入该目录，不作为研究反馈。

当前执行后端是受信进程模式：源码表面检查、Codex 权限和输入白名单是实际措施，**不宣称已建立 OS 级文件读取隔离或对任意恶意 Python 的安全证明**。最终测试隔离指独立进程、独立结果目录及无反馈的数据流边界。模型/训练角色不得主动读取别组、测试结果或数据文件。

角色输入头部仅展开最近 64 个实验、四个起点与当前前沿，以及最近 16 个方案/复盘；完整普通历史另外冻结为该次调用的 `evidence/chronological-history.json`，两组均可按需读取。没有删除旧记录，也没有用图谱摘要替代无图谱历史。已完成调用恢复时复用原输入和当时的图快照，不用后来的图改写旧决策。

## 运维

准备及持续运行的统一入口：

```bash
python3 -m rec_harness.kuairand_suite launch --root artifacts/kuairand-research
python3 -m rec_harness.kuairand_suite status --root artifacts/kuairand-research
python3 -m rec_harness.kuairand_suite stop --root artifacts/kuairand-research
python3 -m rec_harness.kuairand_compare artifacts/kuairand-research
```

首次运行可用 `--replace-preparation` 接管本会话此前的准备进程，保留可能被重叠写入的未校验分片。范围下载仅发布通过官方 MD5 的完整归档；服务器不支持范围请求则回退提交原有下载器。数据/输出锁禁止重复准备与调度。

单组恢复和状态：

```bash
python3 -m rec_harness autoresearch artifacts/kuairand-research/no-graph/campaign
python3 -m rec_harness status artifacts/kuairand-research/graph/campaign
python3 -m rec_harness.kuairand_control stop artifacts/kuairand-research/graph/campaign
python3 -m rec_harness.kuairand_control resume artifacts/kuairand-research/graph/campaign
```

全局停止不会杀掉活动训练，阻止下一任务启动；训练在完整 epoch 上保存恢复身份和 RNG。完成的实验不重训，完成的角色绑定原上下文/响应哈希复用。协调器可自动重试已观察到的瞬时 Codex 连接/限流/服务错误，保留原调用与费用；连续三次仍失败视为服务阻断。审稿否决转为实现失败证据，供后续方案修复。

基线报告将由原始 `scripts/summarize.py --require-complete` 在四模型实际训练全部完成后生成。`suite.json`、`baseline/controller.json` 及各组 `state.json` 是当前状态；不把正在运行写成已有正式结果。

比较报表只读取验证事实、前沿和调用/训练成本，排除最终测试结果；两组请求的实验次数可能不同，需要沿成本轴比较，不直接归因于图谱，也不以同 seed 当作独立重复。

## 2026-10-21 宿主性能检查

用户提出同卡并行后，先在空闲卡测量，原正式训练持续运行。检查目录为 `artifacts/kuairand-research/performance-probes/`，结果不进入正式指标或研究上下文。

- 在 64 位用户的全目录评测面板上，将候选分块从 256 增至 8192，HSTU 用时中位数从 0.838 秒降至 0.228 秒，ULTRA 从 1.118 秒降至 0.526 秒；该面板聚合指标一致。这不是对所有用户逐项排名一致性的证明。
- 完整尺寸合成前后向中，关闭 block checkpointing 只减少约 2%–3% 用时，显存增至约 11–16 GB，因此少占显存并不意味着重算是主要耗时来源。
- `kuairand_collocation_probe` 在同一 GPU 6–7 上测 HSTU、DARE 两套双卡任务，保留 global batch 128、FP32、完整长度及原代码路径，比较各自独占与同时共卡的等量前后向工作。该检查不修改任何正式训练断点或配置。
- 各运行 3 个完整尺寸合成训练步，独占串行总计 43.015 秒，共卡总计 40.621 秒，聚合吞吐约增加 5.9%；HSTU 单步中位数从 7.28 秒增至 13.70 秒，DARE 从 7.10 秒增至 13.45 秒。共卡显存约 8.7 GB/卡。该小样本仅覆盖训练前后向，不涵盖完整 epoch 的评测重叠；未据此改变正式调度。

随后用户明确要求“并行把另外两个模型一起跑了”，覆盖原基线最多两个任务的调度约束。已通过独立 sidecar 启动四模型共卡：GPU 0–1 上 HSTU + DARE，GPU 2–3 上 ULTRA-HSTU + CCFormer。各模型仍双卡、global batch 128、FP32、seed 42、101 epochs；原始提交源码和正在运行的两项任务未修改或重启。GPU 4–7 的研究组分配保持原约定。

sidecar 用带所有者标记的 `baseline/STOP` 取消原协调器的后续队列，活动训练继续。它等待原输出锁释放和四个固定末轮结果齐备，再调用原汇总器并发布完成状态，因此原协调器不会重复启动新增模型，研究也不会提前放行。该标记是内部交接，不代表用户停止整个研究。

```bash
python3 -m rec_harness.kuairand_collocate status
```

状态及归档在 `baseline/co-location.json`、`co-location.log` 和各模型原有 `jobs/`、checkpoint 目录。汇总附带 `summary-scheduling.json`，明确同卡竞争影响：每任务 rank 用时含共享等待，多个任务的 GPU 预留区间重叠，不能声称是独占 GPU 成本。若新增任务失败，保留断点和费用，回退原协调器恢复缺失任务。
