# 输入嵌入偏移候选：执行故障与恢复前置验收

`amazon-frozen-autoresearch-followup-v2` 的候选在原 item embedding 上增加独立、零初始化的 input_offset；输出评分继续使用原 embedding。四个开发任务均在初始化阶段崩溃，没有产生可比较指标，不能评价假设的推荐效果。

四份 stderr 均报告 `libgomp: Thread creation failed: Operation not permitted`。候选的 CPU 大张量零初始化需要 OpenMP 线程池；旧 GPU 启动器只预热 CUDA，在启用禁止新线程的 seccomp 规则后才首次触发该 CPU 路径。原 CPU 小模型准入没有覆盖真实 item 数量下的并行初始化。

| 类别 | 已计 GPU 槽位秒 | 状态 |
|---|---:|---|
| Movies_and_TV | 94.24957847595215 | crash |
| Industrial_and_Scientific | 86.2046582326293 | crash |
| Electronics | 103.83754838258028 | crash |
| CDs_and_Vinyl | 81.81715132296085 | crash |
| 合计 | 366.1089364141226 | 无有效训练结果 |

研究账本保留失败成本，总计 5343.161380402744 秒。控制器终态为一个失败回合，仅执行 evidence、measurement、hypothesis、experiment 四次 Codex 调用；未运行结果评审、确认训练或最终测试。外部七角色成功路径审计失败，不能把控制器写出终态文件等同于研究成功。

启动器修复在启用 TSYNC 前执行固定大小的 CPU 零初始化，预先建立线程池，不消耗随机数。没有放宽禁止网络或创建进程/线程的规则。另将训练收据的成功状态检查移到成功产物读取之前，避免缺失 correctness.json 掩盖原始训练失败。

`gpu-cpu-pool-acceptance-v1/controller-result.json` 保存真实隔离回归：同一候选源码、冻结运行库、GPU 4、65,536 个合成 item，CPU 零初始化及三步 GPU 梯度更新成功；九项边界检查通过。验收计费 58.1410795673728 GPU 槽位秒，单独记录，未计作模型研究训练。未调用 Codex、未读取真实用户数据或最终测试。该验收只证明执行修复，不证明推荐收益。

系统解释器全套 298 项测试（281 通过、17 跳过）；其中依赖 PyTorch 的新线程池回归在实际 PyTorch 解释器单独通过，验证随机状态不变、大张量初始化可用以及网络/进程创建仍被拒绝。随后增加的失败收据回归通过 Amazon 报告模块全部五项测试。

旧研究和已冻结控制器保持原样。下一次真实效果评估需要登记修复后的执行身份并保证候选与对照使用同一执行版本；不得直接把旧版本对照和新版本候选当作已经通过执行一致性审计的配对。

已启动 `amazon-input-offset-recovery-v1`：源码快照为 `499b09f`，继承原 study 的全部 5343.161380402744 秒成本，新 study 总上限 20,000 秒。原方案和候选源码保持不变并重新登记；先重跑四类别开发 seed 42 的对照与候选共八次训练，然后独立 CPU 复评分。方案没有自动释放确认阶段。启动不调用 Codex；旧优化研究和独立框架验收的成本仍单列保留。执行结果以该目录的 controller-result.json、development-readout.json 和八份复评分收据为准，启动记录不构成效果验收。
