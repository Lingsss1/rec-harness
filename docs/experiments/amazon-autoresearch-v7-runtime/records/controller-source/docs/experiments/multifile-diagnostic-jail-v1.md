# 多文件诊断工具的实际隔离执行验收

冻结源码 `0ff3af5`，运行目录 `/home/liuzhishan/lhr/rec-harness-runs/multifile-diagnostic-jail-v1`。`config.json` 绑定完整提交号、源码快照和 worker；`acceptance.json` 绑定运行脚本、配置、两个测量记录及其 CPU 收据 hash。

实际调用 `freeze_validation_tool` → `measure` → `run_cpu_job` → `cpu_launcher` → `diagnostic_worker`，没有替换执行器。使用既有 CPU runtime 配方，在 chroot/降权后的进程中加载 `diagnostic.py`，由它导入同目录的 `helper.py`，并使用带延迟注解的 dataclass。入口检查 UID 非零、对 `/inputs/tool/helper.py` 的写入得到 PermissionError。

| 合成输入 quality | 预期范围 | 实际输出 | 进程状态 |
|---|---:|---:|---|
| 2、5 | 3 | 3 | success，退出码 0 |
| 2、2 | 0 | 0 | success，退出码 0 |

单进程均约 1.717 秒；包含 staging 和收集的作业分别约 8.035、7.676 秒。主机收集结果并写入测量事件和图。每个结果随后在禁止执行器被调用的条件下再次读取，返回完全一致，事件链不变。预算上限 3、实际扣账 2，只有两条 `diagnostic_completed`。最终事件 10、图版本 4、链头 `86dcad301f6acc28b8698a7a5412e45821a153e4c16bcc105bca212bb3ee2bb5`。

这是合成工程验收，未进行工具晋升；不能当作自主缺口发现、真实决策效用或模型效果证据。新增 Codex 调用和训练任务均为 0，最终测试继续暂缓。
