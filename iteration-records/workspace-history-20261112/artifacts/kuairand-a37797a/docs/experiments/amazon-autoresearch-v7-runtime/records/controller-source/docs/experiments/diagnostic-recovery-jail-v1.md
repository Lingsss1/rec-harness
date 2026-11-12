# 实际 CPU 诊断提交中断恢复

源码 `8130886`；运行目录 `/home/liuzhishan/lhr/rec-harness-runs/diagnostic-recovery-jail-v1`。`config.json` 保存完整源码提交和工具/worker 快照；`acceptance.json` 保存结果、成本与事件身份；`receipt-manifest.json` 绑定脚本、配置、测量记录、CPU 请求和进程收据等 12 份文件。

两个合成多文件工具作业通过真实 CPU launcher 执行，均成功退出。随后在 `diagnostic_completed` 提交前注入 KeyboardInterrupt，保留已执行作业和费用。恢复期间将 `run_cpu_job` 替换为立即报错的断言，证明恢复没有请求新执行。

| 情况 | 中断后的处理 | 恢复结果 |
|---|---|---|
| positive，输入 2、5 | 保留原 CPU 收据 | spread=3，复用收据 |
| null，输入 2、2 | 将收据移为 `receipt.before-interruption.json` 留档 | spread=0，根据成功进程收据重新收集；新收据标记 completed_process_collection |

每个测量目录都仅有 `job-1`，重复读取不追加事件。两项总费用仍为 2，预算上限 3；事件数 12、图版本 4，链头 `5edff2f80019e269bd7492cff5805abc3fb3d0abe196a6f0ce0f0822253002a8`。重新收集前后的进程收据完全相同。

同源码全量回归 433 项：412 通过、21 项环境相关跳过，日志 `/tmp/diagnostic-recovery-full.log`。本次真实执行与回归没有新增 Codex 调用或推荐训练，不构成自主测量缺口发现、工具晋升或科学效用证据。
