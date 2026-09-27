# 安装包验收 v7

交付记录：`package-install-acceptance-v7/delivery-acceptance.json`。构建源码为 `dc7077c`，通过离线 wheel 构建、全新 Python 3.12 环境安装和仓库外调用，包含开发配对执行、评审接续、失败停止重试及真实周期回放入口。

| 项目 | 证据 |
|---|---|
| wheel | `rec_harness-0.1.0-py3-none-any.whl` |
| SHA256 | `6e9bb90f5abb440b650a0d751945ca158646d3549ab029d861081c9fa228620b` |
| 文件一致性 | 106 个 Python 模块及原生 C 源码在冻结源码、wheel、安装目录间逐字节一致 |
| CLI | 10 个入口在仓库外通过帮助和导入检查 |
| 安装后回归 | 63 项：59 通过、4 项需专用 OS 条件的测试跳过 |
| 实际隔离验收 | 安装包运行两次独立策略回放，共 8 个本地隔离进程，报告与源码运行一致 |

实际回放使用已完成的 Amazon 输入嵌入偏移周期。安装包生成的 world 与源码版本完全相同，hash 为 `7385666c099bc44f36a97d12727e2355415dbcb9fd0adf6dddfa5502da3408e7`；七个角色和一个知识周期审计通过，导出的开发记忆与原版本一致。回放只传递已揭示的前缀，完整 world 未挂载。源控制器和 study 事件头不变，11 次历史 CLI 调用只是被核验，没有新调用。

59 项通过的回归中，3 项使用显式外部 PyTorch 解释器加载安装包，验证 CPU 线程预热、随机状态保持、源码路径约束及网络/进程创建仍被禁止。普通回归中跳过的 4 项完整 OS 测试仍记为跳过，不能以实际回放替代全部攻击测试。

核心包不打包 PyTorch、CUDA、系统运行时、模型数据或 Codex 登录状态，真实训练与隔离运行仍需按运行指南准备。原冻结评分的解释器与源码路径约束继续有效，本次 Python 3.12 安装检查不迁移旧评分运行时，也没有重新执行模型评分。新增 GPU、Codex、模型推理和最终测试调用均为零。

安装本地 wheel：

```bash
python -m pip install --no-index --no-deps --force-reinstall /path/to/rec_harness-0.1.0-py3-none-any.whl
rec-harness amazon-review-development --help
rec-harness amazon-replay-world --help
```

当前开发包版本号仍为 0.1.0，以提交和 wheel SHA256 区分构建；重新安装使用 force-reinstall，避免同版本号的旧包被保留。该验收证明当前包可安装并执行已覆盖的功能，不代表完整项目目标完成。
