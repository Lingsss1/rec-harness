# v28 安装交付验收

冻结源码：`5f06d643363202e86fbfa8dcecc0213f84b5caae`。包含 Amazon 三组在线工具验证、六阶段 pipeline、激活采集与费用审计、固定候选选择、工具淘汰续接及 v27 之后的其他实现。

交付目录：`/home/liuzhishan/lhr/rec-harness-runs/package-install-acceptance-v28`。

- wheel：`wheel/rec_harness-0.1.0-py3-none-any.whl`
- SHA256：`e58bd6b59b476689a545b7595fb677fef19bd9a27eac42931549903c6b076cdf`
- 142 个 Python 模块及原生 C 源码在冻结源码、wheel、安装目录之间逐字节一致。
- 17 个命令入口检查通过，包含新增的 Amazon 在线工具、费用审计和固定候选选择入口。
- Python 3.12 安装环境完整回归：622 项全部通过，零跳过，154.853 秒，包含 Torch 和 Linux 隔离测试。
- 136 个实际加载模块全部来自安装目录，151 个测试及示例文件与冻结源码一致。

从干净工作树的 git archive 构建，在 `/tmp/rec-harness-install-verify-v28` 与 `/tmp/rec-harness-install-torch-v28` 两个新环境安装。Torch 环境通过显式 .pth 复用既有第三方依赖；测试的 PYTHONPATH 仅含安装目录。`delivery-acceptance.json` 绑定构建、检查和回归脚本、日志、源码提交、wheel 与夹具清单。

版本号仍为 0.1.0，已有环境升级需强制重装：

```bash
python -m pip install --force-reinstall --no-deps /home/liuzhishan/lhr/rec-harness-runs/package-install-acceptance-v28/wheel/rec_harness-0.1.0-py3-none-any.whl
rec-harness --help
```

安装回归没有新增真实 Codex 调用或 Amazon 训练。同期运行的 [Amazon 工具 v2](../experiments/amazon-tool-live-v2.md) 单独记录实际实验；本报告不将工程测试计为在线效用，也不表示所有研究假设已获支持。最终测试口径仍按用户要求暂缓。
