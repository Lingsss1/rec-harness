# v29 安装交付验收

冻结源码：`def2baeeee11d0be5a7b2b72f85231801d3f1863`。包含 Amazon Harness 独立配对入口、前缀合同及三阶段 pipeline、分轴累计费用与恢复支持。

交付目录：`/home/liuzhishan/lhr/rec-harness-runs/package-install-acceptance-v29`。

- wheel：`wheel/rec_harness-0.1.0-py3-none-any.whl`
- SHA256：`38eaaa60d1255c0bca62bd35d4b92ffa6f933d85032043e83eadce1b77d197a8`
- 144 个 Python 模块及原生 C 源码与冻结源码、wheel 和安装目录逐字节一致。
- 20 个命令入口通过，包括 Amazon Harness、前缀和 pipeline。
- 安装环境完整回归：637 项全部通过，零跳过，158.843 秒，包含 Torch 与 Linux 隔离测试。
- 138 个实际加载模块全部来自安装目录；156 个测试及示例文件与冻结源码一致。

使用干净工作树 git archive 构建，在 `/tmp/rec-harness-install-verify-v29` 和 `/tmp/rec-harness-install-torch-v29` 新环境安装。后者通过显式 .pth 复用第三方 Torch；PYTHONPATH 只包含安装目录。`delivery-acceptance.json` 绑定脚本、日志、源码提交、wheel 和夹具清单。

```bash
python -m pip install --force-reinstall --no-deps /home/liuzhishan/lhr/rec-harness-runs/package-install-acceptance-v29/wheel/rec_harness-0.1.0-py3-none-any.whl
rec-harness --help
```

安装回归没有真实 Codex 调用或 Amazon 训练。同期两次 [Amazon 前缀真实调用](../experiments/amazon-harness-prefix-live-v1.md) 单独记账。该安装验收不证明策略效用，也不消除运行中的 Amazon 工具研究发现的平铺归档性能问题。
