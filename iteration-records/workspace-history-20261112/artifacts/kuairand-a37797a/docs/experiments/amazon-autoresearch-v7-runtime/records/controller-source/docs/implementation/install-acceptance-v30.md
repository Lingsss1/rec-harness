# v30 安装交付验收

冻结源码：`c8af717dc324b18ba1e32196cd055fe6dcc1b933`。包含 v29 的 Amazon Harness 三阶段接入及新的[紧凑实验文件清单](amazon-job-index.md)。

交付目录：`/home/liuzhishan/lhr/rec-harness-runs/package-install-acceptance-v30`。

- wheel：`wheel/rec_harness-0.1.0-py3-none-any.whl`
- SHA256：`ff0642e10d28d859c6989198bfabd068c0c60d6b0d49807bfa979961d35b58b2`
- 144 个 Python 模块及原生 C 源码在冻结源码、wheel、安装目录间逐字节一致。
- 20 个命令入口检查通过。
- 安装环境完整回归：639 项全过，零跳过，180.910 秒，含 Torch、Linux 隔离、新旧归档及完整恢复检查。
- 138 个实际加载模块来自安装目录；156 个测试及示例文件与冻结源码一致。

从干净工作树的 git archive 构建，在两个独立环境安装；Torch 依赖通过显式 .pth 复用，PYTHONPATH 仅含安装目录。`delivery-acceptance.json` 绑定构建和审计脚本、日志、源码提交、wheel 与夹具清单。

```bash
python -m pip install --force-reinstall --no-deps /home/liuzhishan/lhr/rec-harness-runs/package-install-acceptance-v30/wheel/rec_harness-0.1.0-py3-none-any.whl
rec-harness --help
```

本次安装验证没有真实 Codex 调用或 Amazon 训练。旧运行的图和冻结内核未迁移，新包不会自动缩小旧数据库；真实三组工具研究仍单独验收。模型源码示例属于源码 checkout，wheel 不包含 examples。
