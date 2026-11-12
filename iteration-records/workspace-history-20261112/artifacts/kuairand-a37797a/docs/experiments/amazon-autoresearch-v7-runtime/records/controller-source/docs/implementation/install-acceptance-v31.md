# v31 安装交付验收

冻结源码：`081464beb33a73a19fabe7fed827a597569ed5f3`。包含 v30 的紧凑实验归档，以及 Amazon 工具套件冻结时保留虚拟环境解释器链接的修复。

交付目录：`/home/liuzhishan/lhr/rec-harness-runs/package-install-acceptance-v31`。

- wheel：`wheel/rec_harness-0.1.0-py3-none-any.whl`
- SHA256：`b95ecdc2b90253ba44121ad0dc054e42d2b661d5fc3d8962046b78bfeddb57f0`
- 144 个 Python 模块及原生 C 源码在冻结源码、wheel、安装目录间逐字节一致。
- 20 个命令入口检查通过。
- 安装环境完整回归：640 项全过，零跳过，151.061 秒，含 Torch、Linux 隔离、新旧归档及完整恢复检查。
- 138 个实际加载模块来自安装目录；161 个测试及示例文件与冻结源码一致。

从干净工作树的 git archive 构建，在两个独立环境安装；Torch 依赖通过显式 .pth 复用，PYTHONPATH 仅含安装目录。`delivery-acceptance.json` 绑定构建和审计脚本、日志、源码提交、wheel 与夹具清单。

```bash
python -m pip install --force-reinstall --no-deps /home/liuzhishan/lhr/rec-harness-runs/package-install-acceptance-v31/wheel/rec_harness-0.1.0-py3-none-any.whl
rec-harness --help
```

本次安装验证没有真实 Codex 调用或 Amazon 训练。旧运行的图和冻结内核未迁移，新包不会自动缩小旧数据库；真实三组工具研究仍单独验收。模型源码示例属于源码 checkout，wheel 不包含 examples。
