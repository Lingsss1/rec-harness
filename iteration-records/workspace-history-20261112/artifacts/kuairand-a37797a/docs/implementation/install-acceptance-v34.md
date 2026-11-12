# v34 安装交付验收

冻结源码 `ba7c44b1740609854150c04fefbf3cf0a731f87b`，包含测量审查前的基线证据范围检查。构建前工作树干净，使用 git archive 固定构建源码、测试与示例。离线构建和两个独立虚拟环境安装成功。

wheel SHA-256：`4446961340290050e800510166ddda9cc3c8c58e22080c65bd549571e1c34883`。

- 144 个 Python 模块与原生 C 源码在源码、wheel 和安装目录间逐字节一致。
- 20 个命令入口检查通过。
- 安装环境完整回归 646 项通过、零跳过，166.762 秒，包括主机进程身份与 Linux 隔离检查。
- 138 个实际加载模块均来自安装目录；161 份测试与示例文件与冻结源码一致。

证据目录：`rec-harness-runs/package-install-acceptance-v34`。`delivery-acceptance.json` 汇总源码、wheel、测试和脚本摘要；`installed-tests.log` 保存全部测试结果，`full-stdout.log` 保存控制台输出。

文件与入口检查使用 `/tmp/rec-harness-install-verify-v34`，完整测试使用 `/tmp/rec-harness-install-torch-v34`。Torch 第三方运行时通过显式 .pth 复用；项目模块必须来自新 wheel。测试不调用真实 Codex 或启动 Amazon 训练。运行中的真实 v5 仍使用原冻结内核，不因本次交付改变历史或在途实验；科学效用与整体方案完成度单独验收。
