# v35 安装交付验收

冻结源码 `75c6cd30c8ce8a07db4d19e6b922a5318b238f6c`，包含本地语义检索、通用/Amazon 接入、外层身份冻结和持久收据恢复。构建前工作树干净，git archive 固定源码、测试和示例，离线构建后安装到两个独立环境。

wheel SHA-256：`e6bf6759a67faf7ed1ae32f37eb62fb792e82e0c658773dc31301603315fd74b`。

- 147 个 Python 模块及原生 C 源码在源码、wheel 和安装目录间逐字节一致。
- 20 个命令入口检查通过。
- 完整安装回归 653 项全部通过、零跳过，169.456 秒，包含主机进程身份及 Linux 隔离检查。
- 141 个实际加载模块均来自安装目录，162 份测试/示例文件与冻结源码一致。
- 安装后的编码器成功复用小例子的六份真实神经检索收据，分数和计时不变，没有导入 Torch/Transformers 或重新推理。

证据目录 `rec-harness-runs/package-install-acceptance-v35`：`delivery-acceptance.json` 汇总源码、wheel、脚本和测试摘要；`installed-tests.log` 与 `full-stdout.log` 保留完整输出。实际检索收据恢复证据为 `rec-harness-runs/semantic-retrieval-v1/installed-cache-acceptance.json`。

文件/入口环境为 `/tmp/rec-harness-install-verify-v35`，完整测试环境为 `/tmp/rec-harness-install-torch-v35`。Torch 第三方依赖通过显式 .pth 复用，项目模块必须来自新 wheel。新增测试不调用真实 Codex 或启动 Amazon 训练；真实 v5 历史仍绑定其原内核。工程验收不等于语义检索、工具或 Harness 的科学收益已被证明。
