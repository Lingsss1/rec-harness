# v18 安装包验收

冻结源码 `9eee9a1e130133b9423800fb6aaec9070a81a768`，从 Git archive 离线构建 wheel，在两个全新虚拟环境安装。运行归档为 `package-install-acceptance-v18`，最终绑定收据为 `delivery-acceptance.json`。

| 检查 | 结果 |
|---|---|
| Python 3.14 安装、导入与 CLI | 12 个入口检查通过 |
| 源码 → wheel → 安装文件 | 125 个 Python 模块及原生 C 源码逐字节一致 |
| Python 3.12 + Torch + Linux OS 检查 | 447 项测试全部通过，零跳过，83.249 秒 |
| 项目模块加载来源 | 119 个实际加载模块全部来自安装目录 |
| 测试和示例夹具 | 113 个文件与冻结 Git 提交及构建源文件一致 |

wheel 为 `rec_harness-0.1.0-py3-none-any.whl`，SHA-256：

```text
56eaadfb9ba6f1ba202594afc2fee37649009a43c2c80778fb93cf28478f5dcc
```

本次包包含 v17 后的项目工具配置预检、论文搜索轨迹、工具搜索策略冻结、生成日志/收据归档、未准入工作区保留，以及 Amazon 失败事件归档和优化器失败轨迹。

全套测试在仓库外运行，`REC_HARNESS_OS_TESTS=1`；没有以仓库源码替代被测安装包。Torch 2.7.0+cu126 通过显式 `.pth` 使用既有 Python 3.12 运行环境依赖，路径、版本和 `.pth` hash 记入 `environment.json`。这不是完全独立的第三方依赖归档。Python 3.14 环境只做文件与 CLI 检查，没有将其记为另一轮全套测试。

验收脚本和日志保留在运行目录。`delivery-acceptance.json` 绑定源码提交、wheel、安装/测试收据、环境、脚本及夹具 hash。没有新增真实 Codex 调用、推荐模型训练或最终测试；通过回归不代表新搜索/草稿传输已经完成新的真实 Codex 验收，也不证明三循环具有正收益。
