# v20 安装包验收

冻结源码 `7dbe6a257f308d578e0e8deccea45f3969d215ef`，从 Git archive 离线构建并安装到两个全新虚拟环境。运行归档 `package-install-acceptance-v20`，最终绑定收据 `delivery-acceptance.json`。

| 检查 | 结果 |
|---|---|
| Python 3.14 安装、导入与 CLI | 13 个入口检查通过，包括 validation_preflight |
| 源码 → wheel → 安装文件 | 127 个 Python 模块及原生 C 源码逐字节一致 |
| Python 3.12 + Torch + Linux OS 检查 | 465 项全部通过、零跳过，82.507 秒 |
| 项目模块加载来源 | 121 个实际加载模块全部来自安装目录 |
| 测试和示例夹具 | 116 个文件与冻结 Git 提交及构建源文件一致 |
| 真实联合项目轨迹 | 3 个控制器共 19 条轨迹，冻结源码与安装包输出完全一致 |
| Amazon v3 历史开发记忆 | 两次导出与原包一致，配置、study 和事件链未变化 |

wheel 为 `rec_harness-0.1.0-py3-none-any.whl`，SHA-256：

```text
8045016c79bddb04e1fd14cc25d5175ec7001ae302f7f24c9c518d02e3b92d6f
```

包含 v19 全部功能，以及显式研究 Skill 正文/版本绑定、自主与优化器测量转交的公开观测保留、已有测量拒绝门槛，以及项目验证器启动预检。

完整测试在仓库外运行，启用 `REC_HARNESS_OS_TESTS=1`。Torch 通过显式 `.pth` 使用既有 Python 3.12 环境依赖；路径、版本和文件 hash 保留在 `environment.json`。第三方依赖没有完整归档。Python 3.14 只验收安装、文件和 CLI，不记为另一轮完整测试。

历史轨迹比较只读执行，三个来源的事件链保持不变。19 条指模型回合 7 条与独立在线两组各 6 条，不含 8 次前缀调用和独立优化器调用；完整联合项目 28 次调用证据见[联合验收](../experiments/joint-project-v1.md)。本次安装验收没有新增 Codex 调用、真实推荐训练或最终测试，不能由回归通过推导工具自主缺口分支或三个循环共同收益。

最终收据绑定 wheel、源码提交、安装回归、Amazon 记忆与联合历史收据、环境、脚本和夹具。构建、审计和运行脚本均保留在运行目录。
