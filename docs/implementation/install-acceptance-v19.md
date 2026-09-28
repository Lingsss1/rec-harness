# v19 安装包验收

冻结源码 `e32b428228ea20eafe94455f41c2921aad698c09`，从 Git archive 离线构建并安装到两个全新虚拟环境。运行归档为 `package-install-acceptance-v19`，最终收据为 `delivery-acceptance.json`。

| 检查 | 结果 |
|---|---|
| Python 3.14 安装、导入与 CLI | 12 个入口检查通过 |
| 源码 → wheel → 安装文件 | 126 个 Python 模块及原生 C 源码逐字节一致 |
| Python 3.12 + Torch + Linux OS 检查 | 454 项全部通过，零跳过，81.528 秒 |
| 项目模块加载来源 | 120 个实际加载模块全部来自安装目录 |
| 测试和示例夹具 | 115 个文件与冻结 Git 提交及构建源文件一致 |
| Amazon v3 历史开发记忆 | 安装包两次导出均与原包一致，配置、study 和事件链未变化 |

wheel 为 `rec_harness-0.1.0-py3-none-any.whl`，SHA-256：

```text
2f61a8f042bced0a1438118628f8625d1f4a3b25e60bc0c251b4fdd19761ec09
```

包含 v18 的全部功能，以及方法来源准入、Method→Source 图关联、通用来源记忆 v4、Amazon 来源记忆 v2 和开发记忆导出的冻结锁校验。引用仍明确标注为 Agent 报告、未经内容核验；这不构成论文适用性证明。

全套测试在仓库外运行，启用 `REC_HARNESS_OS_TESTS=1`。Torch 2.7.0+cu126 通过显式 `.pth` 使用既有 Python 3.12 环境依赖；路径、版本和文件 hash 见 `environment.json`。第三方依赖没有完整归档。Python 3.14 仅验收安装、文件和 CLI，未运行完整测试套件。

最终收据绑定源码提交、wheel、安装/测试/Amazon 只读收据、环境、脚本和夹具。没有新增真实 Codex 调用、推荐模型训练或最终测试；安装回归不证明三个研究循环都有正收益。
