# 安装交付 v17

冻结源码 `b9a4c0d`，wheel SHA256 `390027768ba5b2427a21997ba58c3cebfd088b71336eed6841ee9dee7f55007b`。包位于 `/home/liuzhishan/lhr/rec-harness-runs/package-install-acceptance-v17/wheel/rec_harness-0.1.0-py3-none-any.whl`，完整源码提交见同目录 `source.json`，总收据为 `delivery-acceptance.json`。

本次纳入 v16 后的工具生成接口、多文件诊断 worker、诊断提交中断恢复、角色指令身份，以及通用和 Amazon 开发反馈轨迹。候选和在线效用的真实验收记录也随冻结源码保留。

从 git archive 离线构建，在两个新环境安装相同 wheel。123 个 Python 模块及原生 C 源码在冻结源码、wheel、两个安装目录中逐字节一致；112 份测试/示例文件与提交一致。Python 3.14 环境的 12 个 CLI 入口和包文件检查通过，没有在该环境重复完整测试。

Python 3.12.14 环境 `/tmp/rec-harness-install-torch-v17` 在仓库外执行完整测试：**439 项全部通过，零跳过**，约 80.061 秒。启用 Linux OS 检查，复用已有 Torch 2.7.0+cu126 依赖；117 个实际加载的项目模块均来自该安装目录。依赖路径、版本和用于接入依赖的 pth 文件 hash 记录在 `environment.json`，不宣称对整套第三方环境做了完整归档。

总收据绑定环境、文件/入口检查、完整测试收据、脚本和测试文件身份。测试覆盖此前需要 Torch、进程可见性和显式 OS 权限的用例。本轮无真实 Codex 调用、推荐训练或最终测试；安装回归不能替代真实自主测量转交或未验证的科学效用范围。
