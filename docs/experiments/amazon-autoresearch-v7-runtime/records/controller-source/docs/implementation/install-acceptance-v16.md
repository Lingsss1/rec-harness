# 安装交付 v16

冻结源码 `79a31ca`，wheel SHA256 `f24d4b8b213b7fe31b6258e0ba5924b2f2102ed38086228d49a26a893f54a79b`。完整提交号在 `/home/liuzhishan/lhr/rec-harness-runs/package-install-acceptance-v16/source.json`，包在同目录的 `wheel/rec_harness-0.1.0-py3-none-any.whl`，总收据为 `delivery-acceptance.json`。

本次纳入 v15 后的优化器测量转交、真实 Amazon 开发摘要准入、转交恢复完整性、已有基线量覆盖检查，以及工具注册依据和使用说明向后继研究的传递。

从 git archive 离线构建，在两个新环境安装相同 wheel。121 个 Python 模块和原生 C 源码与冻结源码及两个安装目录逐字节一致；109 份测试/示例文件与冻结提交一致，12 个 CLI 入口检查通过。

Python 3.14.7 环境 `/tmp/rec-harness-install-verify-v16` 在仓库外执行全部 428 项测试：407 通过、21 跳过。111 个实际加载的项目模块全部来自安装目录。独立 Python 3.12 环境 `/tmp/rec-harness-install-torch-v16` 复用已有 Torch 依赖，执行 9 项 CPU 模型合同测试，全部通过、零跳过。CUDA 初始化警告不作为 GPU 运行验收；此次没有启动训练。

总收据绑定三份检查收据、执行脚本和冻结测试文件。本轮没有新增 Codex 调用、真实推荐训练/评估或最终测试，也未重跑 v15 的只读历史实例检查。环境相关跳过项仍不计入通过项；安装验收不能代替真实自主测量转交或工具效用验证。
