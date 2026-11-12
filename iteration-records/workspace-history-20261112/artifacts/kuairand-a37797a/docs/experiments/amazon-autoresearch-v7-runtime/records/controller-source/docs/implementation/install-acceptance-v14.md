# 安装交付 v14

冻结源码 `9ae42094cb6d1e8caba4d4cb2f9c0a12715bf248`，wheel SHA256 为 `43f9b5b672e0a4816246f068d79bd834b470d97409d24dde308843ba23d10b48`。包位于 `/home/liuzhishan/lhr/rec-harness-runs/package-install-acceptance-v14/wheel/rec_harness-0.1.0-py3-none-any.whl`，总验收收据为同目录上一级的 `delivery-acceptance.json`。

本次纳入 v13 后的传输停止与显式恢复、Amazon 进度及历史成本边界、跨轮修订证据检索、Evolution 生成收据恢复、工具/Harness 生命周期图，以及 Harness 提案归档和失败假设关联。

从 git archive 构建，使用本地构建依赖，无依赖下载。119 个 Python 模块及原生 C 源码在冻结源码、wheel、全新安装目录中逐字节一致；106 份测试和示例文件与冻结提交一致。12 个既有入口和 4 个新增入口启动检查通过，新增覆盖 `amazon-progress`、`evolution-sync-graph`、`retry-transport`、`autoresearch`。

Python 3.14.7 的全新环境 `/tmp/rec-harness-install-verify-v14` 在仓库外运行全部 406 项测试：385 通过、21 项环境相关跳过；109 个实际加载的项目模块全部来自安装目录。现有 Torch 是 Python 3.12 构建，不能加载到 3.14；独立建立 `/tmp/rec-harness-install-torch-v14`，安装同一 wheel，并复用本机 Python 3.12.14/Torch 2.7 的依赖目录，9 项 CPU 模型合同测试全部通过、零跳过。第二环境的项目模块与 wheel 也逐字节核对。此验证不声称 Torch 兼容 Python 3.14。

安装后的真实历史检查：v4 两轮和 v3 一轮的进度报告可重复读取，分数及调用成本符合已提交历史，无法证明的旧 GPU 成本边界仍返回缺失；原事件和状态哈希不变。已结束且被拒绝的真实工具演化记录，在独立图副本生成 8 个节点、7 条边，不生成晋升或模型 Claim，原研究不变。

五份验收收据、脚本及测试/示例身份由总收据绑定。本轮没有新增 Codex、训练、推荐评估或最终测试任务。安装回归不替代被跳过的 OS 隔离检查，也不增加图、工具或 Harness 的研究效用证据；最终测试口径继续暂缓。
