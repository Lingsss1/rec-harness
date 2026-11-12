# 安装交付 v15

冻结源码 `311f56e50d3bc215ba6cf8f5773e1ed678806348`，wheel SHA256 `46a84353e840043ce7fd07e553f33735b19f629db4984e36c86f15d29d24621c`。包位于 `/home/liuzhishan/lhr/rec-harness-runs/package-install-acceptance-v15/wheel/rec_harness-0.1.0-py3-none-any.whl`；总收据是其上级目录的 `delivery-acceptance.json`。

本次纳入 v14 后的缺口复测血缘、工具源码继承与多文件保留、连续 Harness 策略合并、三循环联合恢复、无损共享轨迹和历史前缀修订提示修复。

离线从 git archive 构建，无依赖下载。120 个 Python 模块及原生 C 源码在冻结源码、wheel、安装目录中逐字节一致；107 份测试/示例文件与提交一致。16 个入口启动检查通过。

Python 3.14.7 新环境 `/tmp/rec-harness-install-verify-v15` 在仓库外执行全部 415 项测试：394 通过，21 项环境相关跳过；110 个实际加载的项目模块均来自安装目录。独立 Python 3.12.14 环境 `/tmp/rec-harness-install-torch-v15` 安装同一 wheel，复用现有 Torch 依赖，9 项 CPU 模型合同全部通过，零跳过；该环境项目文件也与 wheel 一致。

安装后的只读历史验收覆盖：两份真实 Codex 增量提案的生命周期图精确重建、共享轨迹完整还原、4 个真实前缀案例重建、两个 Amazon 历史研究共 3 行进度报告，以及实际拒绝工具的 8 节点/7 关系独立图导入。源事件和历史文件保持不变。

六份验收收据、脚本及测试文件身份由总收据绑定。本轮没有新增 Codex、推荐训练/评估或最终测试任务；CPU 合同探针属于测试覆盖。此安装验收不替代环境跳过项，也不增加工具、图或 Harness 的科学效用证据。最终测试口径继续暂缓。
