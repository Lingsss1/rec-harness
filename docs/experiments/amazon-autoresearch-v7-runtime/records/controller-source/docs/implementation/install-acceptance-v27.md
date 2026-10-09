# v27 安装交付验收

冻结源码：`db60158eeaa0e50fa5669e56fe869fa2b581e50a`。包含v25之后的方法卡默认启用、项目工具历史继承、正确性失败报告归档、独立逐轮验证计划，以及单计划旧路由兼容修复。

交付目录：`/home/liuzhishan/lhr/rec-harness-runs/package-install-acceptance-v27`。

- wheel：`wheel/rec_harness-0.1.0-py3-none-any.whl`
- SHA256：`538e9bb4859b993aadab523b973c7fc390623747d041f4893e4eb30b1ecbd0be`
- 139个Python模块和原生C源码在冻结源码、wheel与安装目录之间逐字节一致。
- Python 3.14独立环境的14个命令入口通过。
- Python 3.12安装环境575项完整测试通过，零跳过，191.257秒，包含Torch和Linux隔离。
- 133个实际加载的rec_harness模块全部来自安装目录；136个测试及示例文件与冻结源码一致。

使用干净工作树git archive构建。两个新环境为 `/tmp/rec-harness-install-verify-v27` 与 `/tmp/rec-harness-install-torch-v27`；后者显式通过.pth复用既有第三方Torch库。PYTHONPATH仅包含安装目录，不包含项目源码。delivery-acceptance.json绑定脚本、日志、源码提交、wheel、夹具清单和环境。

先前v26构建冻结05df3d4，575项测试有一项旧路由兼容错误：没有配置计划序列时仍读取tool_updates。已修复为单计划直接使用基础计划，并在v27重新完整验收。v26目录及delivery-rejected.json保留，不能视为成功交付。

v26曾另行通过安装包对真实错误工具历史的无新增执行复核，该单项结果保持其原冻结范围，不替代v27完整回归，也不把v26升级为成功交付。v27没有重新运行历史科学实验。

版本号仍为0.1.0；升级已有环境须强制重装：

```bash
python -m pip install --force-reinstall --no-deps /home/liuzhishan/lhr/rec-harness-runs/package-install-acceptance-v27/wheel/rec_harness-0.1.0-py3-none-any.whl
rec-harness --help
```

本轮没有真实Codex调用或Amazon训练。完整安装回归证明当前实现可交付，不证明新的工具在线效用或完整科学方案已经全部验收。最终测试口径仍按用户要求暂缓。
