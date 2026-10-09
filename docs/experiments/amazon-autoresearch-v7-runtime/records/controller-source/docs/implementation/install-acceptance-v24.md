# v24 安装交付验收

冻结源码：`173233a1075b65f8b7db413221351c29bda9a9ef`。本次包含案例、Evolution阶段、项目入口、在线比较组间、Harness费用前缀以及前缀角色调用的恢复变更。

交付目录：`/home/liuzhishan/lhr/rec-harness-runs/package-install-acceptance-v24`。

- wheel：`wheel/rec_harness-0.1.0-py3-none-any.whl`
- SHA256：`1b83e0d023a0baefa23cdb264de3b6d3c6c1e177afe3d048ae39bda676564018`
- 135个Python模块和原生C源码在冻结源码、wheel和安装目录之间逐字节一致。
- Python 3.14独立环境的14个命令入口检查通过。
- Python 3.12安装环境537项完整测试全部通过、零跳过，143.587秒，包含Torch和Linux隔离。
- 129个实际加载的rec_harness模块全部来自安装目录，包含四个新增恢复模块；131个测试及示例文件与冻结源码一致。

构建使用干净工作树的git archive，没有从工作区直接打包未提交文件。两个验证环境分别为 `/tmp/rec-harness-install-verify-v24` 和 `/tmp/rec-harness-install-torch-v24`。后者显式通过.pth复用已有第三方Torch运行库；PYTHONPATH只包含新安装目录，不包含项目源码。该环境依赖安排和所有脚本、日志、文件摘要记录在delivery-acceptance.json中。

同一环境内升级此前的0.1.0包，需要强制重装；版本号本次没有改变：

```bash
python -m pip install --force-reinstall --no-deps /home/liuzhishan/lhr/rec-harness-runs/package-install-acceptance-v24/wheel/rec_harness-0.1.0-py3-none-any.whl
rec-harness --help
```

本次没有新增Codex调用或Amazon训练，没有重跑历史科学实验。安装成功及工程测试不构成新的模型、工具或Harness效用证据。组内未知调用仍按恢复合同停止，不因新包安装而自动重发。
