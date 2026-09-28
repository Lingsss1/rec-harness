# v25 安装交付验收

冻结源码：`a1eacdb4750cf4725fa800fa9e38607894baa6ae`。包含v24之后的工具方法卡及来源图、已有/失败工具历史输入、验证条件排除预检、通用模型归档恢复和Amazon回合终止恢复。

交付目录：`/home/liuzhishan/lhr/rec-harness-runs/package-install-acceptance-v25`。

- wheel：`wheel/rec_harness-0.1.0-py3-none-any.whl`
- SHA256：`5a9c6aa0cca91c0e2d02822db6cc3ae000eb0e24f604003bae8e5db1dcc883a4`
- 138个Python模块和原生C源码在冻结源码、wheel及安装目录之间逐字节一致。
- Python 3.14独立环境的14个命令入口检查通过。
- Python 3.12安装环境562项完整测试全部通过、零跳过，147.916秒，包含Torch和Linux隔离。
- 132个实际加载的rec_harness模块全部来自安装目录，135个测试及示例文件与冻结源码一致。

使用干净工作树的git archive构建。验证环境为 `/tmp/rec-harness-install-verify-v25` 和 `/tmp/rec-harness-install-torch-v25`；后者通过.pth显式复用已有第三方Torch库。PYTHONPATH仅包含新安装目录，不包含仓库源码。delivery-acceptance.json绑定源码提交、wheel摘要、环境、测试日志、脚本和夹具清单。

版本号仍为0.1.0；已有环境升级须强制重装：

```bash
python -m pip install --force-reinstall --no-deps /home/liuzhishan/lhr/rec-harness-runs/package-install-acceptance-v25/wheel/rec_harness-0.1.0-py3-none-any.whl
rec-harness --help
```

本次没有新增Codex调用或Amazon训练，也没有重跑历史科学实验。安装验收证明当前实现已进入可安装产物，不证明新的工具效用、论文正文核验或完整方案已全部完成。最终测试口径仍按用户要求暂缓。
