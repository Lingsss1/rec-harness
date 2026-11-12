# 安装交付 v8

冻结源码 `1cdb6ee`，交付目录 `/home/liuzhishan/lhr/rec-harness-runs/package-install-acceptance-v8/`。`delivery-acceptance.json` 绑定安装、回归与真实隔离回放证据。

| 检查 | 结果 |
|---|---|
| wheel | `wheel/rec_harness-0.1.0-py3-none-any.whl` |
| SHA256 | `3e66304288aeb4c0b77fe761decddb37a610a19a88233bc834487b64190f30d1` |
| 文件身份 | 109 个 Python 模块及原生 C 源码，冻结源码/wheel/安装目录字节一致 |
| 入口 | 12 个 CLI/模块入口检查通过 |
| 安装后回归 | 79 项，75 通过、4 项 OS 环境相关跳过 |
| 真实回放 | 两轮 Amazon 历史转换及两次独立隔离策略，world 和报告与源码相同 |

核心包使用离线构建，在仓库外的全新 Python 3.12 环境安装，清除 PYTHONPATH 并断言实际导入安装目录。覆盖新增的顺序回放、调用次数/节奏报告、开发记忆去重、评审证据摘要、按需确认对照及恢复逻辑。版本号仍为 0.1.0，按源码提交与 wheel hash 区分，更新时须 force-reinstall。

真实回放 world 为 `c38348ae0890e23a095e33ab1530840236fcf045a1ee4df11a0f81ff1b803079`。安装包重复转换一致，两个隔离策略运行均与源码结果相等，前缀无未来结果，完整 world 不挂载，预算不足拒绝第二轮，源账本保持不变。新增 Codex、GPU 训练和模型推理均为零。

该包不打包 PyTorch、CUDA、Codex 登录状态、模型数据或系统隔离环境。真实策略回放复用已经核验的本地隔离运行环境；它不能替代四项跳过的 OS 检查。本轮没有重跑模型评分、迁移旧评分解释器或访问最终测试集，也没有取得新的推荐效果证据。

```bash
python -m pip install --no-index --no-deps --force-reinstall /path/to/rec_harness-0.1.0-py3-none-any.whl
rec-harness amazon-replay-chain --help
rec-harness amazon-call-report --help
```
