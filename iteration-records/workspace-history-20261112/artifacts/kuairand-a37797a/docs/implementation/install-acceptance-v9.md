# 安装交付 v9

冻结源码 `fdc324617edfa04224bc634cfd2e2edc8e6ad2f7`，交付目录 `/home/liuzhishan/lhr/rec-harness-runs/package-install-acceptance-v9/`。`delivery-acceptance.json` 绑定安装身份、安装后测试和真实开发证据投影的验收文件哈希。

| 检查 | 结果 |
| --- | --- |
| wheel | `wheel/rec_harness-0.1.0-py3-none-any.whl` |
| SHA256 | `e2593c212b0d12f26ce5ea08130def54d85bc248ea9fc6d75eb8824bf1864e7a` |
| 文件身份 | 111 个 Python 模块及原生 C 源码，冻结源码/wheel/安装目录一致 |
| 入口 | 12 个 CLI/模块入口检查通过 |
| 安装后回归 | 82 项，78 通过，4 项 OS 环境检查跳过 |
| 真实证据 | 基线历史事件绑定、四类候选执行摘要和只读开发记忆导出通过 |

离线构建并安装至全新 Python 3.12 环境 `/tmp/rec-harness-install-verify-v9`。所有检查在仓库外执行，清除 PYTHONPATH，并断言导入来自安装目录。相较 v8，包含基线历史诊断及图连接、可见输入核验、候选执行协议摘要和只读开发记忆导出；回归覆盖这些功能及原有恢复、费用、回放、评分和最终评估入口。

安装后的代码读取 v3 已完成的开发证据，验证历史投影与事件/CAS 一致，生成的四类候选协议摘要与源码验收逐字段相同（投影 hash `d82cdbd2d3b3b0d985365caa40051885ef9f6275a4cad7aeb3dddb95a45c2b1a`）。v2 开发记忆重复导出与原导出一致，源事件投影不变；验收显式禁止调用 Store.export_events。没有修改在途 v3 的冻结代码或角色输入。

此次新增 Codex、训练和模型推理均为零，未读取确认或最终测试结果。v9 没有重复执行 v8 的真实隔离策略回放，也不替代四项跳过的 OS 检查。包不包含 PyTorch、CUDA、模型数据、Codex 登录或系统隔离环境。版本号仍为 0.1.0，安装时以 wheel 哈希及源码提交区分交付。

```bash
python -m pip install --no-index --no-deps --force-reinstall /path/to/rec_harness-0.1.0-py3-none-any.whl
rec-harness --help
```
