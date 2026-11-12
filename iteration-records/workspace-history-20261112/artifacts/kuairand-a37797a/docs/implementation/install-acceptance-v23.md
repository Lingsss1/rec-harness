# v23 安装交付

冻结源码：`4544e152da13234f02a0e579f0c8dcf41dcd7828`。交付目录：`rec-harness-runs/package-install-acceptance-v23`；wheel 为 `wheel/rec_harness-0.1.0-py3-none-any.whl`。

SHA256：`74abb0e58b6c5718033612bb5c4ab601943b8722b5c0fa4ba48e89e1131d1db1`。

本版纳入多来源轨迹、项目启动预检、Harness正常淘汰后的模型续接，以及公开观测工具、shadow来源和在线合成输出观测。旧项目保持冻结行为，不重新打开封存研究。

验证结果：

- 从干净Git提交归档离线构建，分别安装到独立Python 3.14与3.12环境。131个Python模块和原生C源码在源码、wheel及安装目录中逐字节一致。
- 14个CLI入口检查通过。
- Python 3.12安装环境执行全部519项测试，零跳过，135.226秒，含Torch和Linux隔离检查。125个实际加载模块全部来自该wheel安装目录。
- 7项历史核验与此前源码结果一致：两个项目报表、CPU后继工具送达、Amazon封存历史、两个真实公开观测前缀，以及新的输出观测工具缓存。没有重发角色或重跑这些历史作业。

测试与示例来自同一冻结提交；交付收据逐文件核对fixture内容，并绑定wheel、脚本、日志和环境信息。Torch通过显式`.pth`复用已有第三方依赖目录，`rec_harness`导入路径单独核验，未使用源码checkout作为包入口。

首次安装测试执行了515项，另有4项Linux隔离测试因固定使用系统Python、未找到虚拟环境内的包而在类初始化阶段失败。保留初始日志；随后仅为测试子进程设置指向本次安装目录的`PYTHONPATH`，不修改代码或测试断言，重新完成519项。环境与两次日志均绑定在`delivery-acceptance.json`中。

包版本号仍为0.1.0。升级已有同版本环境时，使用实际wheel路径强制重装，避免pip跳过旧版：

```bash
python -m pip install --force-reinstall --no-deps /path/to/rec_harness-0.1.0-py3-none-any.whl
python -m rec_harness --help
```

安装包不包含源码示例、训练数据、Codex凭据或运行环境。仍需按运行指南准备Codex CLI、隔离recipe及任务配置。本次是交付与兼容性验收，没有新增Codex调用或Amazon训练，不增加工具、记忆或Harness科学效用结论；整体方案的未验收项见[当前清单](completion-matrix.md)。
