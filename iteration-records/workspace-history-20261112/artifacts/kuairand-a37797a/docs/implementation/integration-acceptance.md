# 训练环境与系统隔离组合验收

2026-09-23，本地训练虚拟环境 Python 3.12 执行全部79项测试，全部通过，无跳过。命令：

```bash
REC_HARNESS_OS_TESTS=1 /home/liuzhishan/lhr/RecHarness/.venv-run/bin/python \
  -m unittest discover -s tests -p 'test_*.py'
```

该运行需要有权限启动非特权chroot沙箱的Linux控制器。回放策略的最小运行时由 `/usr/bin/python3 -m rec_harness.policy_sandbox prepare` 构建；控制器继续在训练虚拟环境中运行。此前测试直接从虚拟环境构建运行时被正确拒绝；测试现已按部署方式分开这两种解释器，未放宽沙箱要求。

此次实际执行了PyTorch模型/采样/梯度检查、进程身份与孤儿进程恢复、回放策略的宿主文件/环境/网络/派生进程阻断、超时与非法JSON、双次重放一致性，以及其余运行时合同测试。独立训练环境中的默认运行是74项通过、5项系统条件跳过；系统默认Python运行会额外跳过PyTorch测试，不能替代本次组合验收。

这份证据不证明任意生成模型的GPU代码隔离，不证明新Codex候选、诊断工具或Harness的在线收益，也不替代真实任务的最终测试。模型训练调度故障路径和Harness晋升阶梯仍含明确标注的测试替身；真实36份checkpoint和诊断证据另见amazon-progress.md。完整方案的未完成项保留在completion-matrix.md。
