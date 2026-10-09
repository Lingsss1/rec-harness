# Amazon 开发反馈的真实读出兼容性

源码 `de98c80`；验收目录 `/home/liuzhishan/lhr/rec-harness-runs/amazon-feedback-readonly-v1`，脚本为 `run.py`，结果为 `acceptance.json` 和 `feedback.json`。

输入来自已完成的 `amazon-autoresearch-next-v3/research/rounds/round-1/development-readout.json`。在独立夹具账本调用新 `commit`/`project` 函数，保留 8 行、seed 42 的完整开发读出。没有向原研究回填新事件，也没有新建实际在线研究。

验证结果：

- 重复提交不追加事件；事件截止点之前显示尚未提交，之后导出原读出。
- 另一回合没有反馈；附加含确认哨兵的 auto_round_completed 不改变开发投影，输出不含哨兵。
- 将一行 seed 改为原协议的确认 seed 并重新计算 hash，仍被开发分区校验拒绝，独立账本不变。
- 原研究配置、锁、状态、读出、study 配置的文件 hash 和控制器事件完整性保持不变。

收据绑定脚本、反馈文件、原始文件身份及原事件链。新增 Codex 调用和训练任务均为 0。此项补充真实数据格式与分区兼容性证据，不构成新的在线控制器验收、真实自主测量转交或系统效用验证；封存研究未重新优化。
