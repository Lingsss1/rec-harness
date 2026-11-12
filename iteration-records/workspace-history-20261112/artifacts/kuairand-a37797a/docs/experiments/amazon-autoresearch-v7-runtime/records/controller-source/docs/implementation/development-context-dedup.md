# 开发研究上下文去重

后续 campaign 构造上下文时，同一开发读出只在 `history` 出现一次；`research_memory.development_hash` 引用其哈希。持久化 memory pack 的格式及冻结任务不变。设计、决策、知识和失败记录全部保留。

检索窗口截断后，仍补入所选研究记录引用的读出，避免悬空引用。所有读出校验协议、自身哈希和开发 seed；缺失或不一致的引用拒绝生成上下文。确认集数据不能通过该入口混入开发结果。

离线验收使用 `amazon-autoresearch-next-v1/research/rounds/round-1/input.json`：7 条记录、6 份唯一读出，逐条还原与原记录相等，原文件未改动。规范 JSON 的 UTF-8 字节从 191189 减为 160661，减少 30528 字节（约 16%）。这不是 token 或计费测量，也不减少角色调用次数。验收未调用 Codex、训练或推理。

验收记录：`/home/liuzhishan/lhr/rec-harness-runs/context-dedup-acceptance-v1/acceptance.json`。回归覆盖无损还原、原始输入不变、重复读出、跨检索窗口引用、缺失引用、协议/哈希变更、非开发 seed 及控制器隐私上下文。
