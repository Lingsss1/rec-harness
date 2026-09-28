# 新工具生成接口的真实 Codex 验收

冻结源码 `922399b`，目录 `/home/liuzhishan/lhr/rec-harness-runs/tool-authoring-codex-v1`。复用此前操作者定义的公开聚合测量缺口及独立参考套件，进行一次真实隔离 Codex tool 调用；没有把它伪装成自主发现缺口。

实际接受请求包含新 `diagnostic_contract`，入口为 `diagnostic.py:measure(request)`；完整 prompt/schema 的指令身份重新计算通过，原请求/schema/响应/资源收据保存在 CAS。生成实现 hash 为 `e72ec5ed45d39d5fb44a5af9a6e93fa496cd60c615d5dba0664af1a065b17389`。

正确性、合成对照和稳健性三个独立验证阶段全部通过，每阶段两个真实 CPU 隔离案例。候选停留在 offline_validated；没有执行 shadow、utility、cost_regression，没有晋升或启动推荐训练。

Codex 仅提交一次，无重试，传输耗时约 102.442 秒。控制器账本扣账 4（生成 1、三个阶段声明费用各 1）；三个独立参考账本各记 2。两层账本分列，不能将这些声明单位解释成 GPU 秒或货币费用。

禁止调用 agent 后重复 propose，源码与事件/费用不变；禁止启动 validator 后重复三阶段验证，也无新增过程或费用。控制器事件 15、图版本 0，链头 `8f8980e839ddfb191112dfc846ba4eacef45376e705b29dd86e4572fc2d89d0e`。本验收未调用完整 Evolution.run，因此不声称完成终态生命周期投影。

`acceptance.json`、`validation-resume.json` 和 `receipt-manifest.json` 保存检查与 14 份文件的绑定。首版恢复审计脚本误把 validate 返回的整个接受记录与内部 report 比较而断言失败，原脚本留在 `audit-first.py`；修正比较对象后通过，没有重新生成或验证。`acceptance.json` 阶段字段 `report_hash` 是整个接受记录的 hash；补充收据 `stage_hashes` 明确分别保存 accepted_record_hash 与 validator_report_hash，避免混淆。

这次验收补充真实生成接口、指令档案和离线参考执行证据，不证明自主测量转交、真实决策效用、机制解释或模型提升。

后续[影子验收](tool-authoring-shadow-v1.md)已完成两个真实开发条件的四次 CPU 测量，候选进入 shadow_validated。本文收据仍表示离线检查点；没有工具晋升或在线效用结论。
