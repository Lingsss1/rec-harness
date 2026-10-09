# 真实优化器选择合同验收

冻结源码 `6bb5c172e553a983aa8b20315039baeffd8a19ca`。运行目录 `/home/liuzhishan/lhr/rec-harness-runs/harness-router-codex-v1`；`acceptance.json` 保存结果，`receipt-manifest.json` 绑定配置、源码身份、请求/响应/schema/资源收据与事件导出。

仅新增一次隔离 Codex `optimizer_router` 调用。输入为此前已使用的公开历史轨迹、测试基底策略及派生测量合同，含 4 条具备有效公开基线的可引用轨迹、6 个共享字段和两个分支的 payload schema。没有指定必须返回测量缺口。

角色选择 `harness_patch`，仅修改 evaluation 提示词：区分主机报告的有效性与独立源码合同检查；源码在本次调用不可见时标记未知，不把这一点单独解释成测量缺口。主机通过外层选择 schema、内层 HarnessPatch schema、引用及编辑范围检查，合并后保留已有配置。

提案进入 CAS，图生成 6 个节点、6 条关系。禁止代理调用后重复 propose/sync，结果、7 条事件、图版本和成本均不变；事件头 `53fe13378ad1a897893cfe747a2487ce7379ba79f00db5982f98a370cec8312d`。

资源收据约 76.66 秒，账本 1 BCU；CLI 报告 input_tokens 177000（cached_input_tokens 142080），output_tokens 1077。只记录角色层调用数，不推断底层 API 请求数或服务计费。没有运行验证器、工具生成、模型训练或晋升。

本次证明真实选择合同及补丁分支可运行；不证明真实测量转交正例或工具效用。角色没有提出测量缺口，因此没有重试或提示它改成“正确答案”。测量转交分支目前的端到端证据仍为脚本角色集成测试。
