# Amazon 三组在线工具对照入口

`python -m rec_harness.amazon_tool_online` 为无工具、现有工具、候选工具分别创建独立 study 和 AmazonAutoresearch 控制器，执行真实基线训练、独立CPU复算、研究回合和主机评分。正式路径直接使用隔离Codex，不注入选择结果。可独立运行或作为显式验证命令使用；[六阶段tool_pipeline](amazon-tool-pipeline.md)现已按所配置online套件类型选择此入口。内置预检已识别其套件、输入合同与开发/历史条件排除。

## 配置与运行

复制 [配置示例](../../examples/amazon-tool-online/spec.json) 和 [控制器配置](../../examples/amazon-tool-online/controller.json)，替换模型、已准备的Codex/GPU/CPU运行环境、源study及UID登记路径。示例费用权重仅演示格式，实际实验须在冻结前明确。源study只提供协议、模型源码、worker与数据身份，不导入旧任务、checkpoint、图、研究结论或已花费用，也不修改原study。

每个条件可同时显式指定development_seeds和confirmation_seeds，为新study建立内部开发/确认分区。外层stage与这个内部种子分区不同：外层开发/确认各包含完整研究任务。同一数据集、数据哈希和训练seed不能跨条件重复；不同seed仍可能使用同一验证用户，不算独立用户样本或最终测试。

```bash
python -m rec_harness.amazon_tool_online freeze /path/to/spec.json /path/to/frozen-suite
python -m rec_harness.amazon_tool_online run --suite /path/to/frozen-suite/suite.json --candidate /path/to/candidate-tool --stage utility --output /path/to/development/report.json
python -m rec_harness.amazon_tool_online run --suite /path/to/frozen-suite/suite.json --candidate /path/to/candidate-tool --stage confirmation --output /path/to/confirmation/report.json
```

两阶段使用不同输出目录。冻结绑定所有包源码、模型和worker CAS、私有训练/验证文件哈希、模型服务设置、候选文件集合、费用权重和研究条件。数据仍在主机端，不发送给角色；原始数据和模型结果的出站边界沿用Amazon控制器。

每个条件有3个研究组；每组每轮通常6次Codex角色调用，因此单条件单轮约18次、两阶段各一个条件约36次。角色格式错误可能消耗既有控制器合同允许的额外调用，仍受冻结role_call_limit约束；额度不足或失败回合不产生通过的效用报告。工具作者调用不包含在上述数量内。基线、候选和有条件触发的内部确认训练均按各组账本记录。

## 评分与恢复

每组使用相同模型、协议、种子及各轴额度，分别运行基线及研究。主机检查研究完成事件、incumbent、账本、训练收据及开发成绩一致，并核对三组评估候选集合和数据哈希相同。报告取终态incumbent的开发HR@10宏平均，分别计算候选工具相对两组对照的收益，主gain取两者较小值；不事后挑最高候选。工具组必须实际完成诊断且无诊断失败。

报告保留role_calls、gpu_seconds、diagnostic_units三个费用轴，total_cost仅为它们按冻结权重折算的和。启用激活采集时还有显式activation_units轴。CPU准入/独立复算没有现成计费轴，其wall time不计入该数；它不是全部硬件开销。passed表示执行合同有效，正收益仍须由外层冻结门槛判断。报告类型是end_to_end_candidate_generation，不是固定候选选择。

已提交completed.json的组恢复时重新核对证据，不重新运行；started存在但未提交完成结果的组一律停止，不能因中断自动重启训练或角色。当前不支持组内自动续跑，包括完成后、完成标记写入前中断的情况。

当前控制器配置支持基础研究、聚合诊断及[激活采集](amazon-online-activation.md)，历史记忆和block-score采集选项尚未接入。六阶段pipeline费用审计现已适配Amazon费用轴；见上述接入说明。不能把该入口的完成当成自主缺口发现、所有诊断合同在线效用或三循环整体完成。

## 验收

10项新增测试覆盖新study与独立账本、条件去重、来源不变、源码/数据变化拒绝、三组收益及费用、私有组条件匹配、正式控制器初始化/基线顺序、训练收据复算、未知结果不重发、已完成组复核和内置预检。与Amazon及配置预检相关57项测试通过，5.389秒；训练、角色及部分审计采用夹具，没有新增真实模型效用结果。

真实Amazon v3的公开源码、原协议、私有数据哈希和配置另完成两阶段只读冻结/预检，指定新的种子条件；原研究及study事件头保持不变。未读取或复制旧确认成绩，未启动Codex、CPU/GPU实验或改动最终测试。证据位于rec-harness-runs/amazon-tool-online-v1。变更晚于v27安装包，此时尚未进行完整真实运行；后续pipeline接入及其工程测试见上述说明。
