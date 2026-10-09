# Amazon 工具测量送达审计

`python3 -m rec_harness.tool_exposure CAMPAIGN OUTPUT` 现在同时识别普通 campaign 与 `amazon-auto.json` 控制器。Amazon 路径核对 auto-lock 和绑定 study 配置，从调用前 CAS 输入、此前已提交诊断与角色完成事件建立送达记录。

普通 Amazon 角色使用 `measurement_baseline` 重建诊断请求，并要求它与实际 `development_baseline` 的品类、seed、指标逐项一致。测量审查角色使用精简基线及诊断摘要，由宿主重建诊断身份，再与已提交记录投影比较。只接受冻结的开发 seed 和品类；确认 seed、study 变更、改变后的摘要或基线不能被解释为合法开发测量。`role_completed` 必须绑定对应提交的角色及 context hash，尚未完成的提交保留 `transport_completed: false`。

查询使用只读账本事务，不恢复任务或运行角色。它证明数据确实进入角色输入，不证明角色采用了数据，也不证明工具收益。此工具不审计历史长度、激活和 block-score 等其他诊断的全部来源链；本次范围是通过 diagnostic_tool 注册和执行的生成式工具。

针对性测试合计22项通过：Amazon 实际控制器角色格式、精简摘要与待完成状态、确认条件拒绝、study 修改拒绝，以及通用测量审查、成本和项目回归。训练与工具执行使用工程替身，不计为 GPU 或工具效用实验。

`rec-harness-runs/amazon-tool-exposure-acceptance-v1/acceptance.json` 保存封存 Amazon v3 的两次相同只读输出，事件头仍为 `104512ed28273d9a92276dec1149e54ccfd4ec8d265487ca35b4d8e967221ce8`。该研究未注册生成式工具，因此工具数量和送达数均为0；不将这一兼容性结果当作真实正例。未新增 Codex、训练或推理。
