# 工具方法卡合同

工具计划可显式冻结 `"tool_method_contract": "diagnostic-method-v1"`。主机在实际tool角色请求中加入合同；作者须在model/tool-method.json写入方法卡，并将文件列入changed_files。未启用的旧计划保持原合同，历史工具不会被补造方法卡。

卡片字段：schema、route、measurement、definition、inputs、assumptions、complexity、failure_modes、positive_control、null_control、independent_reference、sources、predecessor_implementation_hash。definition须说明测量量或公式；assumptions和failure_modes为非空字符串列表，其他定义与对照字段为非空字符串。

三条路线分别检查：

- original：明确自主测量定义和独立参考；前驱字段为null。
- paper：至少一个来源，每个来源含locator和method_mapping，后者说明论文公式/章节与实现定义的对应；前驱字段为null。
- existing_tool：前驱摘要必须等于实际继承源码快照的摘要。

方法卡独立保存为候选proposal_ref，通过candidate_implemented事件与候选绑定；不混入可执行工具快照。因此现有validator的tool_files不需要添加此元数据文件。缺卡、未声明文件、定义字段不全或前驱不符，在启动验证阶梯前拒绝。生成成本和已有档案保留。

通过独立验证并晋升后，诊断注册信息包含完整方法卡、摘要和“作者声明”标记，随现有tool_registration进入后继角色上下文。候选投影改换方法卡引用会被完整性核验拒绝。

这是结构与身份准入，不证明论文正文已核验、公式实现正确、对照有效或工具有科学效用。六阶段独立验证仍不可跳过。论文查询继续由Codex工具执行；本次没有新增查询或研究调用。

新增测试覆盖三条路线的必要字段、错误前驱、缺卡时验证器零执行，以及脚本工具作者→固定报告晋升→注册摘要和篡改拒绝。固定报告只是工程夹具，不作为工具效用证据。变更晚于v24安装包。

相关模块49项回归通过（56.657秒）；补充缺卡拒绝测试后，方法卡模块3项全部通过。两次测试有重叠，不相加为独立测试数量。公开边界及diff检查通过，未重跑完整安装验收。
