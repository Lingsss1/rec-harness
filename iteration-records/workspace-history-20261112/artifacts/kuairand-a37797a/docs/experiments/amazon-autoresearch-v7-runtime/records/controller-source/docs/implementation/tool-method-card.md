# 工具方法卡合同

新建 AutonomousEvolution 工具任务和 ResearchProject 工具计划默认冻结 `"tool_method_contract": "diagnostic-method-v1"`。显式空值、false或未知版本在创建目录和调用角色前拒绝。主机在实际tool角色请求中加入合同；作者须在model/tool-method.json写入方法卡，并将文件列入changed_files。未启用的旧冻结计划保持原合同，历史工具不会被补造方法卡。项目创建时即冻结默认值；旧项目创建后续工具子任务时沿用父合同，不套用新默认值。低层 Evolution API 继续执行显式提供的计划，不修改历史调用合同。

卡片字段：schema、route、measurement、definition、inputs、assumptions、complexity、failure_modes、positive_control、null_control、independent_reference、sources、predecessor_implementation_hash。definition须说明测量量或公式；assumptions和failure_modes为非空字符串列表，其他定义与对照字段为非空字符串。

三条路线分别检查：

- original：明确自主测量定义和独立参考；前驱字段为null。
- paper：至少一个来源，每个来源含locator和method_mapping，后者说明论文公式/章节与实现定义的对应；前驱字段为null。
- existing_tool：前驱摘要必须等于实际继承源码快照的摘要。

主机在tool_method_contract请求中明确提供inherited_implementation_hash（无前驱时为null），作者无需根据本地路径或文件内容猜测快照摘要。直接Evolution调用及项目继承路径使用相同规则。方法卡独立归档，未修改的辅助模块继续按原源码快照继承。

方法卡独立保存为候选proposal_ref，通过candidate_implemented事件与候选绑定；不混入可执行工具快照。因此现有validator的tool_files不需要添加此元数据文件。缺卡、未声明文件、定义字段不全或前驱不符，在启动验证阶梯前拒绝。生成成本和已有档案保留。

通过独立验证并晋升后，诊断注册信息包含完整方法卡、摘要和“作者声明”标记，随现有tool_registration进入后继角色上下文。候选投影改换方法卡引用会被完整性核验拒绝。

恢复已实现候选时也立即核对proposal_ref与提交事件一致，避免仅在后续晋升或verify时发现篡改。新增多文件继承与恢复检查纳入46项相关回归，全部通过（39.386秒）；没有新增Codex调用。此前真实生成验收保持原冻结源码范围，不用新代码重写其历史合同。

这是结构与身份准入，不证明论文正文已核验、公式实现正确、对照有效或工具有科学效用。六阶段独立验证仍不可跳过。论文查询继续由Codex工具执行；本次没有新增查询或研究调用。

新增测试覆盖三条路线的必要字段、错误前驱、缺卡时验证器零执行，以及脚本工具作者→固定报告晋升→注册摘要和篡改拒绝。固定报告只是工程夹具，不作为工具效用证据。变更晚于v24安装包。

相关模块49项回归通过（56.657秒）；补充缺卡拒绝测试后，方法卡模块3项全部通过。两次测试有重叠，不相加为独立测试数量。公开边界及diff检查通过，未重跑完整安装验收。

## 图知识投影

候选实现事件现在把已归档卡片投影为Method节点，并以ToolCandidate→BASED_ON→Method关联。节点保留测量定义、前提、复杂度、失败条件和proposal_ref，状态为provisional；paper路线的origin为paper_claim，其余为agent_hypothesis，均不是独立实验单位。

来源投影为Source节点，通过Method→DERIVED_FROM→Source关联。相同locator合并为一个节点，各方法映射去重后保留在边上；来源状态为agent_reported_unverified、版本unknown、正文档案为空。引用或晋升不会把来源升级为已核验事实，也不会生成模型效果Claim。

37项相关测试通过（5.305秒），覆盖真实Store检索能找到方法、终态再运行不新增角色或事件、重复引用合并及未核验标签，同时检查既有模型方法来源和诊断路径。没有新增Codex调用或真实工具生成，仍晚于v24安装包。

后续[真实生成验收](../experiments/tool-method-authoring-v1.md)已完成：一次Codex调用生成工具及original方法卡，六个CPU参考案例通过，恢复无新增执行，合计10 BCU。该验收使用操作者给定的解析测量任务，不证明论文正文核验、自主缺口发现或在线效用，未晋升工具。

## 新任务默认合同

自主工具入口和项目模板现在默认要求方法卡，不再依赖调用者手动打开选项。默认值在配置冻结前确定，不修改传入的配置对象。显式非法合同在调用角色前拒绝。旧控制器恢复不注入默认值；项目子任务沿用父项目已冻结的计划，因此旧项目尚未执行的工具阶段也不会被悄然升级。

相关62项工程回归通过（42.739秒），覆盖工具生成、六阶段调度、注册、历史输入、工具继承和模型续接。脚本作者现在显式提供方法卡；固定验证报告仍只证明工程流程，不代表新工具效用。本变更晚于v25安装包。

完整源码回归564项全部通过、零跳过，155.999秒，包含Torch和Linux隔离。tool-method-default-v1保存日志、源码清单及补丁摘要。没有新增Codex调用或Amazon训练。
