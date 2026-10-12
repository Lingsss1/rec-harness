# 假设前证据图与四阶段研究

后续更新：新任务的 `prefix-evidence-v2` 读回策略见[证据图修复方案](evidence-graph-prefix-v2.md)和[20 轮前缀模拟](../experiments/evidence-graph-prefix-v2.md)。下文原生图限定描述对应最初的 `native-selected-v1`；旧冻结任务继续使用旧策略。v2 增加带来源标记的跨运行只读视图，不补造缺失的原生关系。

Amazon 示例配置现在显式使用 `role_layout: compact-four-v1` 和
`hypothesis_graph: true`。已有 campaign 保持冻结配置；未指定 role_layout 的
旧格式配置仍使用 `six-role-v1`，不重写历史调用或费用。

## 四次正常调用

1. `researcher`：读取证据图，一次生成 GapCard、MethodCard、ProposalCard
   及实际引用的图节点 ID。承担原 evidence/hypothesis 的连续推理。
2. `experiment`：按冻结提案实现候选及生成对照。
3. `reviewer`：独立读取基线、候选、生成对照的实际源码及哈希，核对实现是否
   符合声明、是否改错位置、训练/评分路径是否一致、对照是否实现所述干预。
   request_changes/unresolved 会停止本轮，不进入 CPU 准入和 GPU 训练；
   实质性否决不会作为格式错误自动重问同一份代码直到获批。
4. `analyst`：训练和独立复评分后，一次解释结果并提出知识图补丁。性能选择
   交给主机，机制解释保留 provisional/unknown，不另设重复的 lead 调用。

主机仍执行源码合同、CPU 数值与训练冒烟检查、训练、独立复评分及晋升门槛。
独立 AI 审稿是补充，不替代可执行检查，也不保证发现所有代码错误。
如果显式启用 measurement_discovery，按需增加一次测量评审；自动格式修正
或失败尝试也会额外计费。四次指正常模型轮次，不是隐藏额外调用的总量保证。

归档保留四个真实角色的响应、请求、schema 和费用；不伪造六角色完成记录。
`analyst` 响应内明确分开 decision 与 knowledge；恢复、周期审计和开发记忆
导出使用真实复合响应，并保留其内容哈希。

## 假设生成前的证据图

之前 Amazon hypothesis 主要收到 history/research_memory 文本条目；原生图主要
在训练后的 review 阶段出现。现在在冻结 round input 时从本地 study 图读取
匹配已选历史读出的开发事实及关系，传给设计角色的 `evidence_graph` 包含：

- 当前经审计 baseline 的派生观察。
- 原生开发 Experiment、分组 Observation、机制对比及相关解释/提案。
- TESTS、USES、TARGETS、PRODUCES、SUPPORTS、CONTRADICTS、BASED_ON、
  SUPERSEDES、ALTERNATIVE_TO 等实际存在的关系。
- 两跳检索记录、反证优先项、节点引用、修订遗漏提示、版本与源事件头。

只允许同协议、已选历史范围内的公开开发图；确认、最终测试、其他任务以及
依赖不可见证据的解释/关系均不进入该视图。节点数默认最多 64，可通过
`hypothesis_graph_limit` 冻结配置；证据闭包不能容纳时省略解释并报告数量。
来自别处的历史读出若没有本地图血缘，保留在原 history 并报告 unavailable，
不会编造图中的支持关系。

researcher 必须引用可见节点，GapCard observation_refs 也须属于该图。
输入图及引用保存在真实角色调用档案里。这证明图进入了输入且引用可追溯，
不证明模型理解正确，也不直接证明图检索改善研究收益。

## 验证边界

使用本地脚本角色和训练替身验证控制流、审稿阻断、证据可见性、对照晋升、
恢复、调用计费与记忆导出；没有启动真实 Codex/GPU 验收。

本次最终相关回归 121 项全部通过，覆盖四角色真实归档结构、调用计费、
审稿否决零候选训练、对照晋升、终态恢复、开发记忆导出、证据引用与前缀回放。
此前全量 672 项对照的失败/错误/跳过集合与修改前完全一致；本机仍有原有
环境失败，不声明全套通过。编译检查、公共目录边界检查及 diff 检查通过。

实现角色只接收提案、源码约束与对照定义，避免重复传递完整历史图；审稿者
接收源码与声明，分析者接收开发读出和知识图。旧调用的可见输入保持原样。
