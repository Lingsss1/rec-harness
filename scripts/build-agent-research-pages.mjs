import fs from "node:fs";
import path from "node:path";

const outDir = path.resolve("public/lessons/agent-research-guides");
fs.mkdirSync(outDir, { recursive: true });

const methods = [
  {
    slug:"aide", no:"01", lane:"Agent-as-MLE", tags:["Code Tree Search","Stateless Memory"], title:"AIDE", subtitle:"AI-Driven Exploration in the Space of Code", paper:"https://arxiv.org/abs/2502.13138", code:"https://github.com/WecoAI/aideml", year:"2025",
    intro:"把机器学习工程重写成“在代码空间里优化一个可执行解”的问题。它不维护一段无限增长的对话，而是维护解树：每个节点是一份脚本，每条边是一次 Draft、Debug 或 Improve。",
    idea:["搜索对象不是自然语言计划，而是可运行 Python 程序；执行结果直接成为下一轮决策证据。","Improve 被限制为一次原子修改，使性能差异更容易归因。","选择策略优先高分有效节点，同时保留修复 buggy 节点的入口，避免搜索完全坍缩到单一路径。"],
    formula:"v* = argmax(v ∈ V_t) F(v)\n\nchild = f_operator(parent, Σ(T))\nΣ(T) = compress({score, hyperparameter, traceback}_history)",
    math:"AIDE 是黑盒、梯度自由的程序优化。F(v) 可取交叉验证分数；策略用当前观测到的最高 fitness 做贪心利用。由于 LLM 变换无法微分，搜索质量取决于代码算子是否产生局部、可测量的邻域。",
    memory:{write:"每次执行后写入代码、验证指标、运行状态、traceback 与关键超参。",read:"Σ(T) 只抽取分数、超参和调试线索，注入 Draft/Improve 提示；节点本身保持自包含。",forget:"不是删除原始节点，而是通过摘要压缩控制 prompt；代价是跨分支的结构性经验可能丢失。"},
    schema:"SolutionNode {\n  code, parent_id, operator,\n  status: valid | buggy,\n  validation_score, runtime, traceback\n}\nPromptMemory = summarize(solution_tree)",
    metrics:[["16.9%","MLE-Bench Any Medal（AIDE + o1-preview）"],["82.8%","有效提交率"],["原子变更","每次 Improve 只做一个可测修改"]],
    limits:["贪心选择容易围绕局部最优反复打磨。","全局摘要会混入与当前分支无关的信息，也可能造成模式坍缩。","fitness 依赖验证集，若验证设计不稳，树搜索只会更快地过拟合代理指标。"],
    transfer:"RecHarness 可直接借用“artifact-first”思想：每个 trial 必须对应 commit/config/log/metric，而不是只存聊天。把 Improve 约束为单一假设，可显著提高线上归因能力。"
  },
  {
    slug:"ml-master", no:"02", lane:"Agent-as-MLE", tags:["MCTS","Selective Memory"], title:"ML-Master", subtitle:"探索与推理通过选择性作用域 Memory 耦合", paper:"https://arxiv.org/abs/2506.16499", code:"https://sjtu-sai-agents.github.io/ML-Master/", year:"2025",
    intro:"AIDE 证明了代码树搜索有效，但“把整棵树摘要给模型”仍然可能造成上下文污染。ML-Master 用 MCTS 式多轨探索，并只把父节点与同深度兄弟节点的经验送入推理过程。",
    idea:["UCT 在高回报与低访问次数之间平衡，避免所有算力只投向当前冠军。","Draft / Debug / Improve 对应不同程序状态；停滞次数和连续 Debug 深度构成剪枝规则。","Memory 进入 reasoning/think 上下文，父节点保证连续性，兄弟节点提供对比信号与去重线索。"],
    formula:"UCT(v) = Q_v / N_v + C · sqrt( ln N_parent / N_v )\n\nc_t = R_θ^think(x, M_t) ; r_t = ε(c_t)\nM_t = {(r_parent, f_parent)} ∪ {(r_sibling, f_sibling)}",
    math:"第一项是经验平均收益，第二项是探索奖励。N_v 小的节点拥有更高不确定性奖金；C 控制探索强度。停滞剪枝可写成 Σ_i I[Δ_i < threshold] > τ_improve，从而停止对无产出的分支继续投资。",
    memory:{write:"保存执行指标、编译/运行日志、错误诊断、代码片段，并从完整推理中抽取 salient insight。",read:"只读直接父节点和同层兄弟节点；父节点提供纵向因果链，兄弟节点提供横向反例。",forget:"用作用域选择代替全历史检索。更老祖先只通过父节点的已蒸馏结论间接保留。"},
    schema:"AdaptiveMemory(t) {\n  lineage: [parent.insight, parent.feedback],\n  contrast: siblings.map([insight, feedback]),\n  excludes: unrelated_depths + raw_long_logs\n}",
    metrics:[["29.3%","完整 MLE-Bench 平均奖牌率"],["12 h","单任务预算"],["20.2%","中等难度任务奖牌率"]],
    limits:["论文版本未给出完整消融，难以单独量化 memory、并行与骨干模型的贡献。","同层兄弟并不天然相关；分支定义不合理时，作用域仍会带来噪声。","UCT 的标量回报无法表达新颖性、稳定性、资源成本等多目标权衡。"],
    transfer:"适合把 RecHarness Memory 拆成 lineage memory 与 sibling memory：前者用于延续当前模型方向，后者用于提醒同批候选中哪些改法已经失败，避免 LLM 重复试验。"
  },
  {
    slug:"mle-star", no:"03", lane:"Agent-as-MLE", tags:["Web Retrieval","Targeted Refinement"], title:"MLE-STAR", subtitle:"Search and Targeted Refinement", paper:"https://arxiv.org/abs/2506.15692", code:"https://github.com/GoogleCloudPlatform/agent-starter-pack/tree/main/agents/mle_star", year:"2025",
    intro:"MLE-STAR 不从空白内部知识开始，而是先从 Web 检索任务适配模型，再用消融定位最值得改的代码块。外循环选组件，内循环只深挖该组件，解决“每轮都重写全管线”的粗粒度探索问题。",
    idea:["先检索并实现多个 task-specific model，按验证分数排序后合并成初始解。","用自动消融估计每个代码块的边际贡献，选择最关键组件作为下一轮目标。","内循环并行提出多个策略，只编辑目标块；历史消融摘要和 plan-score 轨迹用来驱动多样化。"],
    formula:"T_abl^t = Summarize(a_t, Execute(a_t))\nblock_t = argmax_b | h(s_t) - h(Ablate(s_t, b)) |\ns_{t+1} = ReplaceBlock(s_t, block_t, best(plan_1…plan_k))",
    math:"核心近似是组件级 leave-one-component-out 归因。若移除或替换块 b 导致性能大幅下降，则把它当作高杠杆点。它不是严格 Shapley value：组件之间有交互时，单块边际效应会依赖当前代码上下文。",
    memory:{write:"保存检索候选、每轮消融脚本/结果摘要、已修改组件集合，以及 plan→code→score 轨迹。",read:"外循环读取历史消融摘要以减少重复选块；内循环读取同一组件的历史尝试作为反思反馈。",forget:"原始网页与完整执行日志不持续塞入上下文，只保留候选来源和结构化摘要。"},
    schema:"RefinementMemory {\n  retrieved_models[],\n  ablation_history[{block, delta, summary}],\n  block_trials[{block, plan, score, artifact}],\n  modified_blocks[]\n}",
    metrics:[["64%","MLE-Bench Lite 获得奖牌的竞赛比例"],["4 × 4","论文默认外循环与内循环规模"],["组件级","消融决定下一步编辑位置"]],
    limits:["Web 检索会引入可用性、数据污染和复现风险。","消融代理可能误判强交互模块，且自动生成的消融脚本本身可能不公平。","只优化局部块有利于归因，但可能错过需要跨组件协同的大改。"],
    transfer:"非常适合模型内部诊断后的闭环：AXG/Shuffle 先提出模块，targeted refinement 只在该模块内生成候选；每轮保留 block-level memory，形成可审计的局部优化轨迹。"
  },
  {
    slug:"aira", no:"04", lane:"Agent-as-MLE", tags:["Policy × Operator","Scoped Memory"], title:"AIRA", subtitle:"Search, Exploration, and Generalization in MLE-Bench", paper:"https://arxiv.org/abs/2507.02554", code:"https://github.com/facebookresearch/aira-dojo", year:"2025",
    intro:"AIRA 的价值不只是一套更强 Agent，而是把 Agent-as-MLE 形式化为“搜索策略 × 算子集”。论文证明：换成 MCTS 或进化搜索并不会自动变强，算子和 Memory 作用域必须与策略共同设计。",
    idea:["统一表示为 (F, π_sel, O, π_op, τ)：评分、选点策略、变换算子、算子路由和终止规则。","AIRA 算子加入动态复杂度、Crossover 与 scoped memory；不同搜索策略在同一 dojo 中公平比较。","Draft/Improve 只看 sibling memory 促进差异化；Debug 读取完整祖先修复链，避免 undo–redo。"],
    formula:"Agent = (F, π_sel, O, π_op, τ)\nMCTS: h(v|u)=Q(v)+c·sqrt(log N(u)/(N(v)+ε))\nEvolution: P(select v)=F(v)/Σ_u F(u)",
    math:"该形式把“模型聪明不聪明”与“搜索结构好不好”解耦。MCTS 的 UCT 用不确定性奖金找未充分探索分支；进化策略按 fitness 比例选亲本，再用 Improve/Crossover 产生新解并淘汰最差个体。",
    memory:{write:"所有 artifact 都保存 fitness、父子关系、操作类型与执行反馈；Memory 算子再生成供下一次操作使用的摘要。",read:"Draft/Improve 读取当前节点的 children/siblings；Debug 读取整个祖先调试链。",forget:"通过 operator-specific scope 丢弃无关历史；思考 token 不写入共享 memory，避免其他算子继承噪声推理。"},
    schema:"SearchAgent { F, selection_policy, operators, operator_policy, termination }\nMemoryScope(op) = sibling if op∈{Draft,Improve}\n                  ancestry if op=Debug",
    metrics:[["47.7%","最佳策略-算子组合的 MLE-Bench Lite 奖牌成功率"],["+8.1 pp","相对论文中的 39.6% 基线"],["22 tasks","多 seed 的 Lite 评估"]],
    limits:["MLE-Bench 的 CV fitness 仍可能与私榜目标错位。","搜索政策与算子存在强交互，单独比较某个模块容易得出错误结论。","结果依赖高算力与长时间预算，工程吞吐本身也是系统能力的一部分。"],
    transfer:"RecHarness 应把 Router、Skill/Operator 与 Memory 作为三项联合消融，而不是只替换路由算法。不同动作需要不同记忆视野：Debug 看 lineage，Idea/Improve 看 siblings，复盘才看全局。"
  },
  {
    slug:"ai-scientist", no:"05", lane:"Auto Research", tags:["Open-ended Loop","Archive"], title:"The AI Scientist", subtitle:"从想法、实验、论文到评审的端到端闭环", paper:"https://arxiv.org/abs/2408.06292", code:"https://github.com/SakanaAI/AI-Scientist", year:"2024",
    intro:"The AI Scientist v1 把自动科研从“某个子任务”扩展为完整生产线：Idea → Novelty Check → Code/Experiment → Plot → Paper → Review，并把完成的 idea 与 reviewer feedback 加回档案库。",
    idea:["初始代码模板提供一个可快速复现的 baseline，LLM 以此为可执行实验环境。","idea 同时携带描述、实验计划和 interestingness/novelty/feasibility 自评分；文献检索作为 novelty gate。","每个实验完成后写实验日志，再根据结果重规划；最终论文只允许引用真实记录和图表。"],
    formula:"idea_i ~ LLM(archive_{<i})\nscore(idea) = [interestingness, novelty, feasibility]\narchive_{t+1} = archive_t ∪ {paper_t, artifacts_t, review_t}",
    math:"这是开放式进化而非单一标量优化：LLM 充当 mutation operator，档案库保留多样化 stepping stones。自动评审把 soundness、presentation、contribution、overall 等维度映射为选择信号，但自评与代理评审都可能产生 Goodhart 偏差。",
    memory:{write:"idea JSON、检索判重结果、实验 journal、错误、结果文件、图、论文和 reviewer feedback 全部落盘。",read:"下一轮 idea 读取历史 archive；实验阶段读取当前执行历史和 journal；写作阶段读取结构化 notes 与 figures。",forget:"主上下文不承载所有旧实验，跨项目只暴露 archive 摘要/评分；项目内依赖文件系统作为持久记忆。"},
    schema:"ResearchArchiveItem {\n  idea:{description, plan, self_scores},\n  novelty_check, experiment_journal[],\n  artifacts:{code, results, figures},\n  manuscript, review\n}",
    metrics:[["3 阶段","Idea / Experiment / Write-up"],["≤5","单个 idea 的实验迭代次数"],["65%","校准后自动评审 balanced accuracy（论文报告）"]],
    limits:["依赖人写的模板，研究方向受起始代码限制。","自动 reviewer 与生成器可能共享偏差，无法替代真实同行评审。","代理指标和少量实验可能产生看似完整但证据薄弱的论文叙事。"],
    transfer:"把 RecHarness 的每次研究路线存成 archive item，而不仅是 best checkpoint；review feedback 进入下一轮路由先验。真正可复用的是 experiment journal + artifacts + negative result。"
  },
  {
    slug:"ai-scientist-v2", no:"06", lane:"Auto Research", tags:["Progressive Tree","Experiment Manager"], title:"The AI Scientist-v2", subtitle:"Progressive Agentic Tree Search", paper:"https://arxiv.org/abs/2504.08066", code:"https://github.com/SakanaAI/AI-Scientist-v2", year:"2025",
    intro:"v2 去掉人工代码模板，并把线性实验升级为四阶段的渐进式树搜索。Experiment Manager 在阶段边界选择 checkpoint、重复关键实验，并用专门节点管理超参、消融、复现与聚合。",
    idea:["四阶段：可行性原型 → 超参调优 → 核心研究议程 → 消融；每阶段有不同终止条件和节点类型。","buggy 节点走 Debug，non-buggy 节点走 best-first refinement；多个节点并行执行。","性能、训练动态和图表质量共同进入 LLM/VLM 评估；阶段末最佳节点成为下一阶段 root。"],
    formula:"root_{k+1} = SelectBest(Tree_k | metric, dynamics, figures)\nNode = {plan, code, trace, runtime, metrics, plots, LLM/VLM feedback, status}\nReport metric = mean(seed runs) ± std(seed runs)",
    math:"它采用分阶段 best-first search，而非固定 UCT。阶段约束相当于课程：先解决可执行性，再优化性能，最后补科学证据。Replication/Aggregation 节点把随机种子重复与统计汇总编码为搜索动作，降低单次幸运结果被选中的概率。",
    memory:{write:"每个 node 完整保存计划、代码、trace、runtime、指标、图路径、LLM/VLM 反馈和状态；阶段末写 checkpoint。",read:"子节点继承父节点 artifact；Manager 读取整阶段候选的结构化摘要；超参/消融清单用于查重。",forget:"阶段切换只携带被选中的 checkpoint 作为新 root，其他分支留在档案中供审计，但不持续占用活动上下文。"},
    schema:"ExperimentNode { plan, code, error_trace, runtime, metrics,\n  llm_feedback, plot_code, figure_paths, vlm_feedback, status }\nStageCheckpoint { stage, best_node, replications, aggregate }",
    metrics:[["4 stages","渐进实验管理"],["3 papers","真实 workshop 投稿评估"],["1 / 3","达到该 workshop 接收阈值"]],
    limits:["LLM best-first evaluator 不是可校准的数值价值函数，选择过程仍可能不稳定。","阶段末只延续单一冠军会丢掉多样性和可组合 stepping stones。","workshop 级接收不等于稳定生成高影响研究；论文也明确指出新颖性与深层论证仍困难。"],
    transfer:"这是 RecHarness 的直接蓝图：把实验生命周期拆成 feasibility、tuning、agenda、ablation 四个 gate；阶段间只传可恢复 checkpoint，并强制 replication/aggregation 后才形成结论。"
  },
  {
    slug:"agent-laboratory", no:"07", lane:"Auto Research", tags:["Multi-role","Top-program Pool"], title:"Agent Laboratory", subtitle:"Using LLM Agents as Research Assistants", paper:"https://arxiv.org/abs/2501.04227", code:"https://github.com/SamuelSchmidgall/AgentLaboratory", year:"2025",
    intro:"Agent Laboratory 用角色分工而不是单个全能 Agent：PhD、Postdoc、ML Engineer 与 Professor 分别负责文献、计划、实验、解释和写作；mle-solver 维护 top-program pool，paper-solver 维护可编译的论文。",
    idea:["文献阶段迭代执行 summary/full-text/add-paper，形成精简语料；PhD 与 Postdoc 对话形成实验计划。","mle-solver 用 EDIT/REPLACE 变换程序，执行后由 LLM reward model 按研究目标评分。","top-program pool 随机采样父程序并批量并行修改，在保留高质量的同时增加多样性。"],
    formula:"p ~ Uniform(TopKPrograms)\nchildren = {EDIT(p), REPLACE(p)}_batch\nreward = LLM(plan, code, observed_output) ∈ [0,1]\nTopK ← keep_best(TopK ∪ children)",
    math:"top-K 精英保留类似进化算法：选择压力来自 reward 排序，随机抽取精英避免只围绕唯一冠军。LLM reward 是“与研究计划的对齐度”，不是任务真实指标，因此更适合评估研究完整性，却也更容易受语言表象影响。",
    memory:{write:"文献摘要/full text、计划、top programs、执行输出、最近错误、self-reflection、结果解释和报告版本分别落入阶段产物。",read:"每个角色只读取其职责需要的阶段上下文；mle-solver 读取 top pool 与有限错误历史，paper-solver 读取计划/结果/文献。",forget:"完整文献使用 history decay；错误只保留最近若干条；角色边界阻止所有上下文进入单一 prompt。"},
    schema:"ProjectMemory {\n  literature_bank, accepted_plan,\n  top_programs[{code, output, reward}],\n  recent_errors[-5:], reflections,\n  interpretation, report_versions, reviews\n}",
    metrics:[["3 phases","文献 / 实验 / 报告"],["3 repair","代码失败后的修复次数"],["Top-K pool","精英保留 + 并行探索"]],
    limits:["LLM reward 可能奖励“看起来像研究”的代码，而非真实科学价值。","角色增多会产生信息损失和协调开销，阶段交接质量决定上限。","论文报告的文献阶段失败率较高，长文输入和命令循环仍是明显脆弱点。"],
    transfer:"适合把 RecHarness 拆成 Researcher / Developer / Reviewer，而不是让同一 Agent 同时提出假设、实现和验收。top-program pool 也比单一 best checkpoint 更适合保留可组合的模型分支。"
  }
];

const esc = (s) => String(s).replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;");
const topbar = (active="") => `<header class="topbar"><a class="brand" href="/"><i>L</i><span>Ling's Lab</span></a><nav class="nav"><a href="/">学习主页</a><a class="${active==='hub'?'active':''}" href="/lessons/agent-mle-auto-research.html">Agent Research</a><a href="/lessons/paper-methods-map.html">论文地图</a></nav></header>`;

function guide(m){
  const toc = [["overview","定位"],["design","设计思想"],["math","数学原理"],["memory","Memory 设计"],["evidence","证据与局限"],["transfer","迁移到 RecHarness"],["sources","原始资料"]];
  return `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>${m.title} 精读｜Ling's Lab</title><meta name="description" content="${m.title} 的设计思想、数学原理与 Memory 设计精读"><link rel="stylesheet" href="../agent-research.css"></head><body>${topbar("hub")}<main class="page"><section class="hero"><p class="eyebrow">${m.lane} · Paper Deep Dive · ${m.year}</p><h1>${m.title}</h1><p>${m.subtitle}。本页不只复述论文，而是把它拆成可实现的状态、搜索、评估与记忆机制。</p><div class="hero-actions"><a class="button" href="${m.paper}" target="_blank" rel="noreferrer">阅读原论文 ↗</a><a class="button secondary" href="../agent-mle-auto-research.html">返回方法地图</a></div></section><div class="reader-layout section"><aside class="toc"><b>On this page</b>${toc.map(([id,t])=>`<a href="#${id}">${t}</a>`).join("")}</aside><article class="article"><section id="overview"><p class="eyebrow">POSITIONING</p><h2>一句话定位</h2><p class="lead">${m.intro}</p><div class="callout"><strong>先看结论</strong><p>${m.transfer}</p></div></section><section id="design"><p class="eyebrow">SYSTEM DESIGN</p><h2>设计思想：它改变了哪一层？</h2><ol>${m.idea.map(x=>`<li>${x}</li>`).join("")}</ol><div class="flow"><div class="flow-step"><b>01 State</b><span>任务、workspace、预算与当前 artifact</span></div><div class="flow-step"><b>02 Select</b><span>选择分支、组件或阶段目标</span></div><div class="flow-step"><b>03 Operate</b><span>生成、修复、改进或组合代码</span></div><div class="flow-step"><b>04 Evaluate</b><span>执行并获得指标、错误与审阅信号</span></div><div class="flow-step"><b>05 Remember</b><span>写回持久状态并决定下一跳</span></div></div></section><section id="math"><p class="eyebrow">MATHEMATICAL CORE</p><h2>数学原理</h2><div class="formula">${esc(m.formula)}</div><p>${m.math}</p><p><strong>阅读提醒：</strong>这些 Agent 多数不是在参数空间求可微最优，而是在 artifact/code space 做黑盒搜索。公式描述的是算力如何在候选之间分配，而不是替代真实实验。</p></section><section id="memory"><p class="eyebrow">MEMORY ARCHITECTURE</p><h2>Memory 不是聊天记录</h2><div class="memory-grid"><div class="memory-card"><strong>Write · 写什么</strong><span>${m.memory.write}</span></div><div class="memory-card"><strong>Read · 谁在何时读</strong><span>${m.memory.read}</span></div><div class="memory-card"><strong>Forget · 如何控噪</strong><span>${m.memory.forget}</span></div></div><h3>建议的数据结构</h3><div class="schema">${esc(m.schema)}</div><p>实现时应把 raw artifact 与 prompt view 分离：前者完整可追溯，后者按 operator/阶段动态裁剪。摘要只是读取视图，不应覆盖原始实验记录。</p></section><section id="evidence"><p class="eyebrow">EVIDENCE BOUNDARY</p><h2>论文证据与边界</h2><div class="evidence">${m.metrics.map(([v,l])=>`<div class="metric"><strong>${v}</strong><span>${l}</span></div>`).join("")}</div><h3>不能从论文直接推出什么</h3><ul>${m.limits.map(x=>`<li>${x}</li>`).join("")}</ul></section><section id="transfer"><p class="eyebrow">DESIGN TRANSFER</p><h2>迁移到 Ling 的项目</h2><div class="panel"><p>${m.transfer}</p><h3>最小复现顺序</h3><ol><li>先定义 artifact schema 与可重复的 evaluator。</li><li>只实现一种 operator 和一种 selection policy，得到可运行 baseline。</li><li>加入按作用域读取的 memory view，并记录重复试验率。</li><li>最后再增加并行、Crossover、Reviewer 或多角色，逐项做消融。</li></ol></div></section><section id="sources"><p class="eyebrow">PRIMARY SOURCES</p><h2>原始资料</h2><div class="source-list"><a href="${m.paper}" target="_blank" rel="noreferrer">论文：${m.subtitle} ↗</a><a href="${m.code}" target="_blank" rel="noreferrer">官方项目 / 代码 ↗</a></div></section></article></div></main><footer class="footer">Ling's Lab · Agent-as-MLE × Auto Research · 内容基于论文与官方项目整理</footer></body></html>`;
}

for (const m of methods) fs.writeFileSync(path.join(outDir, `${m.slug}.html`), guide(m));

const tableRows = methods.map(m=>`<tr><td>${m.title}</td><td>${m.lane}</td><td>${m.tags[0]}</td><td>${m.memory.read}</td><td><a href="agent-research-guides/${m.slug}.html">精读 ↗</a></td></tr>`).join("");
const cards = methods.map(m=>`<a class="method-card" href="agent-research-guides/${m.slug}.html"><div class="method-top"><span class="method-no">${m.no}</span><span class="tags"><i class="tag ${m.lane==='Auto Research'?'coral':'green'}">${m.lane}</i>${m.tags.map(t=>`<i class="tag">${t}</i>`).join("")}</span></div><h3>${m.title}</h3><p>${m.intro}</p><div class="takeaway"><strong>Memory 关键：</strong>${m.memory.read}</div><span class="open">打开论文精读 HTML →</span></a>`).join("");
const hub = `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Agent-as-MLE × Auto Research 方法地图｜Ling's Lab</title><meta name="description" content="Agent-as-MLE 与自动科研方法的设计思想、数学原理和 Memory 架构路线图"><link rel="stylesheet" href="agent-research.css"></head><body>${topbar("hub")}<main class="page"><section class="hero"><p class="eyebrow">RESEARCH ROUTER · 2024—2025</p><h1>Agent-as-MLE × Auto Research</h1><p>从“让 Agent 写代码”走向“让系统持续做可验证研究”。这张路由页用同一套状态—动作—评估—记忆框架，串联 7 个代表方法与逐篇精读。</p><div class="hero-actions"><a class="button" href="#methods">选择方法</a><a class="button secondary" href="#memory">先看 Memory 设计</a></div><div class="stats"><div class="stat"><strong>7</strong><span>独立方法精读</span></div><div class="stat"><strong>4 + 3</strong><span>MLE Agent / Auto Research</span></div><div class="stat"><strong>3 层</strong><span>Artifact / Episodic / Semantic</span></div><div class="stat"><strong>1 条主线</strong><span>可执行证据驱动迭代</span></div></div></section><section class="section"><div class="section-head"><div><p class="eyebrow">UNIFIED ABSTRACTION</p><h2>先建立共同坐标系</h2></div><p>方法名称不同，但都在回答五个问题：当前状态是什么、下一步选哪里、做什么变换、用什么证据评价，以及哪些经验值得进入下一轮。</p></div><div class="flow"><div class="flow-step"><b>State</b><span>Task + Workspace + Budget + Artifact</span></div><div class="flow-step"><b>Policy</b><span>Greedy / UCT / Evolution / Stage Gate</span></div><div class="flow-step"><b>Operator</b><span>Draft / Debug / Improve / Crossover</span></div><div class="flow-step"><b>Evaluator</b><span>CV、执行、LLM/VLM Review、统计复现</span></div><div class="flow-step"><b>Memory</b><span>持久化证据 + 按作用域构造上下文</span></div></div><div class="formula">x_{t+1} ~ π( · | x_t, View(M_t), budget_t )\ny_t = Evaluate( Execute(x_{t+1}) )\nM_{t+1} = Update(M_t, artifact_{t+1}, y_t, trace_t)</div></section><section class="section" id="methods"><div class="section-head"><div><p class="eyebrow">METHOD ROUTER</p><h2>按你的问题选方法</h2></div><p>先用 AIDE 建立最小可执行树，再按瓶颈选择：上下文污染看 ML-Master，粗粒度编辑看 MLE-STAR，策略/算子耦合看 AIRA；需要覆盖完整科研链路时进入 Auto Research。</p></div><div class="grid">${cards}</div></section><section class="section" id="memory"><div class="section-head"><div><p class="eyebrow">MEMORY BLUEPRINT</p><h2>三层 Memory 设计</h2></div><p>最常见错误是把全部历史拼进 prompt。更稳的做法是完整保存 artifact，再为不同角色和操作动态生成有限视图。</p></div><div class="memory-grid"><div class="memory-card"><strong>Artifact Memory · 事实层</strong><span>代码、commit、配置、日志、指标、图、review。只追加、可复现，是最终证据源。</span></div><div class="memory-card"><strong>Episodic Memory · 轨迹层</strong><span>parent/sibling、操作、假设、结果、失败原因。支持 lineage、对比和恢复。</span></div><div class="memory-card"><strong>Semantic Memory · 规律层</strong><span>从多次实验蒸馏的经验、禁忌和适用条件；必须保留 evidence pointer 和置信度。</span></div></div><div class="schema">ExperimentMemory {\n  artifact: {commit, config, logs, metrics, plots},\n  episode: {parent, hypothesis, operator, outcome, failure_class},\n  semantic: {claim, scope, confidence, evidence_ids},\n  views: {debug_lineage, improve_siblings, reviewer_global}\n}</div><div class="callout"><strong>核心原则</strong><p>Memory 的“存储”与“提示词读取”必须解耦。原始记录永不被摘要覆盖；Debug、Improve、Reviewer 分别读取不同视图。</p></div></section><section class="section"><div class="section-head"><div><p class="eyebrow">COMPARISON</p><h2>方法对照表</h2></div><p>同一个 “memory” 词在不同论文里含义差异很大：有的是全树摘要，有的是局部作用域，有的是文件系统档案或阶段 checkpoint。</p></div><table class="compare"><thead><tr><th>方法</th><th>层级</th><th>主要搜索机制</th><th>读取策略</th><th>精读</th></tr></thead><tbody>${tableRows}</tbody></table></section><section class="section"><div class="panel"><p class="eyebrow">RECOMMENDED READING ORDER</p><h2>推荐阅读顺序</h2><p><strong>做工程 Agent：</strong>AIDE → ML-Master → MLE-STAR → AIRA。先理解 artifact tree，再分别看 memory scope、组件归因、policy × operator 联合设计。</p><p><strong>做自动科研：</strong>The AI Scientist → AI Scientist-v2 → Agent Laboratory。先看端到端 archive，再看阶段化树搜索，最后对比多角色与 top-program pool。</p><p><strong>做 RecHarness：</strong>优先精读 AIRA、ML-Master、AI Scientist-v2；它们分别对应 Router/Skill 联合设计、局部记忆视图和阶段 checkpoint/replay gate。</p></div></section></main><footer class="footer">Ling's Lab · Agent-as-MLE × Auto Research · 论文精读路由页</footer></body></html>`;
fs.writeFileSync(path.resolve("public/lessons/agent-mle-auto-research.html"), hub);

console.log(`Generated ${methods.length + 1} Agent Research HTML pages.`);
