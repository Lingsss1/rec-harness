# KuaiRec 与淘宝循环模型：全部迭代实验总账

快照时间：**2026-09-09T07:36:23.974768+00:00**。覆盖工作区当前保存的迭代记录；本次只整理文档，没有重跑实验或改写主线程研究记忆。

## 阅读入口

- [KuaiRec逐实验明细](KUAIREC_EXPERIMENTS.md)：方法、完整验证指标、诊断、失败、归因及后续假设。
- [淘宝逐实验明细](TAOBAO_EXPERIMENTS.md)：四任务指标、宏平均、诊断、失败与未归档实验。
- [最新轮补充记录](LATEST_READOUT_RETRIEVAL.md)：全部10项预定配对、16项读出扰动效应及延迟测量状态。
- [证据与覆盖核对](EVIDENCE_INDEX.md)：读取的元数据文件、完整SHA256、计数口径与缺失记录。

## 覆盖范围与计数

| 数据集 | 历史episode | 历史已归档完整训练 | 快照完整result文件 | 本次补记未归档运行 | 其他历史记录 |
|---|---:|---:|---:|---:|---:|
| KuaiRec | 233 | 167 | 167 | 0 | 66 |
| 淘宝 | 170 | 109 | 117 | 8 | 61 |

episode包括完整训练、固定checkpoint诊断、配对统计、失败启动及历史判读，不能当作独立训练次数。历史备份目录不重复计入；同checkpoint多次诊断不增加独立训练证据。新result存在仅证明已保存指标，完整阶段审计/配对/延迟仍须另核。

补充附件保留最新轮预定10项配对（本快照已有10项完整结果）、1项含16效应的固定读出诊断及延迟测量状态；KuaiRec另保留2次早期smoke。这些补充都不计入完整训练次数。

## 协议与可比较性

| 项目 | KuaiRec | 淘宝 |
|---|---|---|
| 标签/任务 | high_watch = watch_ratio > 2.0 | is_click、cart、fav、buy；macroAUC为四任务等权均值 |
| 切分/行数 | 用户时间80/10/10；train 10,022,544，valid 1,253,001 | 发布协议；train 17,822,296，valid 2,871,767 |
| 训练预算 | 1 epoch，batch8192，1240更新 | 1 epoch，batch8192，2193更新 |
| 训练seed | 前期有20262027/28/29；K06起固定已有20262027 | 固定20262027 |
| 优化器 | 以各manifest记录为准，早期AdamW/Adagrad | 先AdamW/Adagrad；后换PSGDKron；近期每步预条件、denseLR1e-4、sparseLR0.05、clip1 |
| 测试集 | 没有用于迭代评估 | 没有用于迭代评估 |

Cross8是主基线，Cross11是优化参考。KuaiRec初期原Cross8与其他模型存在clip及验证Q3路由差异；全Query恢复的收益必须单独计。三seed均值不能与后续单seed最好值直接相减作为结构增量。两个数据集标签/切分不同，AUC数值也不可横比。

淘宝沿用发布数据的cart/fav/buy窗口标签及历史构造；既有审计保留了用户/类目/品牌24小时标签、历史延迟可用性和同时间戳顺序的限制。结论限定这个发布版离线协议，不当作线上因果验证。

同权重删轮/置零/打乱检验已训练模型的依赖；从头重训才检验结构变更。用户bootstrap区间条件于固定checkpoint，不覆盖训练seed方差或连续候选筛选。早期部分轮次只对正点增量计算区间，不能据未报告区间推断显著性；近期要求保留全部预定配对。

计时通常是同卡、BF16、batch1024、固定真实验证batch的重复前向，不含IO，不代表线上端到端延迟；不要跨轮直接用不同计时相减。

## 当前有证据支持的关键结果

| 对象 | 已保存结果 | 归因与边界 | 后续检验 |
|---|---|---|---|
| KuaiRec r7_small_attention_first | AUC 0.871234058；8,115,473参数 | N→S→A→N(context)→S，两轮共享，仅首次额外attention；相对直接父模型+0.001081，属于当前固定seed优先候选 | 保留每轮职责的对照；不能把后续负结果解释为循环无效 |
| KuaiRec 第二次上下文回读 | 省略回读AUC下降0.001522 | 两次历史读取仍在；支持该结构的第二次上下文读取价值，不能直接外推淘宝 | 对每个目标结构重新控制读取来源和位置 |
| KuaiRec 小容量取舍 | field FFN512 AUC0.871053474，7,852,817参数 | 接近r7且更小，UAUC/NDCG较低；无明确额外速度收益 | 分开参数节省、AUC和排序指标 |
| KuaiRec 变深度排序取舍 | curriculum13 NDCG0.421730872，AUC0.864854885 | 用户排序提高伴随全局AUC/校准明显下降 | 不作为AUC优化成功；课程顺序/norm后续已测且未恢复父模型峰值 |
| 淘宝最佳已归档点估计 | psgd_precond_every_step macroAUC0.828623533 | 优化器贡献须用同结构比较，非循环独有增量 | 保留为所有候选的旁侧参考 |
| 淘宝去反馈两轮 | explore_dual_no_feedback 0.828521213 | 接近最佳单轮；相对有反馈+0.001127326，CI为正，但混合权重也变了 | 保留去反馈核心并分离后续改动 |
| 淘宝小q4两轮 | ctx_q4 0.828472565；51,724,228总参数 | 较q8略低且同批前向更快；大量总参数差来自未启用task gate，不能等同活跃计算变化 | 在小模型上优化检索条件/任务读出 |
| 淘宝q4两轮−一轮 | macroAUC +0.001960375；95%CI[0.000997664,0.002777110] | 匹配q4从头训练，支持这个局部结构的两轮价值；不证明超过最佳单轮架构 | 与任务特定读出和共享读出做控制 |
| 淘宝差值修正重跑 | query_q4_difference 0.825550580 | 已完成必需重跑；低于mean、原q4及最佳单轮 | 降低差值加性修正设计优先级 |
| 淘宝修正/路线/门控 | q4 work修正接近原q4；路线与有界混合8+8组无新候选 | 原检索依赖存在，但新参数/门方差/敏感性不保证重训收益 | 优先最小、可归因的第二轮检索和读出控制 |
| 淘宝第二次attention分配 | q4.buy reuse−native IPW BCE +0.000014311656；预定99.375%区间超过门槛 | 仅该checkpoint/task通过严格方向门；其余7项不确定。不会推出其他任务没有循环收益 | 保留该行为，新增结构还须完整训练对照 |
| 淘宝任务差异 | q4加购/购买AUC收益大于点击；q8删第二整轮四任务均下降 | 任务、指标、模型及干预范围不同；不能说只有购买有新增信息 | 任务独立读出与共享读出比较；固定checkpoint mean/zero只检验依赖 |

## 全部结构迭代路线

下表是阶段索引；每个seed、每个负结果及诊断/失败均在对应明细中保留。阶段“后续假设”记录研究链条，已完成后续轮不重启。

### KuaiRec

| 阶段 | 方法 | 归因 | 后续假设 |
|---|---|---|---|
| [K01 基线、整段与中间层循环](KUAIREC_EXPERIMENTS.md#phase-1) | Cross8 主基线、同优化设置 Cross8、Cross11；整段/中间循环2或4次；独立6/12层。 | 初期同时存在clip及验证Query路由差异。三seed均值的middle2接近参考；整段增加深度没有稳定收益。不能把跨设置差值全归于循环。 | 先统一训练/验证Query路由，再研究入口锚定和更新阻尼；这已在K02及后续轮检查。 |
| [K02 Query路由与阻尼](KUAIREC_EXPERIMENTS.md#phase-2) | 固定checkpoint恢复全Query，另训练入口混合/半幅更新。 | 恢复训练时的全Query改善平均AUC和Logloss；被固定置零Query的零梯度不是冗余证据。 | 改变循环位置和字段/历史交互来源，而非只加次数；见K03。 |
| [K03 循环位置](KUAIREC_EXPERIMENTS.md#phase-3) | 字段末层、历史首/尾层、交替及历史后字段回读。 | 位置和读取来源比一味增加深度更有研究价值；字段回读保留成本取舍，字段循环有明显seed波动。 | 分开用户上下文、目标物品及全部字段回读；见K04。 |
| [K04 上下文回读来源](KUAIREC_EXPERIMENTS.md#phase-4) | 历史后回读context、target或all，并控制归一化。 | 只回读用户/上下文的三seed平均表现较好，但源token数量和计算量也改变；不是跨数据集通用结论。 | 减少独立block并检查额外attention与保留深度的交互；见K05。 |
| [K05 紧凑共享循环与额外attention](KUAIREC_EXPERIMENTS.md#phase-5) | 减少历史/字段块；有无额外共享Query self-attention对照。 | compact_seq1三seed平均AUC=0.870296，attn_full=0.870442；attention收益依赖底层深度，单历史块时也可能退化。 | 保持少量共享轮次，移动回读/attention位置并进一步压缩；见K06。 |
| [K06 调度与小模型](KUAIREC_EXPERIMENTS.md#phase-6) | 12个单seed结构：无循环、单字段块、末尾回读、history-only、三轮、独立权重及组合。 | r6_field_after偏AUC，r6_small_attention_after偏压缩；收益不能直接相加；无循环和独立权重对照支持当前两轮的局部价值。 | 在小模型中只保留首次或末次额外attention，并检查memory接口；见K07。 |
| [K07 首次attention与memory接口](KUAIREC_EXPERIMENTS.md#phase-7) | 单字段/单历史共享两轮，仅首次额外attention；另测末次attention、norm位置和附加K/V。 | r7_small_attention_first AUC=0.871234058；相对每轮额外attention的直接父模型+0.001081。参数8.115M；没有证明所有attention都应只做一次。 | 检验一/二/三轮及norm/残差组合；见K08。 |
| [K08 循环深度与残差/归一化](KUAIREC_EXPERIMENTS.md#phase-8) | 8个固定seed对照：1/3轮、pre/mid-norm及半幅/双幅更新。 | 同参数完整训练：两轮高于一轮0.001138、三轮低于两轮0.003163；一次循环是低成本排序候选。八个初始YAML失败不算模型效果失败。 | 分离第二次上下文回读及共享接口的作用；见K09。 |
| [K09 第二次回读与分轮共享](KUAIREC_EXPERIMENTS.md#phase-9) | 省略/替换/前移/后移上下文回读；克隆独立历史权重；字段/历史分轮norm。 | 省略第二次上下文回读AUC下降0.001522；克隆独立历史权重下降0.000461并增参。结论限定原交替结构和预算。 | 逐一精简第二轮self-attention、cross和FFN；见K10。 |
| [K10 第二轮子层选择](KUAIREC_ROUNDS10_18.md#phase-10) | 第二次字段/历史读取只保留Cross、Cross+FFN或Self+Cross。 | 8组均未超过完整父模型；部分方案省计算。删除子层同时删除其norm，不能独立识别算子作用。 | 保留完整block，以中性初始化的轮次向量/门控条件化；见K11。 |
| [K11 轮次向量与Query门控](KUAIREC_ROUNDS10_18.md#phase-11) | 字段/历史/双位置零初始化轮次embedding与Query条件门控。 | 8组均未刷新r7；history_embedding非常接近父模型，但校准/排序存在代价。非零参数或敏感性不等于重训收益。 | 完整保留子层与norm，仅重排信息流；见K12。 |
| [K12 完整子层顺序](KUAIREC_ROUNDS10_18.md#phase-12) | Self/Cross/FFN顺序改为CAF/AFC，比较仅第二轮改变与两轮均改变。 | 8组均未超过标准顺序；全轮CAF常优于只改第二轮，但各指标不一致。 | 在两轮之间更新历史源，分开Query写入与历史self-attention；见K13。 |
| [K13 历史源写入](KUAIREC_ROUNDS10_18.md#phase-13) | Query或历史self-attention更新历史；不同位置；zero/warm初始化。 | 8组未提高父模型AUC，部分UAUC/NDCG有取舍，新增writer提高开销。 | 在固定循环图上比较参数预算接近的门控FFN；见K14。 |
| [K14 门控FFN与激活](KUAIREC_ROUNDS10_18.md#phase-14) | 字段/历史/双位置SwiGLU、GEGLU和密集SiLU。 | 8组均未刷新r7；同初值SwiGLU优于GEGLU不等于优于原GELU。更高有效秩没有转化为更高AUC。 | 保留GELU，单独改变FFN宽度以分离容量因素；见K15。 |
| [K15 普通FFN容量分配](KUAIREC_ROUNDS10_18.md#phase-15) | 字段/历史/双位置宽度512、680及历史/双位置1536。 | r15_w512_field AUC=0.871053474、7.853M参数，较r7少262656；更少参数有价值，但短基准没有证明明显提速。 | 在相同更新预算下变化训练循环深度、固定两轮验证；见K16。 |
| [K16 可变训练深度](KUAIREC_ROUNDS10_18.md#phase-16) | alternate13、mixed1232、先1后3课程；原容量及field512。 | 6组AUC均低于固定两轮；base curriculum13的NDCG=0.421730872最高，但AUC=0.864854885明显下降。 | 控制各深度更新次数并改变阶段顺序，以两轮结束；见K17。 |
| [K17 训练深度阶段顺序](KUAIREC_ROUNDS10_18.md#phase-17) | phase132/312/123，在两种容量上匹配各深度更新次数。 | phase132相对其他顺序恢复部分AUC，但都未超过固定两轮；phase312也以两轮结束，故不能只归因于末期深度。 | 按执行深度区分最终LayerNorm，保留共享主干；见K18。 |
| [K18 深度专属最终norm](KUAIREC_ROUNDS10_18.md#phase-18) | 两容量×phase132/mixed1232，克隆初始化独立最终norm。 | 4组均低于固定两轮父模型；只在原容量部分安排恢复AUC，不能概括为解决归一化冲突。 | 固定两轮下区分block内部轮次norm；K19仅准备、后因淘宝迁移延后，未训练。 |

### 淘宝

| 阶段 | 方法 | 归因 | 后续假设 |
|---|---|---|---|
| [淘宝迁移与裁剪基线](TAOBAO_EXPERIMENTS.md#phase-initial) | Cross8、Cross8 clip1、Cross11及KuaiRec r7结构从头训练四任务。 | r7直接迁移低于淘宝Cross8；clip10→1对Cross8的macroAUC贡献+0.001979455，不能归于循环。 | 增宽与共享循环分开做因子对照；后见round2。 |
| [增宽与整段循环](TAOBAO_EXPERIMENTS.md#phase-round2) | token宽度256/384/512；单轮/整段循环；增宽r7。 | 384单轮优先，整段循环在三个宽度下均降低macroAUC；增益主要来自点击，且有延迟代价。 | 把循环移到局部并减弱重复更新；见round3。 |
| [局部阻尼循环](TAOBAO_EXPERIMENTS.md#phase-round3) | 整段、尾部和post等位置及半幅更新。 | tail-half在AdamW下提高四任务AUC，macro差+0.001386；它比全栈重复更有价值，但增加计算。 | 控制尾部宽度、幅度和轮数；见round4。 |
| [尾部循环细化](TAOBAO_EXPERIMENTS.md#phase-round4) | 6个宽度/更新幅度/深度组合。 | 全部低于t3_tail_half；继续堆叠和增宽没有改善。 | 固定LR/clip/结构，把AdamW换为PSGDKron。 |
| [PSGDKron首个替换](TAOBAO_EXPERIMENTS.md#phase-optimizer-psgd) | tail-half保持结构、学习率及clip1，替换稠密优化器。 | 优化器变化和结构变化分开归因；单组改进需同结构对照。 | 在另外三种结构上并行比较优化器。 |
| [PSGDKron跨结构对照](TAOBAO_EXPERIMENTS.md#phase-optimizer-psgd-parallel) | Cross8、384单轮、post-half与既有tail-half对照。 | 四结构点估计均受益；并非所有条件区间均为正。宽384单轮更好，不能认为循环与优化器收益相加。 | 检查预条件更新频率和beta，固定其他预算。 |
| [预条件更新与beta](TAOBAO_EXPERIMENTS.md#phase-optimizer-psgd-tune) | PSGDKron beta0.95、每步预条件更新等3组。 | psgd_precond_every_step macroAUC=0.828623533，为当前已归档最高点估计。 | 在该优化器条件下继续研究循环职责和位置。 |
| [中间共享循环](TAOBAO_EXPERIMENTS.md#phase-recurrence-middle) | 中间层两次调用，有无固定入口注入。 | 默认预条件调度下middle_r2接近单轮；anchor0.1退化。 | 检验训练深度和中间监督，区分训练问题与结构问题。 |
| [循环训练方式](TAOBAO_EXPERIMENTS.md#phase-recurrence-training) | mixed123训练深度或两轮中间监督，验证固定两轮。 | 两组均退化；推断深度更鲁棒不等于固定两轮AUC更高。 | 改变共享核心更新或交互方式。 |
| [共享核心更新](TAOBAO_EXPERIMENTS.md#phase-recurrence-core) | 第二轮半幅更新或有界交互。 | 半幅明显退化；有界交互接近但未优于纯R2。 | 将候选放到最佳每步预条件设置下做匹配控制。 |
| [最佳优化器下重测循环](TAOBAO_EXPERIMENTS.md#phase-recurrence-best) | 每步预条件更新下的纯R2及有界交互。 | 两组均低于最佳单轮；优化器与结构存在交互。 | 只放松第二轮输出投影共享；见adapter。 |
| [第二轮输出LoRA](TAOBAO_EXPERIMENTS.md#phase-recurrence-adapter) | 输出投影rank8 LoRA，增加6144参数。 | 较纯R2点估计恢复+0.000256，但低于最佳单轮且有任务/Logloss代价。 | 将工作状态与预测状态分开。 |
| [双状态一/二轮](TAOBAO_EXPERIMENTS.md#phase-recurrence-dual-state) | 工作状态和预测状态分离；比较1/2轮。 | R2−R1 macroAUC+0.000762715但95%区间跨0，加购收益较明确；任务间方向不一致。 | 比较去反馈、GRU、归一化及共享范围。 |
| [循环核心多结构探索](TAOBAO_EXPERIMENTS.md#phase-recurrence-explore) | repeat norm、GRU、entry residual、去预测反馈、独立层、双层core。 | 去反馈两轮0.828521213接近最佳单轮；相对有反馈提高，但输入混合权重也改变，不能把全部增量独归于删反馈。 | 围绕去反馈设计控制轮数、宽度、入口权重。 |
| [去反馈参数控制](TAOBAO_EXPERIMENTS.md#phase-recurrence-no-feedback) | 3轮、256/512宽度、入口权重0.25/0.75。 | 五组均低于原去反馈两轮，继续加轮数/整体加宽优先级降低。 | 研究工作/预测路径和每轮读什么。 |
| [有反馈双状态容量](TAOBAO_EXPERIMENTS.md#phase-recurrence-dual-width) | 双状态R2宽度256/512。 | 宽度增大不保证优于最佳单轮；保留与同架构原宽度的配对，而不混入去反馈效应。 | 固定合适宽度并研究记忆读取和状态路线。 |
| [固定工作记忆](TAOBAO_EXPERIMENTS.md#phase-recurrence-fixed-memory) | 第二轮复用固定工作记忆。 | 移除动态第二次读取降低质量；表明当前模型依赖更新，不给出最优query公式。 | 保留动态历史读取，分解输入条件。 |
| [循环状态路线](TAOBAO_EXPERIMENTS.md#phase-recurrence-state-routes) | 入口去除、工作/预测半幅、预测重置、norm、额外内循环。 | 7组均低于原去反馈；状态持久性和原始混合应作为后续对照。 | 测试上下文回读、attention位置与查询槽数。 |
| [上下文回读、attention位置、Query容量](TAOBAO_EXPERIMENTS.md#phase-recurrence-context-read) | 中间context回读；attention中间/末尾/两处；q4/q6；入口1层。 | 额外回读与attention迁移未提升；ctx_q4宏AUC0.828472565、总参数51.724M，前向均值较q8少约9.8%。总参数下降不等于活跃计算同比下降。 | q4/q8固定两轮，检查局部容量及query信息来源。 |
| [小模型轮数与局部容量](TAOBAO_EXPERIMENTS.md#phase-recurrence-q4-capacity) | q4一轮、post FFN2/8、FFN2、入口3层及q8 FFN控制。 | q4两轮比一轮+0.001960375；其余7个容量改动未超过各自父模型。 | 优先改善第二轮条件形成，避免相邻容量标量扫参。 |
| [第二轮记忆来源拼接](TAOBAO_EXPERIMENTS.md#phase-recurrence-memory-sources) | q4/q8分别拼接入口、历史、预测/工作状态等K/V来源。 | 8组均低于对应原循环；增加memory来源没有自动提升检索。 | 保留原memory，检验入口/工作/均值/差值条件修正。 |
| [第二轮输入低秩修正](TAOBAO_EXPERIMENTS.md#phase-recurrence-query-correction-v2) | rank16加性修正，条件取entry/work/mean/difference；初始函数兼容。 | q4 work接近原q4（差−0.000006789），q8 entry在新增候选中较好；8个完成变体均未超过原循环。修正同时影响检索、交互和residual，归因未分离。 | 用固定checkpoint分路干预，再用单独路线重训验证。 |
| [差值条件重跑](TAOBAO_EXPERIMENTS.md#phase-recurrence-query-difference-retry) | 保留固定seed和参数，修复近零PSGD初始化后完成q4 difference训练。 | 必需差值重跑已完成：macroAUC0.825550580，低于mean、原q4和最佳单轮；初始化失败不是结构反证，完成结果才是。 | 降低该差值加性设计优先级，保留更好对照作为路线研究起点。 |
| [输入修正路线隔离](TAOBAO_EXPERIMENTS.md#phase-recurrence-query-routes) | q4 work/q8 entry各只作用检索、交互、residual或attention。 | 8组均低于对应全路线及原循环点估计，按预定质量/成本规则不留新候选；同权重敏感性没有变成重训收益。 | 改变更新形式，比较有界混合与同容量常量门。 |
| [有界门控混合](TAOBAO_EXPERIMENTS.md#phase-recurrence-query-mixing) | rank16、bound0.25；q4/q8×channel/query×joint/constant。 | 8组均未通过预定候选规则；joint胜过弱constant不等于优于原循环；q4 fav零化/打乱门反而改善BCE。 | 先用诊断定位第二次attention分配是否有独立价值，降低同类门扫参优先级。 |
| [任务读出与第二轮query专属参数](TAOBAO_EXPERIMENTS.md#phase-recurrence-readout-retrieval) | q4/q8×共享读出/任务读出/query norm独立/query投影独立，8组固定预算。 | 8组完整训练macroAUC均低于同q原循环，且点差均超出near阈值；初始函数匹配并不保证优化轨迹一致。读出与检索共享需分别解释。 | 保留全部10组用户配对和同卡延迟的后处理结果；task对shared的相对改善及固定权重诊断，不能替代对原循环的重训比较。 |

## 最新八组的快照状态

这八组已经采用相同训练seed、2193更新、两轮、无预测状态反馈。读出组计算h_i=T(y_i)，r_t=h2+a_t(h1−h2)，a是无约束实数；检索组只解除第二次q_norm或完整Wq+bias的共享。两种假设分别分析，不把读出收益等同检索改进。

| 实验 | 快照完整macroAUC | Δ同q原循环 | Δ最佳单轮 | 完整更新 | 终止回执 |
|---|---:|---:|---:|---:|---|
| `role_q4_query_norm` | 0.827361315 | -0.001111250 | -0.001262217 | 2193 | 0 |
| `role_q4_query_projection` | 0.826303350 | -0.002169216 | -0.002320183 | 2193 | 0 |
| `role_q4_read_shared` | 0.826371326 | -0.002101239 | -0.002252206 | 2193 | 0 |
| `role_q4_read_task` | 0.826192700 | -0.002279866 | -0.002430833 | 2193 | 0 |
| `role_q8_query_norm` | 0.824924790 | -0.003596423 | -0.003698742 | 2193 | 0 |
| `role_q8_query_projection` | 0.826579499 | -0.001941715 | -0.002044034 | 2193 | 0 |
| `role_q8_read_shared` | 0.821507140 | -0.007014073 | -0.007116392 | 2193 | 0 |
| `role_q8_read_task` | 0.826662645 | -0.001858568 | -0.001960887 | 2193 | 0 |

8组均未超过对应原循环，且macroAUC点差均低于near规则下限−0.0001，当前不支持新增质量或近似质量候选。q8 task相对shared恢复质量，仍低于原q8；不能把对较弱shared的改善当作可用增量。
预定10项配对已有10项完整结果，全部四任务及任务间差值保留在[补充记录](LATEST_READOUT_RETRIEVAL.md)。

**后处理快照**：

- `reports/quality-summary.json`：passed。
- `reports/paired-analysis/completion.json`：completed。
- `reports/latency/completion.json`：本快照未发现。
- `reports/readout-diagnostics/aggregate/completion.json`：completed。
- 延迟测量有失败回执：`AttributeError: list has no attribute detach`；未形成完整11模型计时，不能声称提速。

新结果未完成全部预定后处理前不替换已归档最佳。预定保留全部10组配对（8个新−原循环、2个task−shared）、全部四任务AUC/BCE和11模型同GPU延迟。near候选阈值为相对同q原循环ΔmacroAUC≥−0.0001且平均前向延迟比≤1.05；这是队列规则，非统计等价。

## 延后及未执行计划

- KuaiRec K19：`deferred_untrained`。已准备内部轮次norm代码和5项CPU测试，但没有新增训练配置或正式训练，因淘宝迁移优先而延后。
- 淘宝capacity-layout：`not_launched_superseded_by_user_context_reread_direction`。被上下文回读方向替代，不能记作完成实验。

## 当前后续假设与反证条件

1. **第二轮query是否需要不同的归一化/投影。** q4/q8独立对照，若相对原循环的质量和成本都无改善，降低这种解除共享的优先级；文献动机不能替代本地结果。
2. **不同任务是否应读出不同轮次状态。** task必须与shared和原末轮读出都比较；系数不同或同权重扰动有害，均不足以证明任务读出能改善重训。
3. **每轮主动读取什么比增加轮数更重要。** KuaiRec上下文回读有效、淘宝直接照搬回读退化；后续来源/位置设计须在淘宝现有核心上单独控制，不能预设跨数据集迁移。
4. **小模型容量分配。** 优先保留已验证的低成本候选；新增参数只有在固定质量/成本对照中有增量才支持扩展，不继续无边界相邻标量扫参。

这些是可证伪研究问题，不是本次文档整理触发的新训练任务。

## 文档边界

本报告保存来源中的历史结论，同时用后续结果限制早期推测：例如“敏感路径值得保留”与“扩大该路径会提升”分开；“某任务区间为正”与“任务间增量显著不同”分开。缺失证据明确标记，不补造归因。

只上传本目录Markdown文档。原始数据、逐用户/逐样本预测、模型权重、日志、运行环境和主线程memory均不随文档上传或修改。
