(function(){
  var tabs=document.querySelectorAll('.tab-btn');
  tabs.forEach(function(btn){btn.addEventListener('click',function(){tabs.forEach(function(b){b.classList.remove('active')});document.querySelectorAll('.script-panel').forEach(function(p){p.classList.remove('active')});btn.classList.add('active');document.getElementById(btn.dataset.script).classList.add('active')})});

  var questions=[
    {cat:'fit',tag:'动机',q:'为什么选择百度推荐策略，而不是继续做广告大模型或纯 Agent？',a:'我的长期定位是推荐算法，Agent/Harness 是提高实验与迭代质量的方法，而不是替代推荐建模。百度 App 同时有搜索和 Feed，既能利用强意图搜索行为，也要建模持续消费兴趣；再加上亿级规模、多目标列表生成和内容生态，这些问题与我在序列建模、选择偏差和长周期实验上的经历高度契合。我希望先把首页 Feed 的用户建模和排序基本盘做深，再探索诊断工具与 Harness 如何降低工业试错成本。',trap:'不要回答“百度数据多、平台大”；不要把重点放成“想验证我的 Agent 论文”。'},
    {cat:'business',tag:'业务',q:'百度首页 Feed 的 CTR 下降 5%，你会怎样排查？',a:'先确认口径和数据链路：埋点、去重、归因窗、实验分流、版本与大盘是否同时变化；再按用户、场景、内容、位置、新老用户、网络和时间分桶定位。其次检查供给与流量结构，再看召回覆盖、特征新鲜度、分数分布、校准、延迟与降级。全程同时看时长、快速划走、负反馈和留存，避免把 CTR 下降误判为体验下降。',trap:'先验证 SRM/流量污染。若只影响新用户，优先查冷启动召回与画像缺省值。'},
    {cat:'model',tag:'模型',q:'为什么不能直接用某类内容的 CTR 代表用户偏好？',a:'CTR 是被曝光后的条件行为，类别之间的曝光机会、位置、候选质量和历史策略不同，直接比较会混入选择偏差。可以分别建模曝光 propensity 与条件反馈，联合点击、停留、快速划走和负反馈，通过 IPS/SNIPS、随机探索流量、贝叶斯平滑或 doubly robust 估计校正；画像侧区分短期与长期兴趣并维护置信度。最终在重排阶段联合相关性、多样性和质量，线上以时长、留存、负反馈共同验证。',trap:'IPS 依赖 propensity 估计正确且可能高方差；SNIPS 稳尺度，但不能修复错误 propensity。'},
    {cat:'business',tag:'业务',q:'怎样提升多样性，同时尽量不损失用户满意度？',a:'先区分重复抑制、类目覆盖和兴趣探索三种目标。模型层学习主题相关性与用户探索倾向；重排层用 MMR、DPP 或带约束优化，在主兴趣相关性下控制同质内容密度、作者重复和新颖度；对高置信主兴趣用户减少探索，对兴趣不确定用户增加探索。离线看 NDCG、coverage、ILD 与校准，线上看点击、深度消费、负反馈和留存。',trap:'多样性不是越高越好；“直接加 diversity loss”不够完整。'},
    {cat:'model',tag:'模型',q:'短期兴趣和长期兴趣如何建模与融合？',a:'长期侧用稳定历史、用户画像和跨场景统计形成低频更新表征；短期侧按 session/最近行为序列，用时间间隔、行为类型、内容主题和上下文建模。融合可用 gate 根据 session 一致性、意图强度和置信度动态加权。搜索 query 是强但可能短暂的意图，需要时间衰减与消费反馈校正；异常点击和热点事件通过鲁棒聚合与退场机制降低污染。',trap:'可连接你的原始序列 item-item 交互，但不要只说“长短序列拼接”。'},
    {cat:'model',tag:'模型',q:'Target Attention 和 Self-Attention 有什么区别？为什么两者都需要？',a:'Target Attention 用候选 item 作 query，对历史行为做 candidate-aware 聚合，回答哪些历史与当前候选最相关；Self-Attention 先建模历史行为之间的依赖和兴趣演化。如果只做 Target Attention，历史 item 之间在聚合前缺少显式交互；先 Self-Attention 再 Target Attention，可以先形成上下文化行为表征，再按候选聚合，但要控制序列长度、时延和参数共享，并与等参数基线消融。',trap:'不要说 HSTU 没有序列能力；应说本模型里的 HSTU 接收压缩后的虚拟 Token。'},
    {cat:'model',tag:'模型',q:'ESMM 解决了什么？ESCM²、IPS、SNIPS 和 DR 又分别补了什么？',a:'ESMM 同时学习 CTR 与 CTCVR，并用 pCTCVR=pCTR×pCVR 在全曝光空间训练，缓解 CVR 只在点击样本学习的选择偏差与稀疏。ESCM² 类方法进一步引入反事实风险、imputation/propensity 校正。IPS 在 propensity 正确时可校正但高方差；clipping 稳定却引入偏差；SNIPS 自归一降低尺度波动；DR 同时结合 outcome 与 propensity，实际仍受两模型误差和梯度耦合影响。',trap:'不要说 ESMM 完全消除偏差，也不要说 SNIPS 无偏。'},
    {cat:'project',tag:'项目',q:'为什么 RecHarness 用 Thompson Sampling，而不是让 LLM 直接选方向？',a:'长周期推荐实验反馈稀疏、延迟高、单次成本高。LLM 直接选方向难校准不确定性，也容易受最近日志和叙事偏好影响。Thompson Sampling 用成功/失败更新方向后验，在利用高收益方向的同时保留探索；LLM 只在选中方向内结合 Memory 生成假设与代码。这样预算分配与语义生成分工，失败实验也能显式降低后验。',trap:'Beta-Bernoulli 丢失 gain 幅度；可扩展连续 reward 或 contextual bandit。'},
    {cat:'project',tag:'项目',q:'RecHarness 的 47.92% 是否可能来自验证集过拟合？',a:'有可能。Harness 多轮根据验证反馈调整方向，构成 adaptive search；长期使用同一验证集会有 winner’s curse。现有结论主要比较相同预算下的方向命中率。更严谨的设计应保留独立 final test、使用 nested/rolling validation、限制反馈精度，并报告跨 seed/数据集方差。工业环境还需独立线上实验，不能把 validation hit rate 等同上线收益。',trap:'主动提 multiple comparisons、selection bias 和 checkpoint reuse 风险。'},
    {cat:'project',tag:'项目',q:'为什么 batch 512 到 4096 会改变货架实验结论？如何排除其他因素？',a:'按 0.14% 正例率，512 batch 期望只有约 0.7 个正例，零正例 batch 常见，DR/SNIPS 的分母和辅助误差估计不稳定；4096 时约 5.7 个，监督密度更稳定。但必须排除学习率、global batch、总样本、训练 step 与优化器变化。正确实验要保持总样本与优化预算可比，并报告不同 batch/seed 下方差和有效正例分布。',trap:'期望正例数增加不等于估计无偏。'},
    {cat:'project',tag:'项目',q:'RecKit 的 paired shuffle 能说明因果吗？',a:'固定模型和其余输入，只打乱某模块表示在样本间的对应关系，比较同批样本前后 loss/prediction，因此能回答当前模型是否依赖这个样本—表示对应关系，比只看梯度排名更接近功能验证。但不能直接推出该模块对业务结果的因果作用，因为分布外扰动、特征相关性和下游补偿都会影响结果；还要补 zero/noise、模块消融、跨 seed 和线上实验。',trap:'准确措辞是“依赖性干预证据”，不是“因果证明”。'},
    {cat:'system',tag:'系统',q:'请设计一个完整的 Feed 召回—粗排—精排—重排链路。',a:'多路召回覆盖关注、协同、内容、热点、新内容和搜索意图迁移，通过 ANN/HNSW 或规则检索获得候选；轻量粗排快速缩小，精排用多任务模型预估点击、时长、互动和负反馈；重排处理多样性、新鲜度、作者重复、内容质量与业务规则；最后去重、合规与降级。训练侧包含日志、样本、特征平台、离线评估、发布和 A/B；线上重点是特征新鲜度、p99 延迟、召回覆盖和校准。',trap:'加分点：retrieval/ranking 目标一致性、position bias 与 impression logging。'},
    {cat:'system',tag:'实验',q:'怎样设计 A/B 实验并判断可以上线？',a:'实验前固定主指标、护栏、MDE、样本量和停止规则；检查随机性、SRM、AA 与 pre-period。实验覆盖完整周周期，监控触发率、版本污染、延迟、稳定性和分桶异质性；避免中途反复看数停止。上线判断不仅看 p-value，还看效应大小、置信区间、业务价值、护栏和长期稳定。Feed 护栏包括负反馈、低质率、作者集中度、延迟和崩溃率。',trap:'你的货架材料有 AB 7 天与 AA 3 天；准备解释 AA 用途与流量不等原因。'},
    {cat:'business',tag:'业务',q:'新用户、新内容、新作者分别怎么冷启动？',a:'新用户依靠地域、设备、时间、入口、搜索 query 和少量 session 行为，结合热门与多样性探索；新内容依靠多模态理解和作者先验进入内容召回，再用小流量探索；新作者还要控制生态公平，可基于质量审核、同类作者先验和分层流量阶梯给试探曝光。三者都需要不确定性建模、探索预算、质量护栏和快速退场。',trap:'区分 user cold-start 与 item exposure starvation。'},
    {cat:'system',tag:'实验',q:'离线 AUC 提升，线上为什么可能没收益？',a:'可能来自样本与线上分布不一致、曝光/位置偏差、特征穿越、训练—服务不一致、校准问题、召回瓶颈、排序增益被重排覆盖、触发率低、延迟降级，或 AUC 提升集中在低价值样本。先看触发、分数分布、校准、分桶 GAUC、top-K 排序变化和线上替换率，再看业务指标链路。',trap:'可引用 a2a silent failure：特征名偏差导致 48h 后被误读为无增益。'},
    {cat:'fit',tag:'动机',q:'你的最大短板是什么？',a:'我目前最需要补的是内容 Feed 的长期用户价值和供给生态经验。我的工业经历主要在广告精排，目标更偏 CVR、CTCVR 和收入；百度 Feed 还要同时处理时长、满意度、负反馈、多样性、作者成长和质量。序列建模、偏差校正和实验方法能够迁移，但我不会把广告目标直接等同内容目标。入职后会优先补齐指标体系、供给分层和线上召排链路。',trap:'这是可补的岗位相关短板，同时没有否定你的推荐基本盘。'}
  ];

  var list=document.getElementById('questionList');
  questions.forEach(function(item,i){
    var el=document.createElement('details');el.className='question';el.dataset.cat=item.cat;
    el.innerHTML='<summary><span class="qnum">Q'+String(i+1).padStart(2,'0')+'</span><b>'+item.q+'</b><span class="qtag">'+item.tag+'</span></summary><div class="answer"><div class="answer-grid"><div class="core"><h4>参考回答</h4><p>'+item.a+'</p></div><div class="trap"><h4>追问 / 避免</h4><p>'+item.trap+'</p></div></div></div>';
    list.appendChild(el);
  });
  document.querySelectorAll('.filter-btn').forEach(function(btn){btn.addEventListener('click',function(){document.querySelectorAll('.filter-btn').forEach(function(b){b.classList.remove('active')});btn.classList.add('active');document.querySelectorAll('.question').forEach(function(q){q.hidden=btn.dataset.filter!=='all'&&q.dataset.cat!==btn.dataset.filter})})});

  var mock=[
    ['动机与定位','热身','请做一个 60 秒自我介绍，并说明为什么适合百度推荐策略。','为什么不是纯 Agent？广告推荐经验怎样迁移到内容 Feed？','身份落在“工业推荐精排＋实验智能化”。证据只用短视频线上增量和货架选择偏差；RecHarness 作为方法抽象。动机落在搜索＋Feed、长短期兴趣、多目标和内容生态。'],
    ['业务分析','中等','百度首页 Feed 的 CTR 突然下降 5%，你怎样排查？','如果时长上升但 CTR 下降呢？如果只影响新用户呢？','先查口径、埋点、分流和 SRM，再按用户/场景/内容/位置/时间分桶；检查供给、召回、特征新鲜度、校准、触发率、延迟和降级。CTR 降但时长升可能是少点但深消费。'],
    ['用户建模','困难','如何从曝光和行为中估计用户对类别标签的真实偏好？','类别曝光极少怎么办？多个粒度标签怎样融合？','直接 CTR 混入曝光和位置偏差。联合 propensity、点击、停留和负反馈，使用探索桶、IPS/SNIPS/DR 与贝叶斯平滑；区分 session 与长期兴趣，维护时间衰减和置信度。'],
    ['多目标推荐','困难','点击、时长、互动、负反馈和留存冲突时，如何设计模型与排序？','为什么不直接加权求和？权重怎样确定？','任务先校准；结构上用 PLE/MMoE 或任务塔；融合层依据场景动态权重，重排加入多样性和质量约束。权重由业务价值、Pareto frontier 和线上实验确定，主指标与护栏预注册。'],
    ['项目深挖','中等','短视频项目为什么在 Target Attention 前加 Self-Attention？','HSTU 已存在，为什么还需要？怎样证明不是参数量收益？','Target Attention 做候选相关聚合，Self-Attention 建模历史 item 依赖。HSTU 在该结构里处理压缩后的虚拟 Token。归因需保持主干任务头不变，并做等参数、共享方式、长度与时延消融。'],
    ['反事实学习','困难','解释货架项目中的 IPS、SNIPS、DR 与梯度隔离。','SNIPS 有什么偏差？stop-gradient 会不会损失信号？','IPS 高方差；SNIPS 自归一但有限样本有偏；DR 结合 outcome 与 propensity。辅助塔误差可能污染共享表征，所以先 stop-gradient 建安全边界，再用消融判断是否损失协同。'],
    ['训练诊断','中等','为什么 batch 512 到 4096 会让结论翻转？','如何排除学习率和训练步数影响？','0.14% 正例率下，期望正例从 0.7 增到 5.7，SNIPS/DR 更稳定。但必须保持总样本、step、学习率策略和预算可比，报告零正例 batch 比例、有效样本量与跨 seed 方差。'],
    ['Harness','困难','RecHarness 的 47.92% 有效 trial 如何定义？会不会验证集过拟合？','为什么用 Bernoulli 成功而不是连续 gain？','精确数字 23/48；随机 11/49，LLM 10/46。多轮 adaptive search 可能过拟合，需要独立 final test、nested/rolling validation 和多 seed。Bernoulli 简洁但丢失幅度，可扩展连续 reward。'],
    ['模型诊断','困难','Activation×Gradient 排名为什么还需要 paired shuffle？','shuffle 能证明业务因果吗？','AXG 是局部敏感性，受尺度和噪声影响；paired shuffle 破坏样本—表示对应关系以检验功能依赖。它不直接证明业务因果，还需 zero/noise、模块消融、跨 seed 与线上验证。'],
    ['系统设计','中等','请设计百度 Feed 的召回—粗排—精排—重排链路。','亿级规模怎样控延迟？失败怎样降级？','多路召回＋ANN/HNSW，轻量粗排，多任务精排，最后做多样性、质量、作者重复和规则重排。延迟靠分层预算、缓存、特征预计算与并发；准备热榜、轻模型和规则降级。'],
    ['实验设计','中等','离线 AUC +0.09pp，怎样判断是否值得上线实验？','AUC 升但线上没收益，先查什么？','看分桶 GAUC、top-K 变化、校准、触发覆盖、时延和成本；上线前定义主指标、护栏、MDE 和停止规则。线上无收益先查触发率、一致性、重排覆盖、流量污染和校准。'],
    ['综合判断','高压','入职后给你两周优化首页 Feed，你会怎么开始？','不能改大结构，只能做一个方向，你选什么？','第一周建立指标树和诊断切片，复现一个明确痛点；第二周选低风险高信息量实验。方向由数据决定，例如搜索后短期意图过强，就做时间衰减/置信度 gate 与小流量 A/B，并设长期护栏。']
  ];
  var idx=0,remain=90,timerId=null,scores=new Array(mock.length).fill(null);
  var timer=document.getElementById('timer'),q=document.getElementById('mockQuestion'),cat=document.getElementById('mockCategory'),level=document.getElementById('mockDifficulty'),follow=document.getElementById('mockFollowups'),answer=document.getElementById('mockAnswer'),notes=document.getElementById('mockNotes'),bar=document.getElementById('progressBar'),progress=document.getElementById('progressText'),scoreTotal=document.getElementById('scoreTotal'),scoreHint=document.getElementById('scoreHint');
  function fmt(v){return String(Math.floor(v/60)).padStart(2,'0')+':'+String(v%60).padStart(2,'0')}
  function stop(){if(timerId){clearInterval(timerId);timerId=null}}
  function save(){try{localStorage.setItem('baidu-mock-note-'+idx,notes.value)}catch{/* local storage may be unavailable */}}
  function updateScore(){var total=scores.reduce(function(s,v){return s+(v||0)},0),done=scores.filter(function(v){return v!==null}).length;scoreTotal.textContent=total+' / 24';scoreHint.textContent=done===mock.length?(total>=18?'可以进入高压追问；复盘 1 分题':total>=13?'基本盘可用；重练 0 分题':'先回到讲稿和核心题库'):'已评分 '+done+' 题'}
  function render(){stop();remain=90;timer.textContent=fmt(remain);cat.textContent=mock[idx][0];level.textContent=mock[idx][1];q.textContent=mock[idx][2];follow.textContent=mock[idx][3];answer.textContent='参考要点：'+mock[idx][4];answer.classList.remove('show');progress.textContent='第 '+(idx+1)+' / '+mock.length+' 题';bar.style.width=((idx+1)/mock.length*100)+'%';try{notes.value=localStorage.getItem('baidu-mock-note-'+idx)||''}catch{notes.value=''}}
  document.getElementById('startTimer').addEventListener('click',function(){stop();timerId=setInterval(function(){remain=Math.max(0,remain-1);timer.textContent=fmt(remain);if(remain===0)stop()},1000)});
  document.getElementById('showAnswer').addEventListener('click',function(){answer.classList.toggle('show')});
  document.getElementById('nextQuestion').addEventListener('click',function(){save();idx=(idx+1)%mock.length;render()});
  document.getElementById('prevQuestion').addEventListener('click',function(){save();idx=(idx-1+mock.length)%mock.length;render()});
  document.querySelectorAll('.score-btn').forEach(function(btn){btn.addEventListener('click',function(){scores[idx]=Number(btn.dataset.score);updateScore()})});
  notes.addEventListener('input',save);render();updateScore();
})();
