# Amazon NextItNet v5 实验进度（2026-09-23）

完整目标仍未完成。本阶段补齐真实任务适配与对比起点，后续需要新候选、独立确认、工具/Harness效用及完整方案各项验收。

## 已实现并验证

- `adapters/amazon_worker.py`：独立初始化、主机持有的训练与评估、固定1000验证用户及99个唯一负例、因果/尾位置读出检查、数据/源代码/候选集hash、原始逐用户rank与checkpoint、训练曲线和显存/分配GPU时间。
- `adapters/amazon_study.py`：冻结配置、四GPU独立类别并行、训练预算预留与结算、修订前成本继承、按完整配对seed汇总。
- `adapters/amazon_evidence.py`：按实际完成顺序将真实实验、模型和观察导入图；原始产物进入内容寻址存储。未把一次胜出升格为科学机制成立。
- Amazon专项测试4项在PyTorch环境通过；原22项运行时测试通过。

## 协议

保留原项目四类别预处理比例，源为本机既有公开Amazon 2023 5-core last-out预处理。训练上限200epoch、每epoch200batch、每5epoch验证、patience20。每类别和seed独立初始化，公开主指标HR@10，各类别等权平均。保留原模板BPR的数学目标，用softplus形式实现数值稳定计算。

新协议把验证用户/负例固定在独立RNG上，不再随训练抽样RNG改变。这与旧报告的200用户验证不同，不能将新旧数字直接拼接。cuDNN使用原项目非确定性默认算法，不宣称位级可复现。

首阶段预算14400 GPU秒。开发seed42，预注册确认seeds17、83。测试集尚未用于本轮选模或终评。

## 首个配对结果（验证集，seed42）

| 指标 | 修复后的基线 | 既有RecHarness incumbent | 差值 |
|---|---:|---:|---:|
| 平均HR@10 | 0.50575 | 0.55350 | +0.04775 |
| 平均NDCG@10 | 0.3092938166 | 0.3517536113 | +0.0424597947 |

这只是既有方法的复测，不是本轮新方法收益，不是最终测试，也不是显著性结论。

初始适配器启用确定性cuDNN后，独立算子计时发现约80倍性能退化（约0.76 vs 0.0094秒/batch）。已核验并取消该批四个作业，保留取消产物和成本；另建r2冻结协议，所有对比臂重新训练。没有削减epoch或batch数。

24个任务已全部成功完成。连同前驱取消与诊断消耗，账本共记5838.4414 GPU秒，剩余8561.5586秒。所有实验已入图：图版本24、99条事件，hash链通过。

## 独立初始化确认（验证集，seeds17、83）

| 类别 | 基线HR@10 | 既有incumbent HR@10 | 差值 |
|---|---:|---:|---:|
| Movies_and_TV | 0.5490 | 0.5980 | +0.0490 |
| Industrial_and_Scientific | 0.4975 | 0.5285 | +0.0310 |
| Electronics | 0.4380 | 0.4545 | +0.0165 |
| CDs_and_Vinyl | 0.5210 | 0.6175 | +0.0965 |
| 等权平均 | 0.501375 | 0.549625 | +0.048250 |

两个确认seed各自的平均差值为+0.0360、+0.0605，均值超过冻结最小效应0.005。确认只改变初始化，仍使用同一验证用户；不是独立用户集或最终测试，两个seed也不足以证明统计显著性。该方法同时改变维度、正则、序列长度及上下文混合，机制解释仍为provisional。

新增 `python3 -m rec_harness.adapters.amazon_report STUDY`：检查协议、数据/源/worker/checkpoint/候选集hash、因果与尾位置检查结果，从逐用户rank独立重算指标，再分别输出开发和确认分区的配对结果。24份成功回执均已通过。缺失配对不会获得通过门槛。

恢复逻辑已补充Linux进程启动身份与会话存活检查，控制器锁释放不能单独证明worker结束。未观测到退出的worker阻止重启；缺失身份时保守拒绝恢复。失败重试使用新的计费ID并保留旧目录；同类别同seed的重试不算新增独立证据。真实子进程的超时清理、孤儿存活阻断、回执补记成本与幂等恢复测试通过。

## 固定前缀混合消融（已完成）

新增对照模型版本 `legacy_no_prefix`，源码与既有incumbent只有一行执行差异：`blended = 0.85 * x + 0.15 * context` 改为 `blended = x`。保留同样的掩码、参数、初始化、维度、正则、序列长度、数据、训练/评估协议与候选集。注册前通过CPU结构及梯度检查；注册内容及源码hash追加到事件源，原冻结study.json不改写。新接口使用model_versions/model_version和comparison_groups；旧arm/arms只作历史兼容，没有bandit或多臂调度。

对照独立训练四类别×seeds42、17、83，共12个任务成功完成。加上此前24次训练，共36次成功任务。全部成本（含前驱取消与诊断）为7213.2289 GPU秒，预算剩余7186.7711秒。

| 确认集类别均值（seeds17、83） | 无前缀混合 | 保留15%前缀混合 | HR@10差值 |
|---|---:|---:|---:|
| Movies_and_TV | 0.5855 | 0.5980 | +0.0125 |
| Industrial_and_Scientific | 0.5180 | 0.5285 | +0.0105 |
| Electronics | 0.4550 | 0.4545 | −0.0005 |
| CDs_and_Vinyl | 0.5910 | 0.6175 | +0.0265 |
| 等权平均 | 0.537375 | 0.549625 | +0.012250 |

两个确认seed各自平均增量为+0.01375、+0.01075；均超过冻结最小效应0.005。开发seed42的平均增量为+0.03275，其中Industrial为−0.007。正负切片全部保留，不把类别切片当作额外独立seed。按当前改进版设置，观察到前缀混合的增量收益；不能把相对原基线的全部+0.04825归因于该组件，也不将本次单一路径消融当作唯一因果分解。

注册时已有incumbent的全部seed验证结果可见；此次是固定对照的新训练，不是新独立用户集或最终测试。没有新Codex候选，没有统计显著性声明，也没有据此自动创建科学Claim。

`amazon_comparison.ingest_comparison`从独立排名审计导入3个按seed聚合的派生比较及12个类别观察，引用实际训练节点；派生比较不计作新训练或重复收费，重试不增加独立性。图达到版本39、163条事件，完整性通过。报告位置为实验目录下 `comparisons/legacy_no_prefix-versus-legacy_incumbent.json`；本地既有源码/指标出站授权仍待确认。

## 模型结构合同补充验收

新增探索性历史长度诊断：`python3 -m rec_harness amazon-history-diagnostic STUDY OUTPUT_DIR`，先保存固定分箱计划（1–5、6–20、21–50、51–128、129以上），再复核训练数据hash与配对rank；只导出聚合人数、HR/NDCG增量、hit胜/负/不变计数，不导出用户或物品记录。它不是新诊断工具的独立在线效用验收。真实输出位于 `history-diagnostic-v1/report.json`，hash为 `cadd2716b73fe007a710eef254539d1057b323fc3a10c65f9ad58e34d3e2bea0`。

在固定前缀消融中，Movies的21–50历史组58名用户，确认seed17/83的HR@10增量分别为−0.103448和−0.034483，不能支持“历史越长收益越大”的简单解释。其他类别/长度组方向不一致，尤其50以上历史的样本很少。该报告是在总体结果已知之后生成的探索性分析，两个seed也共享用户；不报告显著性、不据此改写现有确认结论。后续可提出长度条件化混合假设，但需新配对实验判别样本波动、优化差异和真实条件效应。

对两份冻结源码使用独立CPU合成输入探针（itemnum=127、batch=2、length=16），分别验证多处因果边界、前缀对未来embedding的梯度为零、尾位置打分、空历史形状/有限性、BPR训练参数梯度连通与有限性、padding embedding梯度为零。两模型全部通过。该检查只覆盖初始权重，不替代真实训练checkpoint验证。

后续已完成36份真实训练checkpoint的独立CPU合同复测，覆盖三个模型版本×四个类别×三个seed，全部通过。检查点使用weights_only加载，加载前后核验训练收据中的sha256，检查全部权重有限值并严格匹配state_dict；探针使用实际词表大小、合成序列和训练权重检查因果边界、尾读出、空历史及梯度。结果位于 `trained-contracts-v1/index.json` 与逐job报告；它证明这些合成检查下的正确性，不是终评或性能增益。入图时必须提供匹配的checkpoint文件并归档至CAS，模型身份包含checkpoint hash，不能与随机初始化模型或其他seed的权重混用。

每个模型记录68个模块调用，身份包含模型版本、源码hash、module_path和调用序号。直接张量对象传递被记录为FLOWS_TO；转置、残差、前缀混合等函数操作未被完整追踪，不能声称完整计算图。68个同位置对应均因64/48隐藏维度的具体张量形状不同而不具备直接形状兼容；它们只是待语义审查的类比，不自动允许插入。

实际模型/探针源码和报告已进入CAS；结构记录写入独立任务scope，主图达到版本26、101条事件，完整性通过。有效运行目录为 `model-contracts-v2`。首份探针报告受源模块导入时重置随机seed影响，已原样保留并标记superseded；v2在模块导入后显式重设记录的seed20260923。

## 隔离状态

主机不支持Landlock，且禁止新mount namespace。已实现chroot构建/降权原型，实际Codex工具调用能够确认宿主测试目录不可见。GPU chroot仍未通过CUDA初始化：驱动还依赖进程proc接口。`native/proc_maps_bridge.c` 是未验收的兼容性实验，不是已生效的生产隔离。

进一步用四个已审查合成GPU探针检查预初始化方案，均失败：CUDA预热后chroot并降权、仅降权、仅chroot，以及两者都不做但同样清理描述符。最后一组证明，关闭继承的pipe/socket描述符本身就会使后续CUDA分配失败；其余三组共同包含该操作，因此不能据此归因于chroot或UID切换。尚未识别具体必需描述符及其通信对端，不允许以保留全部宿主描述符作为生产修复。脚本进入CAS，进程/成本收据保留于 `gpu-isolation-diagnostic-summary-v2.json`；这不是生成模型的隔离验收。

后续按描述符类别细分：探针导入PyTorch前与导入后均无额外描述符；CUDA初始化之后新增设备、eventfd、pipe和一个未连接AF_UNIX/SOCK_SEQPACKET socket。v1分类探针在getpeername遇到未连接socket时提前退出，不能作为隔离结论，费用保留。修复记录逻辑后的v2对照中，关闭pipe的运行超时，单独关闭socket的卷积前后向成功，保留初始化IPC后chroot降至UID65534的卷积前后向也成功。结果缩小了失败原因，但尚未完成pipe端点与线程/网络/设备边界审计，不将其视为任意生成代码的安全沙箱。该预初始化路径不需要已搁置的proc_maps_bridge兼容实验。

组合探针 `gpu-close-sockets-jail-diagnostic-v1` 进一步验证：关闭socket、保留CUDA创建的pipe/eventfd/设备句柄，再chroot降权，卷积前后向成功且梯度有限。汇总为 `gpu-isolation-diagnostic-summary-v3.json`。截至这组诊断，累计预算支出7266.1567 GPU秒，剩余7133.8433秒；先前7213.2289是36次训练及当时诊断的冻结审计数值。

跨线程边界探针 `gpu-tsync-jail-diagnostic-v1` 已通过：三组pipe均在当前进程内持有读写两端，关闭socket及其他非允许描述符，启用seccomp TSYNC，再chroot降至UID65534。预先存在的Python线程不能创建socket，宿主数据目录不可见，GPU卷积前后向仍成功且梯度有限。`policy_sandbox.restrict_syscalls(sync_threads=True)`在无法启用TSYNC时拒绝继续；其参数依据[libseccomp官方头文件](https://github.com/seccomp/libseccomp/blob/main/include/seccomp.h.in)。该结果仍未覆盖GPU设备访问范围、恶意生成模型、真实训练和跨进程pipe所有权的完整审计。

单卡设备边界进一步通过两步合成验收：`gpu-device-fd-diagnostic-v1`关闭CUDA初始化时打开的设备1–7句柄后计算成功；`gpu-scoped-device-jail-diagnostic-v1`使用独立模板gpu-jail-device0-v1，仅包含/dev/nvidia0、/dev/nvidiactl、/dev/nvidia-uvm和基本字符设备，关闭其他GPU句柄，TSYNC过滤后chroot降权，未分配设备路径不可见且卷积反向传播成功。构建器include_gpu=true现要求显式gpu_devices列表，拒绝缺省、重复、负数和非整数，不再通配所有宿主GPU节点。这不等于验证所有驱动ioctl权限或真实NextItNet训练兼容性；后者仍待验收。

随后 `gpu-nextitnet-training-smoke-v1` 完成真实训练兼容性验收：只加入冻结legacy_incumbent源码、冻结worker与Movies的train/validation文件；预初始化CUDA后筛选句柄、TSYNC过滤、单卡chroot、UID65534降权，执行1epoch×2batch并评估64名验证用户。训练、验证rank、检查点及7项产物归档成功，峰值显存246362112字节。这是缩短协议的兼容性作业，未进入36次正式训练的对比统计，也未使用test。`gpu_descriptors.py`将句柄筛选拆为可测试规则：拒绝初始化前继承的非标准流句柄，保留指定GPU、eventfd与完整本进程pipe对，关闭socket、其他GPU及其他文件；目前仍需正式启动器接入、进程外pipe所有权和攻击面验收，不能称为任意生成模型沙箱已完成。

项目内 `gpu_launcher.py` 已实现上述预初始化及降权流程。`gpu-launcher-nextitnet-smoke-v1`经该启动器重跑相同小型作业成功，checkpoint hash仍为 `189845046448af6350b1867cc8811424dd7c133eff4ce7b662817498052bebfa`。入口脚本的hash/UID/单卡清单/符号链接/可写目录所有权由启动器校验；6项设备、句柄及入口拒绝测试通过。启动器目前供已审查代码验收使用，独立作业暂存收集及启动器专属攻击测试仍待接入，不能将其描述为生成候选已获准执行。

`gpu-launcher-boundary-probe-v1`随后通过10项专属边界检查：非特权UID、合成环境变量移除、宿主canary与proc不可见、其他GPU路径不可见、socket/fork/setuid(0)/mknod被阻断、root所有入口不可改写。探针代码位于tests/gpu_boundary_probe.py，未读取真实凭据。`gpu_outputs.collect_outputs`仅在控制器终态收据存在时收集扁平训练输出，通过目录fd及O_NOFOLLOW打开文件，拒绝符号链接、硬链接、FIFO/目录等非普通文件，限定文件数/总字节并检查读取期间是否变化。对应拒绝测试通过；控制器独占进程日志与每作业独立目录仍是调用前提。

`gpu_job.run_reviewed_job`已封装独立目录/UID暂存、输入hash与只读副本、启动器调用、限额收集及CAS恢复。真实验收 `isolated-jobs/gpu-per-job-staging-smoke-v1` 使用UID22827900成功完成相同缩短协议，归档7项产物，checkpoint hash与此前两次兼容性作业一致。累计已花7292.5250 GPU秒，剩余7107.4750秒。该作业不属于正式性能比较；Amazon调度执行版本接入、完整运行环境锁、模型与校验器进程分离仍未完成。

Amazon调度执行版本已接入并通过缩短协议整链路验收：`amazon-adapter-isolation-smoke-v1`使用注册版本`execution-62a024564448a0397dd01c75`，前置合同检查、独立沙箱训练、受限收集和后置checkpoint检查全部成功。它在独立诊断目录执行，未覆盖原36份收据，不计作模型增益。新收据/request均记录execution_id；旧收据视为direct-reviewed-v1。代码身份改变需重新注册，已完成作业不能以其他执行版本覆盖，配对报告与聚合不混用版本。累计支出7306.1490 GPU秒，剩余7093.8510秒。完整环境字节锁、校验器独立进程和生成候选准入仍待完成。

已将准备好的Python/PyTorch及动态库用户态文件归档至CAS：`gpu-runtime-bytes-v1.json`含67907个路径，逻辑字节11974243249（含重复库别名），清单hash为`ee9fece635a36eb31ad17c4367acef7c5efd489281c5759b7c20409309db87be`。内核版本和NVIDIA驱动版本文件记录为主机指纹，作业输入/输出及dev/proc不进入归档。这不追溯证明旧36次训练当时的所有环境字节，也尚未强制绑定GPU预初始化时的宿主Python加载路径。

上述清单已从CAS完整恢复至独立目录gpu-runtime-restored-v1；去重内容5758786779字节，恢复记录为gpu-runtime-restore-acceptance-v1.json。加载器位置须取归档的实际CPython安装子目录，而非假定.python/bin；初次错误路径尝试未启动解释器。

指定恢复后的CPython、PYTHONHOME与site-packages并使用-s/-S后，Python、PyTorch和NumPy均从恢复目录加载成功（torch 2.7.0+cu126），证据为gpu-runtime-import-acceptance-v1.json。本次只验证用户态恢复与导入，未调用GPU或重做模型训练。

当前GPU对比只执行已核验的既有两份源代码。未将失败隔离环境用于自动生成候选。完整权限/Hidden隔离这一项仍未完成，不能因路径白名单或一次布尔检查就标记完成。

## 本地运行定位

实际路径/进程句柄保存在工作区私有运行目录，不提交原始数据或权重：

- 主实验目录名称：`amazon-nextitnet-v5-20260923-r2`
- 前驱取消批次：`amazon-nextitnet-v5-20260923`
- 下一步：完善Agent隔离与真实能力缺口/候选提案，再运行新改造及必要机制对照。向Codex发送本地模型源码和seed42汇总指标的调用被自动审批拒绝，具体载荷出站授权问题已交用户确认；尚未重试该调用。

### 运行归档接入作业准备（2026-09-24）

GPU 作业准备接口新增可选 runtime_archive：按 CAS 恢复 jail 内 runtime，核验
recipe/归档摘要/内核及驱动身份，在请求、jail manifest、完成回执关联归档 hash。
回归用例删除源运行文件并禁止所有 live runtime copy，仍可完成归档恢复；错误
recipe 和缺失 artifact store 在创建 jail 前拒绝。设备节点和 chown 在该用例中
mock，不能据此声称真实 GPU 启动通过。此前完整 runtime 的恢复/import 验收仍
独立有效；chroot 前 launcher/CUDA 加载环境尚未绑定，此轮未重跑正式训练。

### 实际加载库身份审计（2026-09-24）

新增 runtime_loaded 模块及 GPU launcher 归档门控，核验进程文件映射身份与归档
字节。真实 Torch CPU import 后的两次校验均拒绝：沙箱及提权进程中 stat 与
/proc/self/maps 的设备/inode 不匹配；进一步只读检查 /proc/self/map_files 报 EPERM。
证据为 study 根目录 loaded-runtime-cpu-audit-v1.json 和
loaded-runtime-host-cpu-audit-v1.json。没有放宽身份条件，未启动 GPU 或增加训练成本。
单测覆盖实际 mmap 文件的字节修改、路径替换、未归档映射、删除映射和转义路径；
单测明确规范化夹具设备字段以避开当前环境的 stat/proc 差异，不作为真实加载身份
通过证据。完整运行锁仍未完成，需能验证实际映射对象的执行环境或恢复方案。

### 独立 CPU 探测边界（2026-09-24）

新增 cpu_launcher 和 adapters/isolated_contract。首次 isolated-cpu-contract-v1
因 OpenMP 在线程创建受限后启动线程而失败；未抹除失败回执。检查器新增 --threads
参数（原默认 2 保留），隔离路径固定 1 后 isolated-cpu-contract-v2 成功。
原模型源码 hash d32d162182c2c156dd24573819361d513e49960f44429ce6f005afa21ba4d553，
报告 hash 4e808355981305111786b1d23512c097680931e93b198f0041440e1eae914315；
报告及源码/检查器快照已进入原 study 的 CAS，未改变正式训练结果或消耗 GPU 预算。

isolated-cpu-boundary-v1 的 10 项实测检查全部通过，边界结果 CAS hash
2bc5d5f90294ab4861d8b06a7faf8fddd87646131bd4d03a874f2f9a53dc137a。
夹具主动 exit(0) 且不给检查报告，控制器正确返回 invalid、candidate_admitted=false。
这些证据只支持隔离探测边界，候选和检查器同进程的报告真实性问题尚未解决，不能
据此自动准入新的 Codex 模型。全套核心检查 98 项，84 通过、14 跳过。

### 可重复生成的验证报告（2026-09-24）

新增 `amazon-summary` CLI。实际输出位于 rec-harness-runs/amazon-validation-summary-v1，
报告 hash eed7a4a0754f719b5333b4eeee3c53fd47193e0bcf133f20d3593cc9f2dc5313。
重新审计的确认整体 HR@10 差值 +0.04825、前缀对照差值 +0.01225；保留 Electronics
确认均值 -0.0005 及其他负向种子切片。完整账本计费或保留 7306.148984782398 GPU 秒，
剩余 7093.851015217602。报告没有新增训练、Hidden 结果或新候选收益。回归检查覆盖
缺失配对、排名与指标伪造、正总体效果下保留负向切片。

### Amazon 研究交付预注册（2026-09-24）

新增 amazon-plan CLI 和 adapters/amazon_plan.py，将 GapCard/MethodCard/Proposal
及候选/机制对照源码固化为事件与 CAS。开发和确认矩阵按冻结类别/种子生成，保留
基准配对身份与成本规则；源码只读归档不执行。测试确认不改 study、不自动增加
可执行模型、不扣预算、幂等、不可改写、拒绝协议/种子/预算/对照漂移，以及按源码
识别事后换名注册、拒绝跨计划复用模型名称。

本次仅使用合成测试夹具，未向真实研究图追加虚构 Codex 提案，未宣称新候选准入
或完整自动循环已经运行。尚需在真实 Codex 交付到达后接入代码准入、开发执行与
冻结选择后的确认释放；此前出站审批仍未收到用户授权。

### 真实 Codex 缺口角色与后续审批（2026-09-24）

用户明确批准 model.py/context.json 两份文件发送后，首个真实 Amazon evidence 角色
已成功运行（amazon-codex-research-v1/evidence），47.87488627433777 秒；输入
51640 token（其中缓存 30080），输出 1110，reasoning_output_tokens 137。资源原始
字段保留，不将代理 token 成本伪装为 GPU 秒。源码、上下文、响应和资源回执 CAS
引用已追加 amazon_agent_handoff_received 事件，事件数 241、graph_version 76。

角色提出历史信息与表示缩放的竞争解释及 scale-only 对照，但误把源码注释中
0.10→0.15 的旧改动当作新改进，且因上下文未给标准 ID 返回 unknown protocol。
控制器明确保留这些缺陷，将结果标为 agent_hypothesis_requires_correction，未准入
或训练。准备给 hypothesis 的纠正上下文位于 amazon-codex-handoff-review-v1。

后续调用被自动审批拒绝，理由为原两文件授权不包含新增角色缺口分析与控制指令。
调用未执行，已向用户请求本任务后续衍生材料传递范围授权；没有通过其他路径发送。

### 新 Codex 候选进入开发训练（2026-09-24）

用户已进一步授权本任务后续 Codex 角色传递原批准材料及衍生缺口/提案/候选代码、
协议预算指令与开发聚合结果；原始交互、ID、逐用户排名、checkpoint、最终测试及
凭据持续排除。hypothesis、evidence-corrected、experiment 三角色均成功返回，
记录在 amazon-codex-research-v1。修订 GapCard 的 cheapest test 提及另一种 scale-only
对照；本次绑定的方法提案选择匹配当前项系数的 scaled-incumbent 对照，两个建议
不得混称为同一实验。Proposal 明确保留范数/优化效应这一未排除的竞争解释。

正式提案 nextitnet_prefix020_vs_scaled_incumbent_v1 已在结果产生前注册。候选源码
20b585b911c99fb6d55e40a546960480daed9139e2d2f73d785bd63a93df0488；对照源码
5479abcb5b7adbc9af490f56962ea65be4ff41ef4ade9953952fdc7d8a860441。两者均经控制器
逐字节核对为已审阅 incumbent 的单行替换，CPU 结构/梯度合同通过；其余任意改动
不在准入范围内。开发训练为两版本×四类别×seed42，原协议不变，原账本继续收费，
提案额外上限 6000 GPU 秒。尚无开发结果时不宣称收益，确认种子未释放。

独立 CPU 评分验收目录 independent-score-acceptance-v1：Movies incumbent seed42
1000 个用户的重算排名与归档完全一致，HR@10 0.601，正样本同分用户 0；验收报告
hash c1cd055d4d2d045fcdac3282762c78667615428ed02880dbc5593b4bdac2ffa3。
模型进程不接收用户 ID 或正样本位置，控制器从返回分数计算指标。仅支持此次已有
checkpoint 的兼容性结论，不把它算作新实验收益或完整攻击防护证明。

### 首个 Codex 候选开发阶段结束（2026-09-24）

8 个新训练均成功且前后 CPU 门控通过。seed42 四类别平均 HR@10：incumbent
0.5535、prefix020 0.5540、scaled-incumbent control 0.54725。候选相对 incumbent
+0.0005（0.05 个百分点），低于 minimum_effect 0.005；相对对照 +0.00675。
Movies/Industrial 相对 incumbent 各 +0.003，Electronics/CDs 各 -0.002。
完整开发读数 hash f755ca5cface85576700f1fc680fb3ae54611a6acbaacee24286de0df11dfe41。
没有释放确认，不将开发阈值失败解释为机制被反驳。

本轮新增分配 GPU 时间 1330.659189194441 秒；原账本总计 8636.808173976839，
剩余 5763.191826023161。独立 CPU rescore 覆盖三个模型版本×四类别的 12 个
checkpoint，各 1000 用户，所有排名与归档一致且无正样本同分；完整索引位于
independent-development-rescores-v1/complete-index.json。不导出用户、候选集或
逐用户分数给研究角色。8 个训练事实及开发研究图已入图，graph_version 85；
评估角色已返回 discard / unknown，负责人和知识角色处理尚在继续。

首轮 evaluation、lead、knowledge 均已完成。最终 discard / unknown；知识角色的
provisional Claim 引用开发聚合 Experiment 和两个负向 Observation，原事实保持
不变。控制器追加 DecisionTrace 和 amazon_research_cycle_completed，图版本 86。
finalize_cycle 重复调用返回相同记录；实际 plan_guard 仍拒绝 seeds17/83。
本轮自包含读数位于 amazon-codex-research-v1/README.md；cycle identity 为
906a8722f49e120d5b3216370b88c6ed50ffcd39a27417a78058599d8da1b87a。

### 第二轮线性前驱加权候选（2026-09-24）

Codex evidence/hypothesis/experiment 三角色提出并实现 nextitnet_linear_recency_v1。
源码 hash 67d70cf54e261db5794ee0ff08dbad04de57546ff6053d44cb3764a0a9ce4747；仅把
前缀均值替换为按有效前驱位置 1..n 加权的均值，混合比例保持 0.15。既有 incumbent
直接作为配对的单因素对照，明确复用旧证据及历史成本，不重复训练或计数。

数学报告 e1beae543a6fdc74710d888b4f60bb11c68ee0204b461cee585ba13b5c7963b5：
显式循环与向量实现误差 0，手算标量、padding、0/1前驱等价、因果、解析梯度与
双精度 gradcheck 均通过；完整 CPU 模型合同通过。检查器源码、报告、计划和成本
估计已归档到 CAS，引用见 amazon-codex-research-v2/pretraining-artifacts.json。

提案在新结果前注册，额外预算上限 4500 GPU 秒；按首轮最大单次耗时×12×1.5
得到经验估计 4114.655192792416 秒，仅为估计，不承诺完成最坏情况。原研究余量
5763.191826023161 秒继续生效。四类别 seed42 开发训练已启动，尚未生成收益结论；
未释放 seeds17/83。核心验证 118 项，104 通过、14 跳过。

补充共享初始权重的完整模型检查：双向 strict state_dict 加载、参数形状/键/数量
不变、0/1有效前驱时输出相同、填充输出为零，最大等价误差 0。报告
amazon-codex-research-v2/shared-state-contract.json；此补充检查在开发训练进行中
完成，不写成提前执行的检查。此前数学合同和单模型 CPU 合同已在调度前通过。

新增 amazon-run-phase 入口，将已注册计划转为新模型训练任务，先审计全部既有
配对对照。真实第二轮 prepare_phase 返回仅 nextitnet_linear_recency_v1 为新任务、
legacy_incumbent 为复用对照；confirmation 准备在任何训练前正确拒绝，未另起作业。

第二轮四类别 seed42 开发训练及独立 CPU 重评分已完成。候选 HR@10：Movies
0.613、Industrial 0.528、Electronics 0.437、CDs 0.623；宏平均 0.55025，
对 incumbent 0.55350 的差值为 -0.00325（-0.325 个百分点）。只有 Movies
正向，另外三个类别分别 -0.003、-0.008、-0.014。未通过 >0.005 的开发门槛，
不释放确认。独立重评分四个候选 checkpoint 均无排名差异及正样本同分，
索引 independent-linear-recency-rescores-v1/index.json；不增加独立训练证据。

本轮新增 GPU 时间 725.270769640803 秒，研究累计 9362.078943617642 秒，
剩余 5037.921056382358 秒。开发读数 hash
73b6397a11bfd8d1428cf839d11f41f554b8a05929970de70a222f3250747f44。
训练事实与开发研究子图已入图，版本 91；evaluation 返回 discard / unknown，
lead / knowledge 尚在处理。新增阶段入口后的核心验证仍为 118 项，104 通过、
14 跳过，public-tree 边界检查通过。

第二轮六个 Codex 角色现均成功完成，最终决策 discard / unknown。知识图保留四个
类别的完整结果（含三个负向类别），仅添加 provisional 解释及控制器 DecisionTrace。
图版本 92、事件 311；finalize_cycle 重复调用返回同一记录，确认入口实际拒绝。
cycle identity：99731c45ca533fa3d1c930ff0d98b2a6162b8505e90cddd1e293533a10808154。
详见 amazon-codex-research-v2/cycle.json 与 final-integrity.json。整个优化目标仍未完成。

### 第三轮预算检查与授权续接（2026-09-24）

余弦加权候选 nextitnet_cosine_context_v1 的源码和 CPU 数学/完整模型合同通过。
合成 GPU 成本2.620810277760029秒；完整兼容开发耗时更新后，12次训练经验投影
6449.2602551338秒，超出旧提案4500秒，已写入不可变执行阻断并实测调度拒绝。
没有该候选的推荐性能结果。详见 amazon-codex-research-v3/README.md。

用户明确授权追加4000 GPU秒及新campaign。新研究
amazon-nextitnet-v5-20260924-r3 累计上限18400，继承已花费9364.699753895402，
可用9035.300246104598；旧研究调度封存，12份 incumbent 既有配对结果经审计复制，
不当作新独立实验、不重新收取成本。训练协议/源码/数据hash保持一致。新提案预算
由 Codex 重新登记，旧提案不改写。续接的激活边界与双重分叉禁止已有故障测试；
新研究真实激活重复调用幂等，旧研究开支入口实际拒绝。

另外修复研究闭环的图与完成事件分开提交问题：现在单事务提交，并通过提交前
故障回滚及提交后JSONL恢复测试。CPU检查入口将相对路径冻结为绝对路径；失败
准入重试保留原目录。最新核心验证124项，110通过、14跳过，public-tree通过。

预算续接后的余弦候选四类别开发训练已完成：Movies 0.600、Industrial 0.540、
Electronics 0.439、CDs 0.625；宏平均HR@10为0.55100，现有模型0.55350，差值
-0.00250，未达>0.005门槛。新增训练679.965917527676 GPU秒，累计已花费
10044.665671423078，剩余8355.334328576922。没有释放确认；训练协议未缩短。
四个候选checkpoint独立CPU重评分均无排名差异及正样本同分，索引
independent-cosine-rescores-v1/index.json。当前旧+新不同训练共52次成功，复用的
12个既有比较不重复计数。新研究开发图版本17，evaluation/lead/knowledge处理中。

六角色职责的开发闭环现已完成，最终 discard / unknown，保留现有模型、不释放确认。
evidence/experiment 使用已明确标记的原调用；hypothesis/evaluation/lead/knowledge
为新调用。第一次 knowledge 把文件路径放入图证据引用，严格校验拒绝且未改图；
失败调用和拒绝原因保存在 knowledge-rejected-refs，修正后的真实 Codex 响应通过。
最终知识补丁和 DecisionTrace/完成事件以同一事务提交；图版本18、事件44。
重复 finalize_cycle 返回同一记录，实际 confirmation 入口仍拒绝。
cycle identity：6e8d8ba7567465194814ebed4a686fdc6a1c1c3639a2b4b125b5a1393054e5a5。
详见 amazon-codex-research-v4/cycle.json 和 final-integrity.json。整体方案及进一步模型优化仍未完成。

历史长度诊断现可显式选择候选与阶段，例如：

```bash
python3 -m rec_harness amazon-history-diagnostic STUDY OUTPUT_DIR \
  --baseline legacy_incumbent --candidate CANDIDATE_ID --partition development
```

该模式只审计开发初始化，不要求候选拥有确认训练，适用于开发阶段淘汰的候选。省略 `--partition` 保留原有开发和确认两阶段行为；包含确认阶段的报告必须留在主机，不传给研究角色。两个模型须使用一致的执行版本和配对评价输入。分箱使用完整训练历史长度，并非截断后的模型输入长度；观察到的差异不能单独证明长度的因果作用。

Amazon 自动研究配置可增加 `"history_diagnostic": true`，在候选开发训练和独立复评分之后、evaluation/lead 之前生成开发历史长度诊断。默认关闭以保持既有运行行为。报告先归档到控制器 CAS 并提交 `development_history_completed` 事件，后续恢复验证报告、模型/协议/阶段、诊断源码及输入哈希；未完成目录保留为 interrupted 后重新计算，不产生新训练或推理。对角色仅投影类别、开发 seed、分箱人数、聚合指标和命中胜负，排除用户 ID、逐用户排名和输入路径。诊断不新增 Codex 角色调用，确认阶段不能通过该配置进入角色上下文。该诊断是确定性探索性切片，不是自主工具晋升或机制确认。

固定 checkpoint 旁路评分支持 `--bypass-blocks 0 1`，与旧 `--bypass-block 0` 互斥；仍仅允许开发 seed。新增 `amazon-block-interaction REF SINGLE_A SINGLE_B PAIR --output REPORT.json`，要求四个目录构成完整参考/单块/单块/双块条件，重新读取私有主机评分输入计算排名，并检查模型、checkpoint、协议、评分源码、用户、实际训练历史和候选集合一致。输出每个指标的 `M(ab)-M(a)-M(b)+M(reference)`；它是固定权重的指标非加性，不足以认定冗余、机制确认或删除重训收益。重复/非法块索引会拒绝；部分 hook 注册失败也会清理本次 hook，保留原有 hook。双类别真实影子验收已在独立冻结源码目录启动，结果待定。
