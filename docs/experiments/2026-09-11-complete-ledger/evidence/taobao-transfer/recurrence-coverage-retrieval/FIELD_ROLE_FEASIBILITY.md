# 第二轮 K/V 字段分工：实现可行性审计

结论：拟定四组在原 `ctx_q4` / `explore_dual_no_feedback` 上均可实现为零新增参数的局部输入替换；当前没有必须因全局数学退化而撤换的组。最直接的位置是 `unified_qformer.layers[1].cross_attn` 的第二次调用，而非 prediction 层 `layers[2]`。原 attention 已分别接受 `keys` 与 `values`，不必重写 SDPA、交互乘积或 residual。

这只是 [NEXT_STRUCTURE_AUDIT.md](NEXT_STRUCTURE_AUDIT.md) 候选二的条件性实现设计。当前 coverage 轮的 16 配对尚待根代理完成审核/归档；本文件不选中或派发下一轮，不创建 handoff，不把已有负点估计当作已证明全部退步。新假设以原 q4/q8 为父，不叠加本轮 coverage score bias。没有加载数据数组/checkpoint、构造模型、运行 forward/优化器/GPU 或改全局 memory。

## 原字段、目标槽和投影列

[FeatureMap.load](../../runtime/UniRank/unirank/features.py:51) 保留 JSON 字段次序；[原构造](source-snapshot/QFormerCross8.py:822) 按 `source=item/action` 形成 `sequence_features`。现有两父配置 `embedding_dim=16`，feature map 无这些字段的维度覆盖，因此拼接为：

| 历史字段 | source | `sequence_projection.weight` 列区间（左闭右开） |
| --- | --- | --- |
| item_id | item | [0,16) |
| cate_id | item | [16,32) |
| campaign_id | item | [32,48) |
| customer_id | item | [48,64) |
| brand | item | [64,80) |
| price_bucket | item | [80,96) |
| action | action | [96,112) |

`W.shape=[384,112]`、`b.shape=[384]`，`W_I=W[:,0:96]`、`W_A=W[:,96:112]`。实现应从原 `sequence_features` 与每字段实际 embedding 维度推导列偏移，再严格核对此布局；不要根据 item_dict 的运行时遍历顺序猜列，也不要把 item_index 等 meta 字段纳入投影。[原 token 构造](source-snapshot/QFormerCross8.py:1083) 对每字段都取 `[:, :-1, :]`，目标 item 仅由非序列路径的 `[:, -1, :]` 使用。

记 `x_I` 为六字段96维向量、`x_A` 为16维 action embedding，`m` 为原历史有效 mask，`p=position_embedding_scale*position_embedding[position_ids]`，其中 `position_ids=(cumsum(m.bool())-1).clamp_min(0)`。三个历史表示的定义是：

- 原完整表示 `H=m*LN_384(W_I*x_I + W_A*x_A + b + p)`。
- item-only 表示 `H_I=m*LN_384(W_I*x_I + b + p)`。
- action-only 表示 `H_A=m*LN_384(W_A*x_A + b + p)`。

这里完整 H 的公式只用于说明：实际首条路径继续调用原 `_build_sequence_tokens` 的一次完整112列投影，不能拆成两次 GEMM 相加后声称 bitwise 不变。备用表示用原 weight 列视图做一次 `F.linear`，同一 bias 加一次，再复用原 `sequence_input_norm` 的全部 affine 参数和 eps。不得新建 Linear/LayerNorm、复制 Parameter 或重新编译优化器。

`H_I/H_A` 不能通过从 LN 后的 H 减掉某字段来获得：LN 的均值和方差取决于全部384维，字段移除会改变归一化耦合。`item-only/action-only` 仅描述当前事件直接输入的字段；两者仍含共享 bias、位置与 LN affine。它们不是只包含“物品语义”或“动作语义”的纯表示，也不保证归一化后表示范数相同。

## 四组及局部可辨认性

[SDPACrossValueAttention](source-snapshot/QFormerCross8.py:178) 的路径是 `Q=q_norm(w_q(q))`、`K=k_norm(w_k(keys))`、`V=w_fi(values)`，随后原 SDPA，最后 `w_qi(q)*attended_features` 和原 `w_v_pair/w_o`。注意实际 V 投影叫 **w_fi**；`w_v_pair` 位于交互乘积之后，不能把它当作 V 输入投影。

| 组 | 第二 work-attention 的输入 `(q,keys,values,mask)` | 固定权重/同前缀下直接改变 | 静态可行性与局部退化 |
| --- | --- | --- | --- |
| K=item-only, V=full | `(q2,H_I,H,m)` | w_k及后续k_norm/概率；w_fi继续原H | 可行。1条有效历史时概率只能为1，K改动不能改变理论加权输出。 |
| K=action-only, V=full | `(q2,H_A,H,m)` | 地址依赖历史动作与位置；V仍完整 | 可行。若某行备用K完全相同则该行理论均匀读取；当前保留位置，不能据action类别少预判全局均匀。 |
| K=full, V=item-only | `(q2,H,H_I,m)` | w_fi读入内容；Q/K/概率路径原样 | 可行。即使有效长度为1，值变化一般仍改变输出。 |
| K=full, V=action-only | `(q2,H,H_A,m)` | w_fi读入内容；Q/K/概率路径原样 | 可行。值依赖动作与位置，项目相关性仍可经Q2、原K概率和w_qi表达。 |

四组都保留 `q2=(e+z1)/2` 的原对象、第一轮完整 H 读取和 `y1` 不反馈设计。[q4](source-snapshot/QFormerCross8ContextRead.py:13) 在 `context_reread=False, attention_position=end` 下与 [q8 no-feedback 分支](source-snapshot/QFormerCross8RecurrenceExplore.py:35) 都是入口一次、work两次、prediction两次，最后 post attention 一次。局部更改 z2 后，y2 与最终预测自然会变化；“保持 interaction/residual”指它们的直接输入/权重/公式未被替换，不表示整个交互输出仍相同。

没有已证实的必要换组：17个 action 编码中0是padding、1–16为真实动作或组合，`exposure=6`，不是四个标签各自一列。[词表](../../data/Taobao_Action/meta_data.json:129) 与100个有效事件序号位置通常允许多个不同 H_A。只有额外移除/置零位置、同action全行、退化权重等条件才可能使 action-only K 常量；这不符合当前保留位置的拟定设计。若将来离线 fixture 显示实现错误地抹掉位置、LN collapse、切错列导致恒定表示，应停止修正实现/设计，不能未经观察预设另一组填补预算。真实相同K/V导致的局部null须保留，不能挑掉这些行。

四组比较不能证明全模型不依赖被移除字段：Q2已由完整第一轮形成；K-only组保留完整V；V-only组保留完整K。共享参数训练后，第一轮、Q2、梯度、global clip与PSGDKron状态都会随训练改变。零新增参数只消除新增参数布局这个因素，不证明优化器轨迹相同。q4/q8各对自己的原模型，不作合并父结构因果判断，也不把query slot视作任务标签。

## 最小实现位置及恢复

推荐作用域设计：父 forward 开始创建一个普通Python上下文；覆盖 `_build_sequence_tokens` 时先调用原方法得到原 H，再从本次已计算的 `item_embedding_dict` 构造该组唯一 H_I 或 H_A，并仅保存在此上下文。embedding lookup 不重复，full H 不覆盖。随后在 `layers[1].cross_attn` 注册临时 forward-pre-hook，计数第一次原样返回、第二次仅替换tuple中的 keys 或 values，保留同一个 queries、未替换的 H、mask。原 [RecursiveCrossValueLayer.forward](source-snapshot/QFormerCross8.py:236) 仍做原 residual，与 attention 源码完全复用。

上下文应校验恰好两次 work调用、原 `(keys is values is H)` 路径、H_alt形状/设备/浮点dtype和配置匹配；退出 `finally` 移除 hook、恢复任何临时方法、清除 H_alt/item_embedding 引用。不能保留跨 batch 的 Tensor cache，不能 detach 第一轮或备用投影分支梯度。parent构造继续保持参数对象、名字/顺序、state、CPU/CUDA初始化RNG和dense/sparse布局；仅挂无Tensor常驻的配置/计数。gradient checkpointing/model compile仍关闭，原优化器内部编译保持。

最保守实现会重复一次 `position_ids/position_embedding` gather，因为原 token builder 不返回它们；这增加额外大张量开销。若后续为省激活要将原builder有限复制为“共享一次position gather、分别构造full/alternate”，必须先证明原full输出/梯度/RNG bitwise及原操作顺序保持，不能边训练边换实现。不要通过 `H-full_projection` 反推位置来省gather，那经过LN且数值不等价。

## 泄露、mask与精度边界

[collator](../../runtime/UniRank/unirank/pytorch/dataloaders/unirank_dataloader.py:525) 用相同 seq_len 对 item/action 历史取前缀并pad；最后单独追加目标item，目标action列初始化为0且只给 `[:-1]` 填历史action（550行）。原 token builder 又剔除每字段最后槽。`get_inputs` [344行](../../runtime/UniRank/unirank/pytorch/models/rank_model.py:344) 不把四标签加入模型X；原q4/q8不请求task-specific multi_masks。新分支必须沿用历史 `[:-1]`，既不读取当前 labels，也不将目标action占位槽或multi_masks作为额外输入。action词表含历史buy等合法观测，不等于当前标签泄露。

这是既有seq_len/已冻结数据合同下的源码边界，本次未遍历真实行，不能独立重新证明数据生产阶段每行前缀的时间因果性。不得因为知道目标action为0就把目标item槽加回H，也不得重新从当前四标签编码action。

有效mask仍由历史item索引>0构造，不能换成action!=0或“保留字段非零”。未知item字段0与真实padding不同。备用H在LN之后乘同一有效mask，保持mask孔洞与pre/post-padding下的有效事件序号；不能用物理`arange(L)`替换cumsum序号。原 w_k/w_fi 的bias会使padding投影不一定为零，但原bool key-mask继续隔离它们；全padding时需保持原SDPA及后续 affine/residual行为，不手动抹掉最终输出bias或临时放开mask。物理L=0只应按原stage实际支持范围处理；正常空历史是L=100的全padding行，不必把二者混为一谈。

备用token的投影要在原模型amp上下文中执行，保持dtype转换点：F.linear通常在CUDA autocast下产出BF16，而FP32 position参数相加及LayerNorm可能使H保持/回到FP32；attention的per-head q_norm/k_norm又可为FP32，w_fi输出可为BF16。这种混合不能被“所有Tensor必须同dtype”守卫错误拒绝，也不能为了统一dtype而把原H/Q2或全部position参数cast。新H_alt应匹配**原history输入H的实际dtype**，实际SDPA交给原原生autocast；BF16/FP32核验必须看真实CUDA tiny路径，纯CPU FP32无法替代。更改GEMM列宽会改变舍入与内核路径，备用表示只能与预定实现的数值oracle按事先定容差比较，不能要求 split-GEMM与full-GEMM重组bitwise一致。不得在本分支静默启用TF32或改变其他全局精度flag。

## 额外计算及所需的有限验证

每组只构造一个备用history张量，不同时构造H_I/H_A；不增加SDPA次数、历史事件数、模型参数或训练seed。增加一遍选择字段的 projection + 原position加法/gather + LN384 + mask及其反向；替换后的w_k或w_fi仍为原次数。以batch8192、L100、D384计算，item分支额外线性投影约30.20G MAC、action分支约5.03G MAC，均是每批前向的解析乘加计数，不是实测时长，也不含反向、gather/LN及其他操作。

一个 `[8192,100,384]` 张量按BF16为600MiB，按FP32为1200MiB；当前位置加法/归一化预计需要考虑FP32，而不能统一按600MiB估算。训练图还可能保存projection输出、position gather、norm输入/统计和梯度；单个张量尺寸不能当作新增峰值显存上界。item/action分支的额外FLOPs不同，q4/q8额外history构造量相同但原attention费用不同。若之后选择执行，须分别报告真实native smoke峰值与11模型同批延迟，不能用零参数或action低维直接预判通过成本门槛。

进入实现前应固定最小fixture合同：七字段顺序/列切分与目标槽剔除；全字段identity备用路径不启用时父构造参数/RNG/full H及输出exact；四组每次只第二K或V输入变化，首轮输出/Q2/w_qi/residual/mask不被直接改；item/动作列与bias/position/LN共享且梯度回原Parameter；0/1/多事件、mask孔洞/左右padding、未知类别0、相同action不同位置、坏shape/nonfinite、异常与重复batch清理。K-only单有效事件的理论null应作为解释性toy；V-only不要求这个null。CPU tiny公式验证后若获选，再按既有边界做真实CUDA BF16 tiny格式/梯度检查；本审计未运行这些验证，也不把未测要求写成已通过。

## 与已做结构的区别及执行条件

- 淘宝 [MemorySourcesStage](../recurrence-memory-sources/QFormerCross8MemorySources.py:16) 的work_entry拼接 `[H,e]`，另外三类在prediction处拼 `[z2,H/z1/e]`，会改变事件数/读取来源；本提案保留L/m并按同一事件分开K与V字段。[原八组均未超过同q基线](../recurrence-memory-sources/RESULTS.md)。
- KuaiRec [HistoryWriter](../../model_zoo/QFormerCross11SourceUpdate.py:9) 用新attention/norm/scale写回H，后续K/V一起改变，含新增参数和query/history条件；本提案无writer、无跨事件更新、无工作/预测状态注入H。[round13负结果](../../reports/round13/interpretation.md) 限制一般写回假设，不构成这四种字段分工已测的证据。
- 两本episode账本（本次分别195/233条）与上述近邻源码中未发现相同四种第二轮字段分配训练。字符串搜索命中的 `interaction_only`（淘宝131/135条）是query correction的w_qi路由，不是action字段-only。此为限定于现有记录/已审源码的查重，不声称所有可能实现均已检索，也不把结构新颖当作收益证据。

最终可行性判定：保留四组作为可审查设计，每个父模型四组共八组，但不据本文件启动。必须等当前coverage 16配对及结果一次归档关闭后，由根代理结合科学价值与资源选择；若实施仍对同q原循环/最佳单轮、全四任务/直接任务差及成本按已有规则评价，局部正向或attention改变本身不能替代优化候选条件。

审计时间：2026-09-09T12:09:46.851445+00:00。本次仅源码、小元数据及既有账本读取；没有新模型/runner/训练/统计执行或handoff。输入SHA256：

- `taobao-transfer/recurrence-coverage-retrieval/NEXT_STRUCTURE_AUDIT.md`: `279caee1214fe7651cc8dae86dc264048fb284b20b639d2dbcb8b2c5419ebd77`。
- `taobao-transfer/recurrence-coverage-retrieval/source-snapshot/QFormerCross8.py`: `4ef5c5c78f5ef658f24a4786c3927cf254279cf560b9b4746a0a7d9145b42a7f`。
- `taobao-transfer/recurrence-coverage-retrieval/source-snapshot/QFormerCross8ContextRead.py`: `50d8dff6c1c25aa2201f3c4bc8be43461ee77cb28aa5416af3f58af0282719f6`。
- `taobao-transfer/recurrence-coverage-retrieval/source-snapshot/QFormerCross8RecurrenceExplore.py`: `29b750990fd5475bffeceadad0f3fb95ece37a1aa67a331fe0ccee7054f2f17c`。
- `data/Taobao_Action/feature_map.json`: `60c168c763257dbc9d2c4041c1c6d67ae5bbe71357f9ba559f6aa7f119a774dd`。
- `data/Taobao_Action/meta_data.json`: `10ae1cc1a019007b44c8e76ec0ffa35535ed5e3e972b89e9821ebd008ba3f971`。
- `runtime/UniRank/unirank/features.py`: `8359fab5bef5e1a979cc1dfe8b4bdad448d900afec14424d6ff056f0ae8a5c53`。
- `runtime/UniRank/unirank/pytorch/models/rank_model.py`: `c729cd9d20b22a4a2f0120325ddff9a1528206519f514e5e02387e6b1b3dfb9f`。
- `runtime/UniRank/unirank/pytorch/dataloaders/unirank_dataloader.py`: `3e9a9c61f6e56fcab9806df6bd160929cc6e8bb50fb7d9ddbaae7af10d4c61a2`。
- `taobao-transfer/recurrence-memory-sources/QFormerCross8MemorySources.py`: `7da32d4b5fab79d0c3b79afec0cbaa8543eaff356b287982ca714637653a2ed7`。
- `model_zoo/QFormerCross11SourceUpdate.py`: `176194c1188b051e730f0eb06726a7b0985fec691849fa34f6b9f38928e8c8ae`。
- `taobao-transfer/memory/episodes.jsonl`: `977aea1f556512b5dfec4ae73fb8198d51a3e4e9b57313aef9880655b38aaf5f`。
- `memory/episodes.jsonl`: `363d2eddf98e3e47eb384a8a8283bfb138df2fac2ceb213342517ea8763ff116`。


## 补充：条件候选 B 的双字段分工

上文四组记作A，仅证明原始请求可行。根代理现条件倾向B：每个父模型训练 `K=item,V=full`、`K=full,V=action`、`K=item,V=action`、`K=action,V=item` 四组，仍共八组；这取代A的候选选择，不是在A之外追加八组。原 `K=full,V=full` 与B前三组形成地址去action/内容去item的2×2结构，第四组提供反向分工。最终选择仍待当前coverage 16配对及归档完成，尚未实现。

B无需改SDPA：第二次 `cross_attn` pre-hook 可以同时将 `(q2,H,H,m)` 改成 `(q2,H_I,H_A,m)` 或 `(q2,H_A,H_I,m)`；同一原attention调用仍各执行一次w_k和w_fi。查询、w_qi输入、外层residual、mask、首轮及no-feedback流程继续原样。两种备用H必须由同一次原item/action embedding、同一事件顺序/mask构造，不能一个来自不同batch/截断或从第一次工作输出生成。原full H仍保留供入口和第一轮读取。

两种单替换组仍只构造其需要的一个备用H。两种双替换组需要同时构造H_I和H_A：在已共享原参数、不拆改full H路径前提下，可以让两条备用分支共用一次额外position gather/scale结果及valid_mask，但分别执行原列块F.linear（各带一次原bias）、LN384及mask。不得将 `LN(U_I)+LN(U_A)` 用于任何一条支路，不能把bias只分给其中一组，不能用同一个H_alt同时冒充两种字段表示。

双替换组解析新增projection为 `B*L*(96+16)*384`，即约35.23G MAC/8192样本前向；有两次备用LN而非一次，仍没有额外SDPA调用。两个常驻备用 `[8192,100,384]` 表示合计BF16为1200MiB，FP32为2400MiB（2.34375GiB）；这仅是两个张量，不含原H、投影/position加法中间量、autograd保存及梯度，不能当作真实新增峰值上界。额外position gather宜在两备用分支间共享，以免再多持有一份FP32历史大小张量。F.linear/LN前向可以先后计算，但双替换的K/V仍需同时存在，其反向图也不能通过detach释放。

当前保留位置时，双替换两组也无必然全局退化。全padding保持原掩码语义；单有效事件使K变化局部不起作用，但V变化仍有效，因此双替换组在单事件样本上可能只表现出对应V分支的效果，不能据此判定实现失败。action相同且无位置差时出现常量K/V的条件性退化仍适用，不能通过删样本消除。

B能提供比A更直接的联合分工比较，但2×2结果仍是分别从同seed训练的完整结构，包含共享参数更新与梯度变化；不是固定checkpoint上两项局部收益相加的保证。`K=item,V=action` 优于两种单替换时，也仍需对原full/full及最佳单轮的质量/成本规则判断。新增双替换fixture仅需在上文合同上核两条H各自内容、两个替换参数槽、共享参数梯度累计、相同mask及finally清理；本补充没有运行任何fixture或训练。

补充时间：2026-09-09T12:10:42.080192+00:00。
