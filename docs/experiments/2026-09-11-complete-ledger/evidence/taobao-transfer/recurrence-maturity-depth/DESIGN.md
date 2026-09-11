# 成熟行为编码 × 共享循环深度：四格草案

状态：`draft_execution_not_authorized_by_this_file`。当前只准备源代码、测试定义和文档；N 四组按原冻结协议完成。此文件不授权任何测试、数组读取、模型前向、GPU、训练或全局 memory 写入，实际执行由根节点在审查和冻结后另行授权。[机器可读协议与小文件引用清单](protocol.json)

| 格子 | 历史 action 条件 | work / prediction 更新次数 |
|---|---|---:|
| released_r1 | 共同严格时间 mask 后保留原 action | 1 |
| released_r2 | 同上 | 2 |
| mature_r1 | 同一 mask，未成熟三行为改 unknown | 1 |
| mature_r2 | 同上 | 2 |

这四格回答一个窄问题：在严格历史支持集内，保守隐藏尚未成熟的 cart/fav/buy 窗口标签后，独立训练 R2 相对 R1 的任务收益是否仍可观察、是否改变。无需继续结构搜索，也不重复已完成的 [A 时间可知性审计](../history-action-availability-audit/RESULTS.md)。A 的窗口重叠不是实际泄漏率；这四格同样不能估算旧模型有多少 AUC 来自泄漏。

## 共同输入与模型

先取原 `[max(0,l−100):l]`，由匹配 user_info 的 `timestamps[l]` 推定目标时间 t0，仅保留 `item>0 且 th<t0` 的历史。四格同秒和更晚 token 都屏蔽，历史字段归零，不回填，不筛掉目标行，不改 seq_len/目标索引；位置仍走原有效 mask 的 cumsum 路径。目标 item 和四个标签保持原值，目标 action 恒0。目标时间缺少独立列交叉验证，应保留这项限制。

mature 仅在 `th+86400<t0` 保留原 cart/fav/buy 组合；等于边界仍未知。其余同时隐藏阳性和零值，保留 item 及**假设已经可知**的 click。共用19类词表：原0–16映射不动，17为 `delayed_unknown`、18为 `is_click|delayed_unknown`。unknown 不等于 exposure、成熟负例或 padding；实际 action embedding 接收17/18。辅助 task mask 的未知行为0仅表示未观察到阳性，不是已知负例。转换随目标时刻而变，不能永久重写用户完整 action 序列。原发布文件只读，新 metadata/feature map 独立归档。

四格共用 `QFormerCross8MaturityDepth` / 原 `NoFeedbackStage`，仅 `work_steps=1/2`、loader `maturity_mode=released/mature` 两个因素不同。R2 沿 `z2=L1((e+z1)/2,H)`、`y2=L2(y1,z2)`，prediction 不反馈 work；每格保留同一个原末尾 post 和四个完整任务 tower，不加 N 的任务 bias/分支。相同19类构造路径须核四格初始参数、buffer、顺序、构造 RNG 和静态优化器布局一致。预计只比旧17类增加2×16=32个稀疏参数，须以后实际核验；不能声称与旧模型初始化/布局相同，更不能声称四格训练梯度或优化器轨迹相同。

## 匹配训练与唯一统计问题

沿原 q4 真实 manifest：seed20262027、shuffle=True/drop_last=False、原 loader 内部顺序、batch8192、1 epoch/2193更新、BF16、clip1、HeavyBall.PSGDKron 每步预条件，dense LR1e−4/weight decay0、Adagrad LR.05；相同四BCE损失、等权 macro AUC 保存/验证规则，关闭 compile/checkpointing。四格核对原始训练行/批次顺序指纹；两个同 mode 深度的变换输入也应一致。共4×2193=8772次正式更新，另4×2=8次 smoke 更新；无新 seed、test、自动重跑或额外 native CUDA stage 矩阵。R1/R2匹配更新与数据预算，不匹配 FLOPs。

对每个指标定义 `D_released=R2_released−R1_released`、`D_mature=R2_mature−R1_mature`，主交互 `I=D_mature−D_released`。四个新 checkpoint 必须在相同 2,871,767 验证行、268,201 用户和目标标签上比较；沿用已接受的原 AUC/BCE 与用户配对算法，**只使用一次500 draw、seed20260906的共同抽样**。每个 draw 对四模型计算指标后直接生成 D_released/D_mature/I。BCE 分母是该 draw 实际重采样行数。

每个量保留 macro AUC、四任务 AUC/BCE 和全部六组直接任务 AUC/BCE 效果差，即21项、共63个95%区间。所有正负值完整展示；不追加无目标的两两比较、旧 best 配对或新抽样。I区间从同 draw 的四模型差中差得到，不能相减两套区间端点。区间仅为固定 checkpoint 的探索性条件区间，不含训练seed方差、多重或选择校正。

D_mature 有正向 AUC 证据才支持对应条件下仍存在可观察的预测收益；I为负可与正 D_mature 并存，只表示减弱。任务异质性看直接任务差，不能凭“一个显著、一个不显著”断言只有某任务受益。跨零不是等效或没有信息。保守隐藏也会删去实际已在目标前发生的行为，且改变表示和训练轨迹，所以I不能量化实际泄漏贡献。四格都需重新训练：共同19类表、初始化和严格同秒 mask 已改变 released 对照，旧分数只作背景。新输入协议也不直接晋升旧 incumbent 或套用旧近似候选阈值。

## 成本和根节点冻结前的界线

只给四个新 checkpoint 做延迟描述：同一首批1024条原始验证行生成 released/mature 两份输入，记录两个变换 input hash，同 mode 的 R1/R2使用逐字相同张量。跨 mode 的 action 不同，不能宣称四份张量全等。四模型在训练终态后用同一排他物理GPU、BF16、每个10 warmup＋40 timed，共200前向/160计时；不重跑旧7模型，不按延迟挑选新结构。

原始解码、timestamp读取、逐mode转换的 CPU/墙钟/缓存与读列计数，以及 CPU RSS、输入字节与每模型峰值 VRAM 分别报告。转换排除在模型CUDA事件计时外，该数字不是完整部署吞吐量；训练 fit、验证、初始化、smoke 与并行总历时也分别记账。参数只增32不能推断 timestamp 缓存或整批转换的内存代价。

当前额外原始数组审计预算为0，复用A已完成证据，不重复1024样本、不默认全量预扫描、不创建工具候选或handoff。新 loader 的逐行一致性、边界、unknown、空历史与读取成本检查进入以后另行授权的 smoke/正常数据遍历；若最终源审查发现未覆盖问题，根节点须先冻结精确行/列/rowgroup及成本上限，才能增加独立读取。CPU合成测试定义覆盖19编码、共同初始化、短/空历史及接口，具体唯一调用和 CPU/墙钟/RSS 限额仍待根节点冻结；暂不继承 N 的额外 GPU gate。

protocol 中尚为 null 的运行时间/内存/产物限额、新源与数据派生身份闭包必须在执行前补全冻结；它们不是无限预算。没有标签事件/到达时间，click可知性与窗口完整到齐都只能作为假设，类目/品牌代理标签及其他特征可知性也未解决。因此本轮称“保守成熟编码匹配深度对照”，不称完整线上去泄漏或线上因果收益。

## 小文件引用

protocol.references 绑定原 q4 R1/R2 配置选项、两份 manifest/result 身份、NoFeedbackStage/ContextReadStage/原 QFormerCross8 源、原 loader/metadata、A已完成审计及原用户配对实现。仅消费小文件并记录哈希，没有读取/重hash dataset、预测数组或 checkpoint；尚在编写的 B 源与测试身份留给根节点一次冻结。
