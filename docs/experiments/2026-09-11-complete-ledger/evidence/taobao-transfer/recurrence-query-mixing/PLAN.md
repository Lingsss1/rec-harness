# 有界的第二轮工作输入混合

本轮检验：原循环第二轮使用的入口状态与首轮工作状态各占一半，是否应根据样本和查询内容调整这个比例。此前八条 correction route 的点估计均低于各自原循环，差值分支补跑也已完成；本轮从原 q4/q8 循环构建新模型，并单独检验动态混合相对于同参数静态混合的作用。

八组均使用 `QFormerCross8QueryMixing`。固定两轮、预测状态不反馈、原末尾 post attention；不增加上下文读取、内循环或训练 seed。所有模型从 seed 初始化训练，参考 checkpoint 仅用于比较。

## 结构与对照

令 `e` 为入口工作状态，`z1` 为第一轮工作状态，`m = (e + z1) / 2`。仅替换第二轮工作输入：

```text
u_joint = concat(LN(e), LN(z1), LN(z1 - e))
u_constant = [+1, -1, +1, -1, ...]  # 长度 3D，RMS=1
g = 0.25 * tanh(B GELU(A u))
q2 = m + g * (z1 - e)
```

三次 LN 均为沿 D 维的 functional LayerNorm，没有可训练 affine 参数。`A` 为 `16 × 3D`，`B` 零初始化；channel 的 `B` 为 `D × 16`，query 的 `B` 为 `1 × 16`，二者均无 bias。`B=0` 时严格恢复原来的均值入口，后续对 `z1` 的权重为 `0.5+g`、对 `e` 为 `0.5-g`，均限制在 `[0.25, 0.75]` 内。

channel 的 joint 分支为每样本、每查询、每通道产生混合权重；query 的 joint 分支为每样本、每查询产生一个标量并广播到 D 维。网络在查询槽之间共享权重。constant 输入固定且不读取样本：channel 学到在样本和查询之间共享的通道向量，query 学到一个共享标量。

**训练前设计修订**：此前 analysis-plan 中的 constant 输入草案为单位 L2 范数；本轮在训练前改为交替 `±1`、RMS 为 1，使输入尺度与三段 LN 输出相当。其 L2 范数是 `sqrt(3D)`，不能写成单位范数。该修订只定义新轮模型输入，不修改 PSGD、学习率或裁剪。constant 与 joint 保持完全相同的 A/B 形状和可训练参数量；constant 可以跳过 LN 预处理，因此不能声称两者 FLOPs 相同。

## 八卡实验矩阵

| GPU | 实验 | 查询槽 | 粒度 | 输入条件 | 预计总参数量 |
|---:|---|---:|---|---|---:|
| 0 | mix_q4_channel_joint | 4 | channel | joint | 51,748,804 |
| 1 | mix_q4_channel_constant | 4 | channel | constant | 51,748,804 |
| 2 | mix_q4_query_joint | 4 | query | joint | 51,742,676 |
| 3 | mix_q4_query_constant | 4 | query | constant | 51,742,676 |
| 4 | mix_q8_channel_joint | 8 | channel | joint | 81,640,900 |
| 5 | mix_q8_channel_constant | 8 | channel | constant | 81,640,900 |
| 6 | mix_q8_query_joint | 8 | query | joint | 81,634,772 |
| 7 | mix_q8_query_constant | 8 | query | constant | 81,634,772 |

参数量为训练前解析估计，最终必须由模型实例检查。相对原循环，channel 增加 `4Dr=24,576`，query 增加 `r(3D+1)=18,448`。channel 与 query 容量不同；不能把两者的差异只归因于权重粒度。

## 原基线、配置变换与预算

| 比较角色 | 原模型 | 验证 macroAUC | 原总参数量 |
|---|---|---:|---:|
| q4 原循环 | ctx_q4 | 0.8284725654262436 | 51,724,228 |
| q8 原循环 | explore_dual_no_feedback | 0.8285212133742255 | 81,616,324 |
| 最佳单轮 | psgd_precond_every_step | 0.8286235325196747 | 81,616,324 |

q4 从 `recurrence-context-read/configs.json` 的 `ctx_q4` 原配置构建，仅删除旧类专属 `context_reread=false`、`attention_position=end`；q8 从 `recurrence-explore/configs.json` 的 `explore_dual_no_feedback` 构建，仅删除旧类专属 `recurrence_mode=dual_no_feedback`。新模型固定实现这些等价语义，显式设置 `work_steps=2`、`entry_weight=0.5`，并添加四个 `mixing_*` 字段。除了模型名称及这些已列字段，保留全部基线字段；逐字段变换保存在 `references.json` 的 `configuration_transforms`。

固定 seed `20262027`、batch `8192`、1 epoch、每组 `2193` 次优化更新；不能用总行数简单除 batch 推算更新数，发布数据按 payload 分别结束批次。固定训练行数 `17,822,296`、验证行数 `2,871,767`，四任务为 `is_click/cart/fav/buy`，不读取 test。保持 BF16、clip1、稠密 `HeavyBall.PSGDKron` 每步更新预条件器、稠密 LR `1e-4`、稀疏 Adagrad LR `0.05`；不增加训练 seed。

三条参考记录均保存原 result、config、manifest、checkpoint、validation predictions 的 SHA256，并保存完全一致的 122 个训练/验证输入 SHA256。准备阶段重新验证参考产物的哈希及三个 manifest 的输入哈希一致性；未在此阶段再次读取全部大数据文件。最终结果审计须重新验证实际输入文件，不能把保存的预期哈希当作本轮已重算的证据。

## 训练前与完成后检查

训练前确认八组实际参数量；B=0 时对应原循环前向一致、基线部分初始化/RNG 对齐；固定两轮与 no-feedback 路径成立；真实两步 PSGD smoke 必须确认四个头和 B 更新、A 第一步因严格零梯度跳过、第二步正常更新。沿用已经检查过的狭义自动初始化兼容处理，只有第二步且精确匹配的有限非零 A 梯度被误报为零时，允许返回原函数已经算出的完全相同自动尺度；不改变 optimizer options、初始化公式、LR 或 clip。

必须完成以下 12 个配对比较，**无论 AUC 差值正负都报告**：

1. 同 q、同粒度的 joint − constant，共 4 项。
2. 每个新模型 − 同 q 原循环，共 8 项。

同时报告全部八组相对最佳单轮的点估计差值；正增量候选可以补充相对最佳单轮的配对区间，并标为按结果选择的探索性比较。完整 mandatory 清单在 `references.json`，不得以正向筛选替代。

使用验证集相同用户的 500 次配对重采样，预设分析 RNG `20260906`，预期 `268,201` 用户；四任务共享每次用户权重，报告四个 AUC 及 macroAUC 的 95% 区间。这是固定 checkpoint 的条件区间，不包含训练 seed 方差，也不校正此前或本轮的候选选择。区间跨零不能解释成等效；近似候选可以保留为后续优化方向。

同 q、同粒度的 joint/constant 对照检验输入条件是否改善该混合网络的训练结果；它不单独识别检索机制，也包含动态条件带来的梯度路径变化。新模型与原循环的比较同时包含新增参数和混合能力。q4 与 q8 分开解释。

另在独占的同一 GPU 上测量参考模型及八组 checkpoint，记录 BF16、固定真实验证批次、batch 和迭代数、前向时延及峰值显存；排除数据加载时间，不能用并行训练墙钟替代。尤其应实际报告 constant 省略 LN 的计算差异。训练、统计分析及计时均需分别记录产物和退出状态；失败候选保留失败原因，不填零 AUC。
