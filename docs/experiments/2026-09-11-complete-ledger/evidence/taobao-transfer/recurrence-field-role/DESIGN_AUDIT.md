# 第二轮字段角色：独立设计与公式审查

当前选择为方案 B：每个 q4/q8 原父模型各 `item/full`、`full/action`、`item/action`、`action/item` 四组，斜杠前后依次表示第二次 work attention 的 K/V 输入。原 `full/full` 复用既有训练。已完成 coverage 轮全部不保留、已归档 133 次训练和 203 条 episode；复制到本目录的早期可行性文档中“coverage 尚在收尾”等文字是历史时点，不是当前运行状态。本文件不据这些负结果预测新轮收益。

结论：四种分配有可证伪的不同作用，可以按已选八组实施有限验证；目前未发现要求撤换某组的全局数学退化。此结论是设计可行性，不是已通过原生训练或已取得增益。

## 字段公式与必须保留的位置

按原 `sequence_features` 次序拼接六个 item 字段和历史 action；当前布局是六段 16 维 item 共 96 列，最后 16 列 action。实现应由字段 source、实际 embedding 宽度推导列，并核验原总宽度与预期七字段，不能依赖运行时字典顺序。所有字段先取 `[:, :-1, :]`，目标槽不进入任何历史表示。

令 `m` 为原有效历史 mask，`p = position_embedding_scale * position_embedding[(cumsum(m.bool())-1).clamp_min(0)]`，共享原投影的列块为 `W_I,W_A`，共享 bias 为 `b`，共享原 `sequence_input_norm` 包括其 affine 参数与 eps。定义为：

```
H_full   = m * LN(W [x_item; x_action] + b + p)
H_item   = m * LN(W_I x_item + b + p)
H_action = m * LN(W_A x_action + b + p)
```

`H_full` 必须直接调用原父 `_build_sequence_tokens`，保持完整投影的一次 GEMM 和原操作顺序。上述分块写法不授权把原 full 路径改成两次 GEMM 相加，也不要求两者在 BF16 下逐位等同。不能从 LN 后的 full 减去字段贡献；`H_full != H_item + H_action` 一般成立。两个备用分支各自加一次原 bias/位置，复用同一个 LN；组合组包含两次共享 bias/LN 的独立使用及相应梯度累积，不是新增两个参数副本。

仅 `unified_qformer.layers[1].cross_attn` 的第二次调用替换选定 K/V 输入。原 attention 内 `w_k` 后再做逐 head `k_norm`；V 是 `w_fi`，不是交互后的 `w_v_pair`。原 SDPA、布尔 mask、scale、`w_qi(q2)`、交互乘积和层 residual 公式均复用。入口、第一 work、第一 prediction 和 `q2=(entry+first_work)/2` 的计算沿原父路径；prediction 不反馈到 work。V 被改变后交互输出自然改变，“保留交互”只指直接 query 输入、权重和公式没有替换。

历史 action 合法使用已有历史动作编码，不能读取四个当前标签，不能因目标 action 为零占位而把目标 item 槽放回历史。第一轮 full、第二轮 query、未替换的 K/V 仍可能含被排除字段；本轮不是全模型字段删除实验。七字段来源和 `[:-1]` 是源码/已有数据合同的检查，不能替代重新证明数据生产阶段每行时间因果性；本审查不读取真实行。

## 各对照能说明什么

记父模型为 `00=full/full`，`10=item/full`，`01=full/action`，`11=item/action`。每个父模型的八项预定配对包括四候选对父、两个单改彼此、组合分别对两个单改，以及反向 `action/item` 对组合；两父合计 16 项，全部保留正负结果。

- `10-00`：第二轮 K 的字段与归一化路径改变是否改善整个训练结果。
- `01-00`：第二轮 V 的字段与归一化路径改变是否改善整个训练结果。
- `11-10`、`11-01`：在另一项分配已改变时，加入剩下一项是否有帮助。组合仅胜单改不足以成为优化候选，仍须满足对原父和计算开销的规则。
- `10-01`：两种单改的整体结果比较，不能单独证明地址比内容更重要。
- `action/item - item/action`：角色方向是否有区别，同时改变两条路径，不能独立归因某一个字段/投影。反向若胜出也应按同一候选政策处理。

这些是分别训练的完整模型比较。共用 seed 和初始参数控制了部分差异，但训练后原第一轮、共享权重、梯度、clip 和 PSGDKron 状态都可能改变。零新增参数只保证可保持原静态参数集合/布局，不保证函数、梯度、优化器轨迹、激活峰值或延迟相同。两个父结构分别判断，不能合并成 q4/q8 尺寸的因果效应；query slots 不是任务标签。

root 已在新 `protocol.json` 预声明组合效应 `I=AUC11-AUC10-AUC01+AUC00 = D11-D10-D01`，其中 D 均相对同一父模型。应使用共同用户抽样中对应的每个保存 draw 形成 I，再求区间，不能相减三个区间端点。I 正只表示指定训练结果/指标下的非加性，不保证组合优于原父、不能证明同权重机制或“互补信息”。每个父模型预定四任务 AUC/BCE、macro AUC 和六项任务 AUC/BCE 效果差，共 21 个区间、两父合计 42 个；只消费保存的 500 draws，预算 60 秒 wall、120 秒 CPU、256 MiB RSS、零新增抽样/forward，适合独立纯 NumPy 后处理，不应为此导入 Torch。定义和实现仍须在运行前冻结；本文件及公式 fixture 不执行真实组合统计。原 16 个比较的四任务与六项直接效果差仍完整报告，不因某个任务正向改变原 macro 目标。

## 退化情形不是要排除的样本

- 全 padding：三类历史 token 在 LN 后乘 mask 为零，但 `w_k/w_fi` 有 bias，后续 `w_v_pair/w_o` 也有 bias。必须保留原 key mask 和原 SDPA/affine/residual 行为；不能强制完整输出为零。正常空历史为固定 L 的全 padding 行，不等于物理 L=0。
- 单有效事件：概率只能为 1，固定权重下 K-only 的检索输出理论不变，第二 K 路径的局部梯度为零；共享 K 参数仍可从第一轮获得梯度。V 改动一般仍有效，组合的差异可完全来自 V。不要把此 null 当作实现错误。
- 若某行所有有效 K 完全相同，则任意 Q 得到均匀权重；如果所有 V 相同，K 改动也不能改变加权值。相同 action 不自动满足这些条件，因为 `H_action` 保留有效事件位置与共享 affine；不得为避免局部 null 删除这些行。
- 字段投影恰为零、落入 LN 可消去的统一平移方向或被后续 K/V 投影消去时，可出现更局部的不敏感。LN 还有 eps、非均匀 affine 与浮点舍入，不能泛称任意正比例变化都严格不变。
- 孔洞、左右 padding 均使用有效事件累计序号；未知字段类别零不能用于替代原历史 mask。只看 action 是否非零来重建 mask 会改变研究对象。

## 数值、梯度和恢复合同

备用 `F.linear` 在原 autocast 中执行，使用原 weight 的对应列（或有正确反向的选列操作）、原 bias 和原 LN。不得 detach、转成新 Parameter，或改优化器成员顺序。CUDA 下线性输出可能为 BF16，FP32 位置相加/归一化后的历史 H 可能为 FP32；逐 head Q/K norm 可为 FP32，V 可为 BF16。备用 token 必须匹配原历史 H 的实际 dtype/device，而不是要求 attention 所有输入同 dtype。保持原全局精度开关，不因 CPU fixture 成功就宣称真实 CUDA 路径通过。

建议上下文仅存本次 forward 的 full/必要备用表示，在原父构造历史之后绑定；原 full 表示不替换。首 work 调用原样返回，第二调用只改选择的 keys/values，保留同一 queries、mask 和未改的 full 对象。完整返回要求恰好一轮历史构造、两次 work、两次 prediction、一次 post。异常退出亦须移除所有局部 hooks、还原临时 builder 方法、清空 Tensor 引用且不吞掉原异常；拒绝嵌套/未绑定/多余调用。跨 batch 不留计算图，gradient checkpointing 与模型 compile 保持关闭。

完整初始化 smoke 必须独立重建原父并核 state、persistent/nonpersistent buffers、参数名/次序/shape/dtype/值、CPU/CUDA 构造 RNG 和 dense/sparse 静态布局；reference 构建后恢复其前 Python/NumPy/Torch RNG。`identity` 控制直接绕过备用构造和替换，用于原函数 exact 检查。active 四组本来就改变初始函数，不能要求其输出与父模型恒等，也不能把“初始化参数相同”报告成“初始化函数相同”。

独立 CPU 公式测试只用解析合成输入和共享参数小 fixture；其通过范围限于公式、局部梯度/退化与原源码操作，不等价于全模型/原生 BF16/真实 PSGDKron smoke。模型 owner 和 runner owner 分别负责生产上下文恢复及完整原生合同。实际执行次数、失败和资源成本随测试报告保存，未执行项目明确 pending。

## 开销与结果边界

单改各需一条备用历史，组合与反向各需两条，不能把后一类按单个小张量预算。B8192/L100/D384 的一个 FP32 历史张量约 1200 MiB，BF16 约 600 MiB；两条并存及反向保存值会进一步增加峰值，这不是精确峰值估计。额外 item/action 线性投影约 30.20G/5.03G MAC 每批前向；双替换约两者相加并多做两次 LN/位置处理。SDPA 次数和历史长度不增加，但不同 GEMM 列宽与共享梯度累计仍影响性能。

本轮应由真实 native fixture、八组原批量两步 smoke、固定 2193 更新和最终 11 模型同批同设备延迟逐级证实可运行性及成本。训练/检索路径改变本身不构成质量收益；全部候选按原父 macro 点差与近似候选/延迟政策判断，区间跨零不表示等效。过去循环必要性和任务异质性只约束解释，不证明字段角色分工会提升。

## 本次独立 CPU 证据

`test_formula_reference.py` 的 7 项测试已实际通过一次，原始输出和警告保存在 [formula-offline.json](reports/formula-offline.json)，完整子进程命令、退出码与成本保存在 [formula-execution.json](reports/formula-execution.json)。没有构造完整模型，只抽取生产 `field_column_layout` / `build_role_history` 与原 token builder 的 AST；生产 helper AST 前后同为 `4563714b9f039e56704c0c7809194b55a0b8eaf57f80b4e72bf1aa7e0d19da08`。完整模型文件当时仍由 owner 收尾，其报告时 SHA 仅是来源快照，不充当最终冻结身份。

测试覆盖生产 112→384 宽度和不同字段宽度；NumPy 逐标量列累加的 FP64 projection/LN oracle；目标槽与被排除字段不影响备用表示；孔洞/左 padding 的有效位置一致；原 mask 下零字段与空历史分离；共享参数六坐标的独立有限差分和被排除列/目标槽局部零梯度；LN 非加性、相同 action 保留位置差，以及单 key/恒定 V/全 padding 后 affine 不为零的数学边界。helper 不修改参数或 CPU RNG。两种角色 helper 的数值比较容差为 FP64 `atol=rtol=2e-12`；梯度 finite difference 为 `atol=2e-7, rtol=2e-6`，不用于宣称 CUDA 逐位等价。

suite wall 为 0.07653 秒、CPU 0.20514 秒；包含导入/进程退出的整次 wall 为 1.70789 秒、CPU 2.21721 秒、peak RSS 715415552 bytes。最初 `/usr/bin/time` 不存在导致命令退出 127，Python 测试尚未启动；该工具错误记录在 execution 收据，之后使用标准库计时启动上述唯一一次测试。PyTorch 在 CPU 反向时产生了内部 CUDA availability 的 error304 警告，已原样保留；`cuda_initialized=false`，没有显式 CUDA 张量/内核操作。没有真实数据、checkpoint、optimizer 更新或 bootstrap draw。

当前生产上下文源码的静态检查未发现新的阻碍：只在第二调用替换 K/V，保留原 attention/SDPA；with 上下文在异常时也移除 hooks、清空 histories，模型 finally 清除上下文引用；构造与 forward 均拒绝模型编译/activation checkpointing。此处不是完整生产上下文执行证明，owner 的 stage/full-model tests 和 root 的原生 smoke 仍独立负责这部分准入。
