# 第二次历史检索前交换工作槽内容

本轮检验一个结构问题：首轮已从历史读出的工作槽内容，是否需要先在槽间交换，再形成第二轮检索查询。它不预设 AUC 收益，也不把查询槽解释成四个任务。

## 来自实际源码和 memory 的动机

原 q4/q8 的 `e=L0(q,H)`、`z1=L1(e,H)`、`x=(e+z1)/2` 保留原浮点运算。给定入口 e 和历史 H，首轮工作更新及 x 的形成没有跨工作槽的信息传递。原入口 ns 模块及预测层 `L2(e,z1)` 已有跨槽处理，因此不能说全模型各槽完全独立。无 prediction→work 反馈意味着预测层汇集的内容不会用于下一次历史寻址。

已查 [ContextRead](source-snapshot/QFormerCross8ContextRead.py)、[Explore](source-snapshot/QFormerCross8RecurrenceExplore.py)、此前 Middle、Adapter、Core、StateRoutes 和 MemorySources 源码：中间 post attention 改的是预测 y；末尾 post 位于第二 work 之后；work_entry 向历史拼入口 e。这些均不是本轮的 `z1` 内容交换→第二 `w_q` 路径。此前 coverage 是历史 attention 分数偏置，字段分工改变原始 K/V 供给，方向反馈来自历史 key；它们的负结果保留，不被当作本轮收益证据。

任务独立第二轮检索也是一个未覆盖的方案，但为保持每任务原有查询槽，它需四份第二轮下游状态，且任务自由度和计算都改变。当前优先选择共享工作槽内容交换；本轮不同时展开任务分支模型。任务异质性仍仅作为预定四任务及直接效果差分析，不改变 macro AUC 主目标。

## 固定结构及四项对照

只给原第二次 `w_q` 一个新输入：

```text
r = post_self_norm(x + relay_attention(x, z1, z1))
second_work.w_q(r)
second_work.w_qi(x), outer_residual=x
```

复用既有 post attention 的全部投影、Q/K 归一化以及 post_self_norm。第一轮、H/K/V/mask、预测状态及末尾原 post attention/FFN 均沿用原结构。无新增参数、buffer 或构造随机抽样；原 post 权重现在多参与一次计算，梯度和优化轨迹会改变。

| 模式 | 内容交换 | 能回答的问题 |
|---|---|---|
| all | 学得的 attention 读取所有 z1 槽 | 完整候选是否改善原模型？ |
| own | 对角 mask，只读对应的一个槽 | all 相对 own 的改善是否需要访问其他槽？ |
| uniform | normalized Q 乘零后，用同形状 SDPA 均匀读取全部槽 | all 的适应性选择是否优于均匀汇集？ |
| other | 屏蔽对角，学得的 attention 读取其他槽 | 在此结构中保留本槽内容是否有利？ |

四项在 q4/q8 各训练一次，共八组。own/other 用 `[B,H,S,S]` 可广播的布尔 mask；不把 batch 改成 B×S。all/uniform 可使用原无 mask SDPA。统一投影维度不保证 CUDA kernel 或耗时相同；own/uniform 的额外 Q/K 梯度可能为零。末尾 post 仍使这些共享参数正常参与训练，不能将此称为优化器动态匹配对照。

只有候选优于自身原循环或通过原定近候选质量/成本门槛才保留。all 优于 own 但仍低于原模型，只能作为失败结构内的相对结果。14 项必选配对包括八项对原模型、每父模型三项结构比较；所有四任务 AUC/BCE、六项任务直接效果差和全部延迟都报告。不增加组合量 I 或新的统计抽样规则。

## 文献背景与边界

[Set Transformer，§3.2、式11–12](https://proceedings.mlr.press/v97/lee19d/lee19d.pdf)先用多个 seed 聚合输入，再用 SAB 让多个读出交互。其聚类解码器对照同时改变了 pooling 和后续 SAB，不能独立证明槽通信的收益，也没有检验下一轮历史读取。这提供了多个读出先通信的动机。

[Slot Attention 原文](https://proceedings.neurips.cc/paper/2020/file/8511df98c02ab60aea1b2356c013bc0f-Paper.pdf)通过沿槽轴归一化的 attention 竞争耦合各槽，再逐槽更新。[补充材料](https://papers.nips.cc/paper_files/paper/2020/file/8511df98c02ab60aea1b2356c013bc0f-Supplemental.pdf)比较归一化轴，也将直接槽消息传递列为未来扩展；它没有本轮的直接内容 relay。两篇均未验证淘宝本轮共享 post 权重、w_q-only 路径或四种控制的优劣。

## 执行与诊断边界

固定 seed20262027、两轮/no-feedback、PSGDKron 每步预条件、dense LR1e-4、sparse Adagrad .05、clip1、BF16、batch8192、2193更新。协议、八组配置和 native 源码快照已写入。实现先通过 CPU 公式/路由/清理检查，再通过一次有明确预算的 native BF16 小 stage 检查和八个真实两步 smoke，最后执行正式训练。保留故障证据，不自动重试或修复优化器，不读取 test。

当前没有 ready diagnostic handoff，`coevolve check` 所需 task id 不存在；[路由检查](reports/routing-check.json)记录实际目录检查、原模型身份和协议。已有 donor 诊断因不能指导具体结构而继续搁置。本轮是依据源码和训练证据提出的可证伪结构实验，不创建一个虚假的诊断结论来触发它。
