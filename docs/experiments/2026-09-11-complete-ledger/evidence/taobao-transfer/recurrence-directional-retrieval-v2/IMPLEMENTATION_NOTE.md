# 无参数第二次检索方向：实现说明

状态：合成实现阶段；未运行真实 checkpoint、数据、训练或 GPU。四源及 floor 公式由 root 于 2026-09-09 审定；不引入训练 seed、参数或优化器修改。

## 最小路径

公开类 `QFormerCross8DirectionalRetrieval(QFormerCross8)`，显式 `parent_family=ctx_q4|explore_dual_no_feedback`，严格对应 `num_queries=4|8`；`directional_mode=extrapolate|uniform_key|attended_key|centered_key`，`identity` 仅合成控制。构造使用原 ContextReadStage / ExploreStage 的 partial 传给原 QFormerCross8 构造；q4 forward 委托原 QFormerCross8ContextRead.forward，q8 委托原 QFormerCross8.forward。不嵌套注册完整父模型，不新增 Linear、Parameter 或 buffer，不复制 stage 或 attention 数学。

每次 forward 创建局部 context，唯一目标为 `unified_qformer.layers[1].cross_attn`。其 pre hook 保存第一次 mask；q_norm / k_norm hooks 捕获第一次可微 Q1、K1，并原对象返回。第一次 attention post hook 在原 V 读取、w_qi 及输出投影已执行后生成源；第二次 q_norm 返回方向变换结果。外部 queries 不替换，原 w_qi 与 residual 仍读取 `(entry+work1)/2`，预测状态没有进入工作路径。第二次 K/V 按原代码重算。

仅 q_norm hook 不足以得到归一化 K：原代码先 q_norm 后 k_norm。两个只读 capture 加 attention post hook 即可保留原完整 forward；额外 SDPA 调用函数接口，不触发目标 module hooks。

## 数学与精度

每个 `[batch, head, query]` 独立计算，`d=head_dim`。四源：

| mode | 第一轮源 s | 额外 SDPA 次数 |
|---|---|---:|
| extrapolate | `-Q1` | 0 |
| uniform_key | `SDPA(zeros_like(Q1), K1, K1)` | 1 |
| attended_key | `SDPA(Q1, K1, K1)` | 1 |
| centered_key | `SDPA(Q1, K1, K1-mean_valid(K1))` | 1 |

额外 SDPA 使用原 mask、scale、dropout=0、非 causal 与原 BF16/autocast dtype。uniform 实际调用 SDPA；centered 不改成另一有限精度计算“条件质心减均值”。valid mean 排除 padding，计数分母先 clamp；all-padding 源显式为零。均值的归约精度应明确记录到实现/协议，不增加第二次额外 attention。

`n2=norm(Q2)`，`u2=Q2/max(n2,1e-12)`，`v=s/max(norm(s),0.1*sqrt(d))`，`t=v-dot(v,u2)*u2`，`Q2_new=n2*normalize(u2+0.25*t)`。norm/floor/tangent/最终归一化用 FP32，最后 cast 回原 Q2 dtype。源或 Q2 norm≤1e-12 时逐向量回退原 Q2。所有除法先 clamp，不能先 0/0 再用 where 遮蔽；fallback 的 backward 必须有限。identity 直接返回原 Q2 对象。

非退化精确实数下范数保持且转角≤atan(0.25)；FP32 关系容差 2e-5，BF16 cast 后采用 dtype 对应容差，不能声称 bitwise 范数相等。Q1/K1/source 不 detach、不 no_grad，第二次变换可向第一 Q/K 反传。floor 抑制小源单位化噪声但不保证收益；key 源共同投影 bias 与 -Q1 的来源差异仍须保留在结果解释中。

## 生命周期及替代方案

handles 与 Q/K/mask/source 全为局部 context 数据，finally 无条件 remove/clear；stage 最多保留无 Tensor 的 mode、计数、cleanup 元信息。同一 stage 重入拒绝；正常 forward 必须恰好两次目标 attention/q_norm/k_norm。异常不要求完整计数，但必须清理。context 退出不会切断返回张量已有 autograd 图；连续 batch 不保留上一 batch Tensor。

拒绝 gradient checkpointing 与模型 torch.compile：backward 重算时 scoped hook 已退出，动态 hook 也不在当前模型编译契约内。保留 RankModel.compile 的原优化器构造，以及 HeavyBall 原内部编译；不能将禁止模型 compile 扩展为禁用优化器编译。

显式 attention 子类须重述 Q/K/V/SDPA/w_qi/output 公式或增加跨层接口，扩大漂移面；替换原层可能在 optimizer 构造后留下失效参数引用。当前原 forward 加局部 hooks 足够。两家原父模型构造硬编码 `_unified_qformer_cls`，不能再传同名 kwargs；公共类直接调用 QFormerCross8 构造避免冲突。

同一模型进入/退出 context 前后 Parameter 对象必须相同；独立原模型/候选比较 name、顺序、shape、dtype、值、RNG，不能要求跨实例 object id 相同。identity 输出及梯度 exact；非 identity 函数本来会改变，不另加零门来追求初始化输出一致。无新增参数排除新增参数布局扰动，不保证 gradients、global clip、预条件器值或 AUC 不变。

## 合成验收

只用 tiny 原 q4/q8 stages、确定性公式输入；构造 RNG fork/restore，不设置新 seed。检查 identity 原输出/参数及输入梯度 bitwise exact；四模式与独立 attention 和方向公式相符；Q1/K1 可微；每 head/slot 范数及角界；第一轮、rawQ2、K/V/mask、w_qi、residual 输入不变；读取两次、额外 SDPA 仅第一次一次、末尾 attention 保留。

覆盖 all-padding、单有效 key、零/近零 source 或 Q2、平行/反向源、混合 fallback、错误 shape/非 finite、连续 batch backward、异常清理/重入、CPU BF16 有限前后向。单 key 的原读取输出应不变，但 source/Q 几何不必相同。检查 state、Parameter 对象、hook 数恢复，无全 P 或调试 Tensor 缓存。真实 FeatureMap 初始化/布局审计及原生 GPU smoke 由 root 另行安排，本说明不执行。

## 源码依据

- 上轮冻结 `recurrence-readout-retrieval/source-snapshot/QFormerCross8.py:148` 至 215：Q/K norm 顺序、mask/scale、V 读取、w_qi 独立输入；218 至 251：residual。
- `recurrence-context-read/QFormerCross8ContextRead.py:12` 至 25：两轮 no-feedback；27 至 33：stage hardcode；35 起：原 q4 完整 forward。
- `recurrence-explore/QFormerCross8RecurrenceExplore.py:33` 至 40、62 至 64：原 q8 工作读取和末尾 attention；67 至 78：构造。
- `runtime/UniRank/unirank/pytorch/models/rank_model.py:96` 至 110、192 至 234：checkpoint/model compile；845 至 851：保存与加载 state_dict。
