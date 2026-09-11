# 循环核心更新实验

主对照固定两轮 middle_r2，默认PSGDKron、width384、LR1e-4、Adagrad0.05、clip1、seed20262027、2193更新、batch8192不变。两组回到末轮四任务BCE，仅改变一个核心结构因素。

- middle_repeat_half：第二次中间层更新 q <- q + 0.5*(F(q,H)-q)，首轮正常。此前tail-half改的是末层，这里只作用于中间循环。
- middle_bounded_interaction：仅中间共享层 w_qi(q) 改为 2*sigmoid(w_qi(q))，与注意力结果相乘；attention QK、其余层、读出不变。参数量保持一致，但将有符号线性交互变成正值门控，这是表达能力与稳定性的联合改变，不能把任何收益直接归因为避免发散。

依据：混合深度/中间监督减轻深度退化但降低峰值AUC；当前QK归一化未约束乘法交互分支。后一结构是代码启发的待测假设，尚无中间表征发散证据。

先CPU/CUDA契约测试，再真实8192 smoke与完整训练。不增加seed，不评估test。
