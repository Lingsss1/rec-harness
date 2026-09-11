# 百度 ITT 原文复核

用户确认的论文为 ACL 2025 **Inner Thinking Transformer: Leveraging Dynamic Depth Scaling to Foster Adaptive Internal Thinking**。此复核不改变正在执行的模型或协议。

- [§3.2，式 (4)，PDF 第 3 页](https://aclanthology.org/2025.acl-long.1369.pdf#page=3)：`x[t] = sum_{i=1..t} f(x[i-1]) * phi[i]`。共享变换读取上一步累计状态，thinking step encoding 按通道乘权；不能直接等同普通加法位置编码。本地原层已经包含残差，若移植须明确完整输出与增量的区别。
- [§3.3，PDF 第 4 页](https://aclanthology.org/2025.acl-long.1369.pdf#page=4)：router 根据当前表示选择并加权继续计算的 token。连续权重参与梯度，不表示硬 Top-K 索引可微。
- [Algorithm 1，PDF 第 14 页](https://aclanthology.org/2025.acl-long.1369.pdf#page=14)：后续选择性前向传入并返回 `KV_new`，说明循环接口携带并更新 cache；没有充分规定覆写、追加和未选位置语义，不能断言每轮全量重写历史 K/V。
- [Table 3，PDF 第 7 页](https://aclanthology.org/2025.acl-long.1369.pdf#page=7)：完整／去 RTC／去 router／去 step encoding 的 PPL 为 10.25／11.02／10.44／10.56。最后一项实际差值为 +0.31，表中 +0.22 为不一致处。这些是语言模型消融，不能直接预报淘宝 AUC。

本地第二轮已有 `z1` 参与查询，输入历史 `H` 固定；其 K/V 投影仍逐次执行。当前八组实验检验历史标签成熟度与训练轮数，以及预计算 `H1=G(H0), H2=G(H1)` 的读取顺序。后者不是依赖 `z1` 的历史写回，也不构成 ITT 精确复现。

对理论的使用应更谨慎：不能据附录声称循环必然改善优化或增加信息。独立数学复核指出，对于共享参数的增量递推 `y[k+1]=y[k]+delta_k(y[k],theta)`，参数雅可比须包含每步直接依赖：`J[k+1]=(I+d(delta_k)/dy) J[k]+d(delta_k)/dtheta`；增量小本身也不推出误差收缩（零增量即为反例）。这是我们的数学审查意见，不是论文的实验结果。

当前有效问题是：在相同输入证据下，额外计算能否构造更有用的表示和检索条件。需要由匹配实验检验，不能从循环结构或信息流图直接推出收益。
