# 后续循环机制核对（非第十五轮冻结训练计划）

来源：[Geiping 等，Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach，v2 §3](https://arxiv.org/html/2502.05171v2)，本轮等待训练期间核对。论文采用输入与状态拼接投影、随机训练深度，以及仅对末尾若干循环反传；输入注入让前端继续收到梯度。这是语言建模证据，不是 KuaiRec 提升证明。

本地现状：QFormerCross11Loop.py 的 anchor_mix 是固定凸混合，代码明确每轮全反传。当前 r7 的额外 attention 仅在第一轮后执行；直接 detach 第二轮输入可能切断它的训练梯度。字段/历史 CrossValue 本身已反复读取源，因此“无输入条件”的理论情形不直接适用。

尚未实现的候选：先比较显式初始 Query 拼接投影与参数/初值兼容的残差注入；或比较训练深度1/3交替（总前向轮数匹配固定2）与固定2、验证固定深度。若再试截断梯度，必须验证首次专用模块和初始化器的梯度路径，不能把不再训练的模块误当作正常消融。深度/梯度变化会改变训练计算与优化轨迹，须明确预算口径。

仅记录未来假设，不改变正在运行的宽度实验、配置、模型或选型标准。优先等完整验证与成本结果再决定下一轮。

实现核对：UniRank 的 train_step 一次包含 forward/backward/optimizer 更新，当前 accumulation_steps=1；run_validation 包装该方法计数。若做可变深度，可在 train_step 内设置本次深度并在 finally 恢复，避免验证/诊断 raw forward 改变训练计数。必须记录各深度实际 batch 数与样本数；620 次深度1加620次深度3只匹配调用轮数，因 batch 尾部大小及历史长度不同，不自动等于精确 FLOPs 匹配。尚未实现或运行。
