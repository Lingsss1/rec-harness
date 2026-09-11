# 第二轮如何形成更好的检索条件

现状：q2=(entry+first_work)/2；前轮预测状态不得反馈。已完成的 R1/R2 对照支持循环，但不证明查询构造最优。正在运行的 memory-source 八组继续完成，不重复启动。

先用 rec-harness 的 geometry 工具检查 q1/q2，保存原始槽表示、固定样本ID、checkpoint/code哈希及任务阳性数。初步4096行仅375用户、buy4例，不用于稀疏任务重要性排序。

下一步构建更充分的固定分层验证子集，比较第二次 L2 输入中 entry 与 first_work 的跨样本 shuffle / zero，保持另一项和全部权重不变，加入 identity 控制；输出逐批逐任务 paired loss 表交给 analyze_perturbations.py。不能把扰动损失作为训练目标。

只有当结果支持样本相关信息有用而当前融合限制该信息时，才排队一个有对应消融的 query adapter（如 entry锚定的first_work残差），并按固定训练协议验证。注意 attention权重或几何变化本身不证明检索质量改善。没有支持证据则保留原查询公式。需要新度量时按 evolver propose→六门验证→handoff→check→实验→close 流程，不直接修改稳定工具。
