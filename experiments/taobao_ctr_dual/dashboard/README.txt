CTR 研究 tmux 看板

当前多课题实验使用 topics_launch.py，入口和五个窗口的说明见本目录 TOPICS.txt。
当前运行目录：artifacts/taobao-ctr-topics-latest/。以下内容保留为原双版本看板的说明。

启动／复用：
benchmark_res/.venv/bin/python experiments/taobao_ctr_dual/dashboard/launch.py --root artifacts/taobao-ctr-dual-20261104 --session ctr-research

进入：
env -u LD_LIBRARY_PATH -u LD_PRELOAD /usr/bin/tmux -L ctr-research attach-session -t ctr-research

连接时使用 /usr/bin/tmux；本机 Conda 中另有 tmux 3.8，与看板的系统 tmux 3.2a 混用会出现 libtinfo 警告及误导性的 not a terminal 错误。
启动器已固定系统二进制和客户端库环境，无需停用 Conda 或更改默认 tmux 会话。

窗口：
总览：预算倒计时、两版最佳验证 AUC 阶梯曲线、当前实验进度、GPU／CPU／内存、模型调用及事件。
研究：左右并排查看 auto 与强化版的问题、机制假设、相近工作差异、否定条件、已完成对照和最近归因。
成本：验证 AUC 与真实作业耗时／稠密参数的散点，当前观察到的效果—成本前沿，缓存与非缓存 token、训练和推理成本。
切片：当前最佳模型对 DIN 的历史长度、行为组成、类目匹配和用户频次切片变化，附样本和正例数。

鼠标可选择面板和窗口。C-b n / p 切换窗口；C-b z 放大或还原当前面板；C-b d 脱离看板。
看板是独立只读观察器。关闭或脱离 tmux 看板不会停止训练、研究 Agent 或实验监督进程。
启动器只创建观察器，不会将已有训练进程移动进 tmux，也不更改其 stdin/stdout。

数据边界：
只对已完成、已提交、数据身份一致的全量验证结果画效果曲线。训练中的进度不进入最佳 AUC。
校准和测试分数不画入曲线；不读取 final.json 或 assessment 下的评分输出。
旧 auto 结果不混入本次重跑曲线。其失败和开销在预算与事件提示中保留，不恢复时间预算。
原测试协议受影响的提示常驻界面；auto 重跑仅验证集、个人知识禁用。
对照、候选、外部知识与场景解释分别显示，单种子切片和前沿不表示显著性或机制确认。
LLM 输入总 token 已包括缓存，输出 token 已包括推理 token；不重复求和，不编造费用。
吞吐和 p95 继承现有验证流程测量；p95 是整批耗时，不是单请求延迟。

实现：单采样进程默认每 2 秒读取公开验证收据，GPU 每约 4 秒采样；每个面板读取同一缓存。
缓存、采样错误与 tmux 布局位于 artifacts/taobao-ctr-dual-20261104/dashboard/。
telemetry.jsonl 每约 5 秒追加观察器指标，不参与 Agent 研究或实验筛选。
主机资源是共享机器观测值，不能直接视为某个模型独占的计算量。

回归检查：
benchmark_res/.venv/bin/python experiments/taobao_ctr_dual/dashboard/test_dashboard.py

单个视图的无交互预览：
benchmark_res/.venv/bin/python experiments/taobao_ctr_dual/dashboard/app.py view --root artifacts/taobao-ctr-dual-20261104 --view overview --once --width 100 --height 30

依赖：tmux；现有 Python 环境额外安装 rich，未改变实验模型、训练器或评分器。
