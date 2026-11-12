淘宝广告 CTR 双版本对比

运行目录：artifacts/taobao-ctr-dual-20261104/
持久化入口：controller.py suite；每个版本由独立 worker 持有排他锁。
默认实际运行解释器：benchmark_res/.venv/bin/python，不要解析其软链接后绕过虚拟环境。

auto：单次提案／编码循环，读取已有训练记录，可主动搜索相同的初始论文和五张行业卡。
enhanced：相同模型和来源，增加显式问题检索、条件化资料供给、实验后多轨迹反思与场景知识检索。
这检验整个工作流组合的效果与成本，不能直接回答其中某个组件的单独贡献。
个人知识禁用；两个版本不共享新实验结果、模型或生成的 Agent 知识。

两版各分配四张 A800，模型调用均指定 gpt-6.1-sol / high。
24 小时从第一次原始数据获取启动计时；同一截止时间覆盖公共准备、校准、失败、研究与最终评估。
GPU 0-3 给 auto，4-7 给 enhanced；每个 GPU 通道同时只执行一个模型变体。
无固定研究轮数。下一轮必须留出归因、重复与测试预算，失败不重置时钟。

数据：天池 dataset/56。使用已有三张原始表，补齐 Kaggle zerotrustai/taobao-ad-display-click v1 的行为日志。
该公开镜像中三张对应表的 SHA256 与现有 Hugging Face 镜像逐项匹配。
完整归档与行为 CSV 的 SHA256 在 assets/acquisition.json；没有声称核对到官方行为日志校验和。
下载失败与范围续传保留记录；CSV 不进行粗粒度事件去重，缺失画像不删除样本。

每个原始曝光为一条样本，clk 是标签，clk/nonclk 不作为特征。
按北京时间八个自然日分为六天训练／一天验证／一天测试。
历史事件严格早于曝光，排除同秒事件；最长回看十四天，当前提案最多供应最近 128 个事件。
共有类目、品牌、pv/cart/fav/buy 和时间，不使用虚构的历史商品 ID。
采用预先声明的哈希字段表示；哈希碰撞和静态快照 as-of 未知均属于解释边界。
起点模型：字段 MLP、掩码均值历史、DIN 思路的类目／品牌目标注意力。
这些是任务字段适配基线，不冒充作者实现的逐项精确复现。

正式搜索之前先运行因果可见性测试和完整实验单位成本校准，再冻结 protocol.json。
每个候选／对照从头训练，固定样本、评分、历史供给和配对种子。
可修改候选表示、交互和网络；优化器／训练损失变更须写入前瞻计划并安排相应对照。
所有变体最多四个，按一个机制实验登记；训练、推理、失败和模型调用均有收据。
验证指标驱动选择。候选、关键对照和种子先冻结，在余量允许时完成三个模型种子。
随后冻结检查点列表，执行一次封闭测试阶段；测试失败没有自动重试或重选。
三次模型种子不等于三条独立 Agent 研究轨迹，本次结论保持这个统计边界。

Agent 固定指令保持通用；初始消息仍然只有 task_ref/runtime_ref/workspace_ref。
科学目标、数据字段、模型接口、阶段动作和知识均来自外部可解析契约。
编码调用运行于独立 chroot 和非特权 UID，只导出当前版本的公共材料。
宿主 /proc、另一版本和原始私有标签不进入编码工作区；认证材料保留在私有运行目录。
宿主固定训练、构造特征和评分。候选静态 I/O 规则降低意外访问风险，不是任意 Python 的安全证明。

查看状态：
benchmark_res/.venv/bin/python experiments/taobao_ctr_dual/controller.py status --root artifacts/taobao-ctr-dual-20261104

因果和模型验收：
benchmark_res/.venv/bin/python experiments/taobao_ctr_dual/test_runtime.py

关键记录：
status.json / budget.json / protocol.json：实际阶段、截止时间和冻结协议。
prepared/progress.json：全量数据处理阶段。
auto/status.json、enhanced/status.json：两个版本的实际阶段。
comparison.json、REPORT.txt：持续汇总及最终报告。
trials/*/process.json：模型的精确训练复现命令。
rounds/：前瞻假设、相近工作、候选代码、检索、反思和所有模型调用日志。
assessment/：关闭后的独立测试输出，不提供给搜索 Agent。

控制：在运行目录创建 STOP，要求两个版本在当前实验结束后关闭研究并进入预留评估。
不支持通过再次调用 suite/worker 覆盖已有研究状态；异常恢复必须先核对 pending 进程与已完成收据。
