# 旧版 RecHarness＋修正版 NextItNet 迭代记录

原版 RecHarness 固定于 `74a927fefaf7d120c52b6e3a0809958bd33984f0`。本次运行于 2026-10-31；模型起点是已修正左填充历史读出位置的 NextItNet。原版 Thompson 路由、jump、候选晋升和回滚保留，LLM 及训练/评估通过本地匹配适配器执行；这不是未经适配的上游复现。

共 **29 轮（含基线）、107 次训练尝试：104 次成功、3 次超时**。实际搜索累计 **43,165.380 GPU 秒**，预算 43,200 GPU 秒；LLM 与最终测试推理单独记录。每次试验使用单卡、搜索提供四卡槽位。LLM 为 `gpt-6.1-sol / medium`，关闭独立模型集成。

四个 Amazon 类别的预处理 seed=0，训练和评估 seed=42；验证每类别最多 200 用户，最终测试每类别最多 10,000 用户，1 正例与 99 个唯一负例。以四类别宏平均验证 HR@10 搜索和选模，测试只在选模冻结后执行。仅一次独立搜索，不能据此推断稳定性或单独归因各项改动。

## 关键路径与 jump

| 轮次（含基线） | 改动／事件 | 验证 HR@10 | 结果 |
| --- | --- | ---: | --- |
| 1 | 修正版 NextItNet 基线 | 0.5425 | 起点 |
| 3 | 0.1 倍可学习 recency embedding | 0.5587 | 晋升 |
| 6 | 零初始化 log-recency／内容交互 | 0.5625 | 晋升 |
| 10 | 损失 jump：BPR → 使用批内负样本的 sampled-softmax 交叉熵 | 0.5525 | 暂时下降，进入新分支调优 |
| 11 | batch size：128 → 112 | 0.5650 | 超过 jump 前的 0.5625 |
| 12 | 每 epoch 学习率乘 0.95 | 0.5737 | 最终选中 |
| 16 | 架构 jump：卷积残差更新加入通道 sigmoid 门控 | 0.5625 | 暂时进入架构分支 |
| 17–20 | 架构分支后续调优，最好 0.5650 | ≤0.5650 | 未超过保存的 0.5737，第 20 轮回滚 |
| 21–29 | 回到损失分支继续搜索 | ≤0.5737 | 保留第 12 轮模型 |

**确实执行两次 jump。最终保留损失 jump 分支，撤销架构 jump 分支。** 第 10 轮保留旧模型作为临时回退点，第 11 轮新分支超过旧模型，随后 pending 观察在第 14 轮清空；第 16 轮架构跳跃直到第 20 轮仍未超过保存模型，状态中的 basin 返回 `default|change_loss_function@10`。`promoted=true` 在 jump 当轮表示暂时进入新分支，不应单独解释为质量提升。

最终选中 `round-0011-slot-0-attempt-1`（内部编号从 0 起，对应日志第 12 轮）。最终验证 HR@10 **0.5737**，测试 HR@10 **0.4990**、NDCG@10 **0.3126**。模型保留 recency 特征、批内负例交叉熵、batch112 和指数学习率衰减；未保留第二次 jump 的卷积残差门控。损失跳跃同时改变负样本使用方式，不能将后续提升归因于交叉熵公式本身。

## 全部逐轮索引

轮次中的 winner 是本组候选最好结果，不一定晋升；同分不晋升。详细假设、每个候选结果和原始状态位于对应 JSON。

| 轮次 | 本组最好 arm | 验证 HR@10 | 晋升／暂时进入分支 | 逐轮原始记录 |
| --- | --- | ---: | --- | --- |
| 1 | baseline | 0.5425 | 是 | [JSON](records/iterations/iteration_0001.json) |
| 2 | tune_num_heads | 0.5425 | 否 | [JSON](records/iterations/iteration_0002.json) |
| 3 | add_features | 0.5587 | 是 | [JSON](records/iterations/iteration_0003.json) |
| 4 | tune_num_heads | 0.5587 | 否 | [JSON](records/iterations/iteration_0004.json) |
| 5 | tune_num_heads | 0.5587 | 否 | [JSON](records/iterations/iteration_0005.json) |
| 6 | add_features | 0.5625 | 是 | [JSON](records/iterations/iteration_0006.json) |
| 7 | tune_num_heads | 0.5625 | 否 | [JSON](records/iterations/iteration_0007.json) |
| 8 | tune_weight_decay | 0.5625 | 否 | [JSON](records/iterations/iteration_0008.json) |
| 9 | tune_num_heads | 0.5625 | 否 | [JSON](records/iterations/iteration_0009.json) |
| 10 | change_loss_function | 0.5525 | 是 | [JSON](records/iterations/iteration_0010.json) |
| 11 | tune_batch_size | 0.5650 | 是 | [JSON](records/iterations/iteration_0011.json) |
| 12 | add_lr_scheduler | 0.5737 | 是 | [JSON](records/iterations/iteration_0012.json) |
| 13 | tune_num_heads | 0.5737 | 否 | [JSON](records/iterations/iteration_0013.json) |
| 14 | tune_num_heads | 0.5737 | 否 | [JSON](records/iterations/iteration_0014.json) |
| 15 | tune_num_heads | 0.5737 | 否 | [JSON](records/iterations/iteration_0015.json) |
| 16 | change_architecture | 0.5625 | 是 | [JSON](records/iterations/iteration_0016.json) |
| 17 | tune_num_heads | 0.5625 | 否 | [JSON](records/iterations/iteration_0017.json) |
| 18 | add_lr_scheduler | 0.5637 | 否 | [JSON](records/iterations/iteration_0018.json) |
| 19 | tune_dropout_wd | 0.5625 | 否 | [JSON](records/iterations/iteration_0019.json) |
| 20 | tune_lr_batch_scheduler | 0.5650 | 否 | [JSON](records/iterations/iteration_0020.json) |
| 21 | tune_num_heads | 0.5737 | 否 | [JSON](records/iterations/iteration_0021.json) |
| 22 | tune_num_heads | 0.5737 | 否 | [JSON](records/iterations/iteration_0022.json) |
| 23 | tune_num_heads | 0.5737 | 否 | [JSON](records/iterations/iteration_0023.json) |
| 24 | tune_num_heads | 0.5737 | 否 | [JSON](records/iterations/iteration_0024.json) |
| 25 | tune_num_heads | 0.5737 | 否 | [JSON](records/iterations/iteration_0025.json) |
| 26 | tune_num_heads | 0.5737 | 否 | [JSON](records/iterations/iteration_0026.json) |
| 27 | tune_num_heads | 0.5737 | 否 | [JSON](records/iterations/iteration_0027.json) |
| 28 | tune_num_heads | 0.5737 | 否 | [JSON](records/iterations/iteration_0028.json) |
| 29 | tune_seq_len | 0.5500 | 否 | [JSON](records/iterations/iteration_0029.json) |

## 最终测试

| 数据集 | 测试用户数 | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Movies_and_TV | 7467 | 0.3964 | 0.5203 | 0.6471 | 0.2984 | 0.3383 | 0.3703 |
| Industrial_and_Scientific | 10000 | 0.3731 | 0.5021 | 0.6454 | 0.2642 | 0.3062 | 0.3424 |
| Electronics | 10000 | 0.3136 | 0.4237 | 0.5674 | 0.2237 | 0.2591 | 0.2953 |
| CDs_and_Vinyl | 10000 | 0.4186 | 0.5498 | 0.6933 | 0.3048 | 0.3470 | 0.3833 |
| macro |  | 0.3754 | 0.4990 | 0.6383 | 0.2728 | 0.3126 | 0.3478 |

## 审阅发现

`tune_num_heads` 被选择 21 次，而该 NextItNet 是无 attention heads 的卷积模型；相关提案主要记录不适用或只改注释，仍重新训练，累计消耗 **8,321.413 GPU 秒（19.28% 实际搜索消耗）**。这些同分结果不构成 attention-head 调参有效的证据。29 轮不能等同 29 次实质模型创新。

两次 jump 都暂时降低验证指标。损失分支在后续调优中超过保存模型，架构分支未恢复而回滚。这是本次单 seed 路径的事实，不是受控消融或 jump 普遍有效的证明。

## 文件与复核

- [逐轮 CSV](rounds.csv)、[逐轮 JSON](rounds.json)：29 轮完整索引。
- [候选 CSV](candidates.csv)、[候选 JSON](candidates.json)：107 次候选执行、状态、修改摘要和 GPU 消耗。
- [测试指标 CSV](test-metrics.csv)：四数据集完整六项指标。
- [最终代码](models/selected.py)、[修正版基线](models/baseline.py)、[完整差异](selected-vs-baseline.diff)。
- `records/iterations/` 保留原始 29 个迭代 JSON，`records/execution/` 保留 107 个候选执行 JSON。
- `records/trials/` 保留每次训练源码、请求、执行回执及已生成的单模型审计；`records/native/` 保存最终选择和完整 Thompson 状态。
- `source/` 保存本次匹配控制器源码。上游依赖和数据处理版本由固定 commit 与配置记录定位。
- [最终结果](records/summary.json)、[原始配置](records/config.json)、[预算账本](records/budget.json)、[最终测试回执](records/final-test/receipt.json)。
- [SHA256 文件清单](manifest.json)；执行 `python3 verify.py` 核对文件完整性、逐轮／候选计数、预算及测试结果。

原始 JSON 中的绝对路径是运行机器的来源定位，不保证在克隆后可直接执行。归档包含运行源码与必要记录；不包含数据、训练权重、逐用户预测、运行环境、完整 LLM 调用转录或 API 凭证。保留这些边界，不把文件一致性检查等同于方法学审计。
