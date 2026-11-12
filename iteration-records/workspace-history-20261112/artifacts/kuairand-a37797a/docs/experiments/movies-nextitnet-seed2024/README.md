# Movies_and_TV NextItNet 迭代记录

本次对照在 Movies_and_TV 上固定训练种子 2024，保存 MLE-STAR、Karpathy autoresearch、staged-autoresearch A/B/C/D 和 MLEvolve 的逐轮候选、方案、复盘、训练曲线及评测结果。7 个主方法各分配 44 次候选尝试，共 308 次；D 的目标向量归一化另有 4 次补充训练。

从 [统一迭代索引](iterations.csv) 查看全部 312 次尝试；从 [最终结果表](comparison.csv) 查看评测器 v2 下的结果。原始搜索文件保持原样，v2 的检查点重评单独保存。当前验证集选中的单模型最高验证 HR@10 为 B 的 0.715；MLE-STAR 固定集成的测试 HR@10 为 0.567247。单种子和已在先前研究使用的测试集不支持统计显著性或独立泛化结论。

## 训练和评测协议

Amazon Reviews 2023，5core last_out，按上游预处理逻辑以种子 2024 采样 5%；12,290 用户、17,566 物品，训练交互 131,548，验证交互 8,238，测试交互 7,755。输入来源和哈希见 [数据来源](protocol/data-provenance.json) 和 [数据统计](protocol/trainval-dataset-stats.json)。

所有方法使用相同的宿主训练器：BPR、Adam、梯度裁剪 5，最多 200 个 epoch，每个 epoch 200 个 batch，每 5 个 epoch 验证，patience 20。实际训练长度受提前停止影响，不能把 44 次尝试理解为相同 GPU 时间或相同成功训练数。验证取 200 个用户；最终测试 7,755 个用户；评测种子 42，每个正例配 99 个负例。协议配置、模型和检查点哈希保存在各方法的 `config.json`、`frozen.json`、工作进程记录以及 staged 注册文件中。

原始基线对右对齐历史使用 `count_nonzero-1` 读取位置，验证 HR@10 为 0.395；改正位置后可达到 0.590。迭代收益包含这项正确性修复，不能全部归因于架构改进。修改读取位置还会改变验证时的检查点选择和提前停止，不是仅对固定权重做推理替换。

MLE-STAR 和 autoresearch 使用 GPU 0–3；staged A/B/C/D 在 GPU 4–7 并行；MLEvolve 使用 GPU 0–3。staged 和 MLEvolve 显式使用 `gpt-6.1-sol`、`high`；较早的 MLE-STAR 和 autoresearch 未显式固定推理强度，保留其历史配置。尝试预算对齐不代表推理成本完全对齐；未把未经核验的会话累计 token 用量换算为费用。

## 各方法记录入口

| 方法 | 次数与结构 | 记录 |
| --- | --- | --- |
| MLE-STAR | 4 条轨迹；8 个初始候选、16 次消融、16 次局部改进、4 次合并训练，共 44 次 | [外层 0](mle-star/outer-0.json)、[外层 1](mle-star/outer-1.json)、[结构化候选方案](mle-star/proposals)、[训练记录](mle-star/trials)、[选择和集成权重](mle-star/selection.json) |
| Karpathy autoresearch | 4 条轨迹 × 11 轮；按验证指标 keep/revert | [11 轮决策](autoresearch/rounds)、[结构化候选方案](autoresearch/proposals)、[结果账本](autoresearch/results.tsv)、[训练记录](autoresearch/trials) |
| Staged A | 普通逐轮迭代，44 次 | [完整状态及候选说明](staged/A/state.json)、[候选代码](staged/A/artifacts)、[逐轮评测](staged/A/validation) |
| Staged B | 阶段规划和结构化复盘，44 次 | [计划及复盘](staged/B/state.json)、[候选代码](staged/B/artifacts)、[逐轮评测](staged/B/validation) |
| Staged C | 在 B 的流程上于复盘后重置原生会话，44 次 | [计划及复盘](staged/C/state.json)、[候选代码](staged/C/artifacts)、[逐轮评测](staged/C/validation) |
| Staged D | 允许额外诊断，44 次；本次实际请求和执行均为 0 次 | [计划及复盘](staged/D/state.json)、[候选代码](staged/D/artifacts)、[逐轮评测](staged/D/validation) |
| MLEvolve | 原生树搜索，44 次尝试；36 次调用训练 worker，28 次成功、16 次失败 | [44 个节点及方案和父节点](mlevolve/progress.json)、[候选代码](mlevolve/models)、[训练记录](mlevolve/trials)、[训练前拒绝](mlevolve/invalid) |

每条 `iterations.csv` 记录附有模型文件和原始记录的相对路径。MLEvolve 的节点列表保留原始顺序，不另行推定节点的时间顺序。训练曲线、epoch 和 GPU 分配时长保存在 `trials` 或 staged 的 `validation` 子目录；CSV 中缺失的训练信息应回查原始文件，不能按零处理。staged 的 `state.json` 同时保存 `plans`、`pre_reviews`、`final_reviews`、`records`，其中 `proposal` 是候选的公开方案说明。

MLE-STAR 最终方案是对候选分数标准化后加权集成。成员池有 5 个模型，其中仅 3 个权重非零：`refine-1-0-t0` 为 0.1，`init-1-t2` 为 0.405，`refine-0-1-t3` 为 0.495。集成评估不增加候选训练次数；集成和单模型分别报告。

## 原始评测器失败与 v2 重评

v1 强制训练分数必须等于编码向量与共享 `item_emb` 的内积，误拒绝合法的独立输出 embedding 和目标向量归一化。C 的 32–40 次在训练后被拒绝；42–44 次在训练前被拒绝。原始错误保存在训练日志中，主搜索记录仍保留当时的失败状态。

修复后，v2 检查分数有限性、形状、因果性、目标角色和集合独立性、batch 独立性以及公开候选评分的顺序和集合独立性；共享 embedding 和尾位置读取只作为诊断项。见 [修复说明](source-at-export/mle_star_codex/EVALUATOR_V2.md)、[回归审计](evaluator-v2/regression.json) 和 [旧源码快照](evaluator-v2/legacy-source)。

[v2 重评](evaluator-v2/reevaluation/report.md) 检查了 27 个已有检查点，包含 17 个原先失败但有检查点的候选；新增训练为 0。另有 16 个失败候选没有已训练检查点，保持缺失。先在验证集选择，再进行测试；没有使用测试分数反选。它恢复了已有权重的评测，不会恢复原搜索中被错误反馈消耗的预算或改变后续搜索路径。

| 方法 | 验证 HR@10 | 测试 HR@10 | v2 选择 |
| --- | ---: | ---: | --- |
| 原始基线 | 0.395 | 0.365055 | 原始模型 |
| MLE-STAR 单模型 | 0.655 | 0.540039 | 原始优胜者 |
| Karpathy autoresearch | 0.655 | 0.531270 | 原始优胜者 |
| Staged A | 0.670 | 0.547647 | 第 5 次 |
| Staged B | 0.715 | 0.548678 | 第 43 次 |
| Staged C | 0.665 | 0.540812 | 第 22 次 |
| Staged D | 0.685 | 0.545970 | 第 23 次 |
| MLEvolve | 0.670 | 0.534236 | 恢复候选 `1b54f127090f4d949a4afb8343b3bce7` |
| MLE-STAR 固定集成 | 0.680 | 0.567247 | 保持原权重 |

MLEvolve 原始优胜者为验证 0.665、测试 0.535139；v2 恢复候选以验证 0.670 被选中，尽管测试略低。原始与重评结果分开保存：[原始汇总](mlevolve/summary.json)、[重评选择](evaluator-v2/reevaluation/selection.json)。

## B C D 的阶段规划与诊断分析

完整分析见 [逐轮分析表](staged/iteration-analysis/iterations.csv) 和 [分阶段统计](staged/iteration-analysis/analysis.json)。本次阶段分配为 10、10、10、10、4；完整阶段在第 8 次后复盘，决定是否继续第 9、10 次。B/C/D 各有 5 份阶段计划、4 次中期复盘、5 次结束复盘，各 58 次 agent 调用；所有计划仅有版本 1，没有计划修订。

B 的阶段末验证分数依次为 0.620、0.660、0.675、0.685、0.715；C 为 0.615、0.635、0.665、0.665、0.665；D 为 0.605、0.645、0.685、0.685、0.685。12 个完整阶段的第 9、10 次共占 24 次尝试，仅 C 第 9 次和 D 第 19 次刷新当时最佳值；B 的 8 次阶段尾部尝试均未刷新最佳值。但 C 第 17 次仍有增益，不能据此断言阶段固定只需 4 次。

这些记录支持尝试以 2–4 次实验启动方向，再按结果滚动延长，把 10 次作为上限。D 没有实际调用额外诊断，不能把 D 的成绩归因于诊断调用。B 比 A 的测试 HR@10 高约 0.001032；单次运行且缺少单独归因模块消融，不能证明规划或归因模块有稳定收益。

## D 的目标向量归一化补充实验

在 D 第 23 次的骨干上使用 `e / clamp(norm(e), 1e-8)^beta`，同时应用于训练正负例和公开候选评分；历史输入 embedding 的查表路径保持原实现。归一化的梯度参与训练。模型与执行代码见 [归一化源码](source-at-export/nextitnet_target_norm)、[4 个冻结候选](d-target-norm/models) 和 [运行报告](d-target-norm/report.md)。

| beta | 验证 HR@10 | 测试 HR@10 | 验证选择 |
| --- | ---: | ---: | --- |
| 0 | 0.685 | 0.545970 | 保留 |
| 1 | 0.675 | 0.534494 | 否 |
| 0.5 | 0.655 | 0.546099 | 否 |
| -0.5 | 0.640 | 0.536041 | 否 |

4 个系数在训练前固定，各训练和测试一次，独立于原有 44 次预算。0.5 的测试命中比对照多 1 个，但验证更低，仍选择 beta 0。此次结果没有显示归一化改善验证指标。

beta 0 复跑与旧 D 的每位验证用户目标排名和全部验证指标一致，但原先过严的浮点分数容差在 4 次训练后中断流程。随后仅修复审计容差、验证冻结文件和检查点哈希并补齐测试，没有重新训练。最大绝对分数差为 0.0000470281，最终容差为 `atol=1e-4, rtol=1e-5`，并同时要求目标排名完全相同。见 [审计修复记录](d-target-norm/audit-repair.json)、[对照审计](d-target-norm/control-audit.json) 和 [修复前 runner](d-target-norm/run-before-audit-repair.py)。

## 预算纠正与执行恢复

最初 staged 误设 2000 次预算的运行已停止，4 次已完成、4 个中断或待处理槽单独计账，不纳入 308 次主对照。其状态及候选保存在 [已停止运行](staged/stopped-2000-budget)，纠正依据见 [预算纠正](staged/budget-correction.json)。正式对照没有复用旧候选结果，仅复用基线校准资产。

staged 汇总器曾假定有 5 个重复，实际协议为 1 个重复；[恢复记录](staged/report/recovery.json) 和 [报告恢复脚本](staged/generate_report.py) 仅重建报告，没有额外训练或 agent 调用。MLEvolve 最终测试的接口错误和 CUDA 执行恢复分别保存在 [首次恢复](mlevolve/final-test-recovery.json) 与 [GPU 恢复](mlevolve/final-test-recovery-gpu.json)，原始搜索记录未被覆盖。

## 源码版本与完整性

| 来源 | 固定提交 |
| --- | --- |
| [RecHarness 上游](https://github.com/6lyc/RecHarness) | `74a927fefaf7d120c52b6e3a0809958bd33984f0` |
| [Google ADK MLE-STAR 实现](https://github.com/google/adk-recipes) | `8c0862c103e36e1bb503a87bc1b134ddf6319d62` |
| [Karpathy autoresearch](https://github.com/karpathy/autoresearch) | `228791fb499afffb54b46200aca536f79142f117` |
| [MLEvolve](https://github.com/InternScience/MLEvolve) | `f4a37b96c948bfed6564e62f3229e0b38e908d33` |

MLE-STAR 和 autoresearch 是 NextItNet 任务适配，MLEvolve 使用上游搜索引擎和任务适配器；不是原论文基准的直接复现。[导出时源码](source-at-export) 是上传时的适配器和测试快照，[修复前源码](evaluator-v2/legacy-source) 是 v1 评测器修复前的存档。具体历史文件是否相同，应以每次运行冻结的哈希为准，不能把上传时源码视为所有历史运行的逐字版本。

[manifest.json](manifest.json) 列出所有发布文件的大小和 SHA256；[copy-sources.json](copy-sources.json) 映射逐字复制文件在原工作区的来源。原始 JSON 中的绝对路径保留为溯源字段，下载到其他目录后不应直接作为可执行路径。模型权重、逐用户数据、预测数组、向量索引、凭据和原生私有会话未包含在记录包内。

无需 GPU 或第三方库即可核对发布内容：

```bash
python3 docs/experiments/movies-nextitnet-seed2024/verify_records.py
```

从原始工作区重新导出可使用 [导出脚本](export_records.py)；它仅复制白名单记录并生成索引，不会启动训练或调用模型。README、验证脚本和完整性清单属于发布时添加的文件。
