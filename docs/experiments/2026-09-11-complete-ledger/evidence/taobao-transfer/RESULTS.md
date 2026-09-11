# 当前最优迭代模型迁移到淘宝

将KuaiRec当前AUC最优r7_small_attention_first的结构迁移到Taobao_Action，四任务从头训练，未加载KuaiRec权重。共享两轮N/S block、首次额外attention和全8Query；新增四任务独立预测头。

数据来自UniRank列出的[salmon1802/Taobao](https://huggingface.co/datasets/salmon1802/Taobao)，固定版本fd01fb8edbfa58bb4b0d6556f1d5838074b0dbe5。训练17822296条、验证2871767条；同seed20262027、epoch1、batch8192、max_len100、BF16、AdamW1e-4/Adagrad.05。无新增seed，无测试集评估。

| 模型 | 点击AUC | 加购AUC | 收藏AUC | 购买AUC | 平均AUC | ΔCross8 | ΔCross8同clip |
|---|---:|---:|---:|---:|---:|---:|---:|
| taobao_cross8 | 0.660003 | 0.860225 | 0.875396 | 0.888655 | 0.821070 | +0.000000 | -0.001979 |
| taobao_cross8_clip1 | 0.661438 | 0.861456 | 0.878362 | 0.890941 | 0.823049 | +0.001979 | +0.000000 |
| taobao_cross11 | 0.657694 | 0.855578 | 0.875769 | 0.884216 | 0.818314 | -0.002755 | -0.004735 |
| taobao_r7_loop2 | 0.659763 | 0.855817 | 0.871620 | 0.884498 | 0.817925 | -0.003145 | -0.005125 |

平均AUC为四个任务等权平均，与原配置monitor一致；各任务都单独报告。Cross8原始clip10，另加clip1对照；Cross11和r7使用clip1。

| 模型 | 点击Logloss | 加购Logloss | 收藏Logloss | 购买Logloss | 平均Logloss |
|---|---:|---:|---:|---:|---:|
| taobao_cross8 | 0.182595 | 0.019116 | 0.014386 | 0.008151 | 0.056062 |
| taobao_cross8_clip1 | 0.182632 | 0.018405 | 0.013938 | 0.007360 | 0.055584 |
| taobao_cross11 | 0.183180 | 0.019251 | 0.014458 | 0.007993 | 0.056221 |
| taobao_r7_loop2 | 0.182789 | 0.019142 | 0.014508 | 0.008098 | 0.056134 |

| 模型 | 参数/M | 推理p50/ms | 训练含验证/s | 峰值显存/GiB | 更新数 |
|---|---:|---:|---:|---:|---:|
| taobao_cross8 | 54.764 | 11.879 | 1499.7 | 15.24 | 2193 |
| taobao_cross8_clip1 | 54.764 | 11.878 | 1499.7 | 15.24 | 2193 |
| taobao_cross11 | 56.940 | 13.592 | 1593.3 | 17.91 | 2193 |
| taobao_r7_loop2 | 53.648 | 11.017 | 1490.3 | 14.79 | 2193 |

推理在同一GPU、BF16、batch1024，10次预热/40次测量；不含IO。参数统计包含原实现保留但未启用的任务residual gate，不能全部视为活跃计算。

## 数据协议边界

沿用发布方的编码、样本切分和历史，不将此数据集的AUC与KuaiRec直接比较。cart/fav/buy是同用户、类目、品牌在曝光后24小时内的行为监督，并非广告ID级直接转化。历史action包含这些延迟标签，其请求时可用性未建立；同时间戳事件可能按行序进入历史。该结果用于发布协议内的结构比较，不证明在线因果收益。
[原始元信息](../data/raw/Taobao-unirank/meta_data.json) · [数据审计](reports/data-audit.json) · [协议记录](reports/protocol.json)

CPU/GPU各4项测试验证四头更新、优化器归属、N S A N S调用、padding/目标action隔离和checkpoint恢复。真实数据smoke与完整训练记录分别保存。
[计划](PLAN.md) · [配置](configs.json) · [完整结果](reports/summary.json) · [训练状态](reports/training/status.json)

## 首轮结论

淘宝当前最佳为Cross8＋clip1：平均AUC=0.823049127，点击/加购/收藏/购买分别0.661438/0.861456/0.878362/0.890941。原始Cross8平均AUC=0.821069672；裁剪设置改变带来+0.001979455，不应归因于循环结构。

KuaiRec最优r7迁移后平均AUC=0.817924507，比原始Cross8低0.003145164，比同clip1 Cross8低0.005124619。用户配对bootstrap500次的平均AUC差95%区间分别为[-0.004290,-0.002027]、[-0.006203,-0.004112]。相对同clip1 Cross8，四任务区间均小于0；相对Cross11平均差区间包含0。区间仅条件于当前checkpoint和数据协议，不能代替训练seed波动评估。

r7推理p50=11.017ms，原始Cross8=11.879ms，约快7.3%，保留为质量/速度取舍；未成为淘宝AUC首选。源数据审计发现15876122条训练/验证样本的前一个历史事件与目标同时间戳，因此结果严格限定为发布协议内的比较，不证明请求时可用性。

按最新要求，下一轮测试增加token宽度384/512和共享历史阶段循环，使用只增宽/只加循环对照，并保留增宽r7。所有新实验仍用seed20262027。

[用户配对bootstrap](reports/paired-bootstrap.json) · [最终审计](reports/final-audit.json)
