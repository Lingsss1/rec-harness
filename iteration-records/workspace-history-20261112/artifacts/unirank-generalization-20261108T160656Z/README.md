# UniRank 跨数据集固定结构迁移实验

模型来自淘宝多任务验证审计的 g3-r0029-context_shared_primary。固定结构、固定超参数；在每个数据集从头训练，与配对 DIN 比较。此次不启动课题搜索。

原淘宝源码 SHA256：ef9ec33eb797346f91f52cc3445dc1b8fb5fb13206500881efa6acafcaa6ef09（以 interface-checks.json 的实际校验值为准）。

## 协议

- 使用作者发布的固定 revision、train/valid/test 分块及官方历史前缀语义；逐文件核验 LFS SHA256。
- 所有原生任务等权平均 AUC 选模，平均 Logloss 破平；逐任务 AUC/Logloss 全部保留。验证后冻结 checkpoint 及 SHA256，随后进行一次完整测试。
- 每个模型单独 GPU，1 epoch、batch8192、history100、seed20262027、embedding16、BF16、AdamW1e-4 + Adagrad0.05；每次完整训练最多48h。
- 完整数据核验后先运行20步成本标定；标定不访问验证/测试，正式训练恢复初始种子并从头开始。标定成本单独保留。
- 模型只有接口适配：任务列表/首任务角色、两个类别字段匹配槽。路由、门控、私有辅助 embedding 和梯度归属保持继承。
- TAAC 的 list 用户字段使用官方 FeatureProcessor 默认的 masked average pooling；DIN 和候选一致。词表以当前发布 metadata 为准，修正旧 YAML 中 TAAC104 的过期词表大小。
- 单种子初步验证；这是重训练后的结构迁移，不能解释为零样本迁移，也不能自动推断线上因果收益。

| 数据集 | 任务数 | 首任务角色 | 两个匹配字段 |
|---|---:|---|---|
|QK-Video|4|click|['video_category', None]|
|KuaiRand|6|is_click|['video_type', 'primary_tag']|
|TAAC-25|2|is_click|['100', '101']|
|MerRec|5|Like|['c2_id', 'brand_id']|

QK 缺失槽及对应交叉匹配槽为零。TAAC 的匿名字段不被解释为类目/品牌；KuaiRand 使用视频类型/标签。该映射在结果产生之前固定。

## 运行与产物

```bash
artifacts/unirank-env/bin/python artifacts/unirank-generalization-latest/source/dashboard.py --root artifacts/unirank-generalization-latest --once
/usr/bin/tmux -L ctr-research attach -t ctr-research
```

现有 tmux 会话新增“跨数据集泛化”窗口。若终端报 not a terminal，需先通过有 TTY 的终端或 ssh -tt 连接。

每个运行保存在 runs/<数据集>/<candidate|din>/{calibration-01,fit-01}/：train.log、progress.json、result.json、failure.json、selection.json、checkpoint.pt。
汇总由 report.json / report.txt 自动更新，包含每个任务、平均 AUC/Logloss、参数量、训练耗时；完整延迟/显存/历史长度切片在单次 result.json。

单次复现（请为复现指定新目录，已完成结果不可覆盖）：

```bash
CUDA_VISIBLE_DEVICES=0 artifacts/unirank-env/bin/python artifacts/unirank-generalization-latest/source/train.py --root artifacts/unirank-generalization-latest --dataset QK-Video --kind candidate --output /tmp/unirank-qk-candidate-repeat
```

将 kind 改为 din、dataset 改为其他三个名称可复现其余运行。所有原始 source/protocol、revision/hash 和失败记录保留；旧淘宝研究与测试结果不被修改。
