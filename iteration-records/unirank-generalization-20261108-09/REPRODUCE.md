# 同协议重训练

此目录是结果和源码归档，不含原始数据与训练权重。原始 JSON 中的绝对路径仅为来源证据。使用独立工作目录重建路径，可避免修改归档凭据。以下操作不会由结果发布程序自动执行。

## 1. 环境与基准代码

使用支持 BF16 的 CUDA GPU；原实验为 A800 80GB，每个完整训练单独占用一张卡。确切 Python、PyTorch、NumPy、pandas、PyArrow、scikit-learn 版本及驱动见 [environment.json](evidence/environment.json)。`source/train.py` 还需要 UniRank 所导入的依赖。原环境没有安装未使用的可选 heavyball。

```bash
git clone https://github.com/salmon1802/UniRank.git /path/to/UniRank
git -C /path/to/UniRank checkout 90bb99a60724e975ec565836266fd86dcad0039d
python -B tools/verify_export.py .
python -B tools/prepare_replay.py --bundle . --workspace /path/to/unirank-replay --upstream /path/to/UniRank
```

`prepare_replay.py` 只复制冻结源码、特征映射和配置，并改写磁盘路径；不会训练或下载。它把本地 acquisition 状态设为未就绪，避免将历史下载凭据误认作当前机器已有数据。建议同时核对 `evidence/upstream-source-identity.json` 中核心上游依赖的 SHA256。

## 2. 取回固定数据版本并逐文件校验

共约66GiB。各数据集发布 revision、文件大小、LFS SHA256 均在 datasets/ 对应目录。下列命令以 QK-Video 为例；其他名称为 KuaiRand、TAAC-25、MerRec。

```bash
python -B tools/acquire.py --folder /path/to/unirank-replay/datasets/QK-Video --workers 8
```

应等到 acquisition-status.json 为 downloaded_and_verified 后再训练。该下载器仅请求作者的公开固定发布版本；不从测试结果构造输入。

## 3. 完整训练、验证选择和最终测试

```bash
CUDA_VISIBLE_DEVICES=0 python -B /path/to/unirank-replay/source/train.py --root /path/to/unirank-replay --dataset QK-Video --kind candidate --output /path/to/unirank-replay/runs/QK-Video/candidate/fit-01
CUDA_VISIBLE_DEVICES=1 python -B /path/to/unirank-replay/source/train.py --root /path/to/unirank-replay --dataset QK-Video --kind din --output /path/to/unirank-replay/runs/QK-Video/din/fit-01
```

替换 dataset 即可运行其余三组。每组先按发布训练集训练1epoch，验证集全任务平均AUC选择 checkpoint，写入选择凭据及 checkpoint SHA256，再进行完整测试。源脚本拒绝覆盖已完成的 result.json，复跑需新输出目录。完整参数来自各 datasets/*/protocol.json 与模型配置，不再搜索超参数。

如需复现20步成本标定，向同一训练命令添加 `--calibration-steps 20` 并改用独立输出目录；标定不访问验证/测试。正式训练再次使用原始种子从头初始化，不接续标定权重。

## 证据的范围

- 原始模型的初始化、前向和反向等价检查，以及本地输入审计保存在 evidence/。这些是当时工作区的检查记录；`source/test_contract.py` 中的原淘宝历史路径不是跨机器通用测试入口。
- `source/configure.py` 与 `source/supervise.py` 是原始执行源码的完整冻结副本，用于审计。重建路径使用本目录的 prepare_replay.py。
- 不同设备和依赖版本不承诺位级复现。单种子结果不构成重复种子波动或显著性证据。
