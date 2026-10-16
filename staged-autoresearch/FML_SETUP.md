# 三个内置 FML 任务：环境、数据、评估器与一键运行

内置 ColoredMNIST、OpenOOD、Opacus 的真实训练、预测、检查点与评分实现。
**不需要你编写评估脚本，也不需要填写评估器路径或手工输入基线分数。**
脚本面向 Linux x86_64 GPU 服务器，无 Docker；需要系统已有 Python 3.9+、git 和联网能力。
环境、代码和数据全部安装在指定 prefix 内，不使用 sudo，不覆盖系统 Python。

## 最短使用方式

在 `staged-autoresearch/` 目录，配置 Claude Code 兼容模型服务的 `ANTHROPIC_API_KEY`。
使用明确模型 ID，下面的 `YOUR_FIXED_MODEL_ID` 替换成你的版本：

```sh
./run_fml.sh \
  --prefix /srv/staged-fml \
  --model YOUR_FIXED_MODEL_ID \
  --gpu 0
```

该命令依次自动完成：

1. 下载校验固定版本 micromamba，创建三个独立任务环境。
2. 获取固定 commit 的 FML 与 DomainBed，导出公开基线源码。
3. 下载 MNIST、CIFAR-10、CIFAR-100 和 SVHN，生成固定、互不重叠的训练/验证/隐藏测试数组。
4. 检测原生 Claude Code；没有时在 prefix 内安装固定版本 Node/CLI，不调用模型。
5. 用少量**开发数据**执行三个真实模型的小规模训练/检查点检查，不使用真实隐藏测试。
6. 各训练一次完整任务基线，保存检查点，并测量基线验证和测试指标；准备成本另记。
7. 自动生成 `server.json`、预检、冻结，执行四组×三任务×五次运行，最后生成报告。

正常正式评测最多 6000 次训练/验证调用，另有 60 次最终测试；前置小规模检查和三次
完整基线训练不混入这 6000 次，所有记录保留。**运行此命令会开始真实训练和模型调用。**
中断后重执行相同命令；已完成的数据、基线和轨迹不会重新训练。
结果未知的基线/隐藏测试不会自动重跑，脚本会停下要求核对记录。

如果数据已下载，在同一命令增加 `--data-dir /your/torchvision-cache`。
如已有原生编码 CLI，增加 `--agent-binary /absolute/path/claude`；否则脚本会自动安装。
使用模型代理时可以预先配置 `ANTHROPIC_BASE_URL` 及 HTTPS 代理环境变量，密钥不写入配置。

## 分开下载、检查、训练

只查看准备计划，不安装、不下载、不调用模型、不训练：

```sh
./run_fml.sh --prefix /srv/staged-fml --stage plan
```

只准备环境、源码、数据和配置，**不做任何训练**，不需要模型 API 密钥：

```sh
./setup_fml.sh --prefix /srv/staged-fml --model YOUR_FIXED_MODEL_ID --gpu 0
```

只下载/恢复数据，使用已经创建的任务环境：

```sh
./download_fml_data.sh --prefix /srv/staged-fml --data-dir /your/torchvision-cache
```

只执行真实模型的小规模开发数据检查，不调用模型 API、不运行完整基线：

```sh
./run_fml.sh --prefix /srv/staged-fml --model YOUR_FIXED_MODEL_ID --gpu 0 --stage smoke
```

只完成小规模检查和三任务完整基线，不启动 agent 搜索：

```sh
./run_fml.sh --prefix /srv/staged-fml --model YOUR_FIXED_MODEL_ID --gpu 0 --stage calibrate
```

完成准备后，也可以使用原有底层入口：

```sh
./evaluate.sh --config /srv/staged-fml/server.json \
  --root /srv/staged-fml/study --output /srv/staged-fml/report
```

`--keep-going` 在某条轨迹失败后继续其他单元。固定模型、源码、后端和环境发生变化时
使用新 prefix，不修改已冻结研究。普通 Ctrl-C/SIGTERM 清理后可继续；进程身份或原生
CLI 会话状态不明确时会保留现场，不能用自动换会话偷偷改变 A/B 条件。

## 数据位置与划分

现有下载缓存应符合 torchvision 布局：

```text
torchvision-cache/
  MNIST/raw/...
  cifar-10-batches-py/...
  cifar-100-python/...
  svhn/test_32x32.mat
```

没有缓存也可不传 `--data-dir`，使用 `prefix/raw-cache/` 自动下载。torchvision 检查
原始文件校验值；准备数组记录 SHA256，正式运行禁止临时下载数据。
不把完整原始缓存复制进验证目录。

| 任务 | 训练 | 开发验证 | 隐藏测试 |
|---|---:|---:|---:|
| ColoredMNIST | env0 16,334＋env1 16,334 | env2_out 6,999 | env2_in 16,334 |
| OpenOOD | CIFAR-10 可见池中的 49,979 | ID 3,021＋CIFAR-100 OOD 10,000 | ID 7,000＋SVHN OOD 5,000 |
| Opacus | CIFAR-10 34,997 | 4,503 | 10,500 |

ColoredMNIST 沿用拼接 MNIST 60k＋10k 后的领域构造、标签噪声、颜色翻转和官方种子。
OpenOOD 固定 seed=42 切出 CIFAR-10 原测试集的 70% 为隐藏 ID；剩余 30% 与训练集组成
可见池，再按固定 val_ratio=0.057 分割。Opacus 固定 seed=42，从原 50k 训练数据先取
21% 隐藏，再以 val_ratio=0.114 划分剩余可见池。split_config 不可更改。

## 真实评估器如何工作

- 验证阶段：复制固定源码，接入允许修改的文件；训练子进程只获得训练输入。
  训练完成后保存真实权重与重建参数，再启动独立预测子进程。
- 预测子进程只接收特征，不接收评估标签。可信父进程在子进程退出后计算准确率或 AUROC/FPR95，
  不采用候选打印的成绩，不在评分进程里导入候选 Python。
- 最终测试：检查选中源码与检查点身份，只重建模型、加载权重和预测；没有训练分支。
- Opacus：固定 ε=8、δ=1e-5，校验模型兼容性；保存实际噪声、采样率和会计步数。
  评分进程使用独立 PRV accountant 重新计算 ε，并核对训练步数；超预算即无效。
- Opacus 用 BatchMemoryManager 将物理 batch 默认限制为 64，保持逻辑 batch 和 Poisson
  采样/会计语义，减少显存需求。小规模检查使用单独的 smoke 配置，不参与研究成绩。

使用相同的 FML 任务定义、划分与默认训练量，但这是受控任务适配，**不是上游脚本逐行复现**。
尤其原 DomainBed 会在训练中计算所有领域的隐藏指标，原 wrapper 最终读取已算出的隐藏分数；
这里改为仅保存模型、最后按需评分，取消训练中隐藏/中间评估，也改变了部分随机数消耗顺序。
必须使用本机重新测得的基线，不能套用上游 `baseline_results`。

## 固定环境与资源

| 任务 | Python | PyTorch / torchvision | CUDA wheel | 主要依赖 |
|---|---|---|---|---|
| ColoredMNIST | 3.10.16 | 1.12.1 / 0.13.1 | 11.3 | NumPy 1.22.4、timm 0.9.16 |
| OpenOOD | 3.9.21 | 2.5.1 / 0.20.1 | 12.1 | NumPy 2.0.2、scikit-learn 1.6.1 |
| Opacus | 3.10.16 | 2.5.1 / 0.20.1 | 12.1 | Opacus 1.5.4、NumPy 2.2.6 |

环境设置参照固定 FML 源码的任务环境，保存实际 pip freeze、conda 安装记录和文件哈希。
系统 NVIDIA 驱动须兼容对应 PyTorch/CUDA wheel；脚本检测可用 GPU，不能自动安装驱动。
默认面向包含 V100 在内的 NVIDIA GPU；V100 需要使用支持 CUDA 12.1 的系统驱动，
不能仅凭装有 CUDA toolkit 判断驱动可用。是否能实际训练由三个任务的小规模检查确认。
`--cpu` 可明确创建 CPU 环境，全部方法一致使用 CPU，速度较慢。不要将 CPU 与 GPU 的结果
混在同一冻结研究里。默认每次训练/评分超时 7200 秒，正式冻结前可统一调整配置。

完整研究会保留每次检查点，磁盘占用取决于候选模型大小；基线 ResNet-18 每份约 45 MB，
仅 OpenOOD 的 2000 份检查点就约 90 GB，应为完整研究预留足够空间。

## 产物与故障定位

```text
prefix/
  envs/                 # 独立任务环境、冻结依赖清单
  upstream/             # 固定 FML/DomainBed 源码
  public/               # agent 可见基线源码
  raw-cache/            # 私有原始下载缓存
  data/<task>/{validation,test}/
  logs/                 # 安装、下载日志
  smoke-runs/           # 小规模真实模型检查（仅开发数据）
  calibration/          # 基线预扣记录、模型、指标和成本
  server.json           # 自动生成的完整运行配置
  study/                # 正式实验轨迹与证据
  report/               # summary.json、curves.csv、report.md
```

安装失败看 `logs/*-environment.log`，下载/划分失败看 `logs/*-data.log`；评估器失败看对应
结果目录的 `adapter_scratch/*.stderr.log` 和 `transport.json`。不吞掉非零退出码，也不把失败
训练记成科学反证。完整基线或隐藏测试状态不明时不删除记录重跑到成功。

## 文件隔离与当前验证边界

普通进程模式没有操作系统层面的文件/网络隔离。同一用户下候选仍可能主动访问其他路径，
不能宣称抵抗恶意代码。严格环境可用 `--isolation bubblewrap`：外层隔离 agent/任务，
内层候选训练/预测再隔离，只挂训练数组或无标签特征；评分标签留在外部评分进程。
需要 Linux 已安装 bwrap 且允许嵌套用户命名空间；预检失败时不自动降级。

这些检查提高普通研究代码的执行可核查性，不构成针对任意恶意 Python 的隐私证明。
独立 PRV 重算也依赖受信任训练器与 Opacus 实现未被恶意篡改。
开发机上的小规模 CPU 集成不能替代目标 Linux/CUDA 组合的验收，服务器入口会再执行
真实模型的小规模检查及完整基线；没有用开发机的合成数据成绩冒充 FML 成绩。

上游版本、许可与适配差异见 [THIRD_PARTY.md](THIRD_PARTY.md)，
研究方案见 [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md)。
