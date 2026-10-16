# FML 任务评估器契约与真实集成验收

控制器只接收受信任的外部评估器结果。三个 FML 任务的真实实现已位于
`staged_research/fml/`，通过 `run_fml.sh` 自动安装环境、下载数据、测量基线并运行；
操作见 [FML_SETUP.md](FML_SETUP.md)。用户无需编写或绑定评估脚本。
本页同时保留通用 Docker/本地评估契约；本项目不提供预构建的 Docker 镜像。

内置评估器先在子进程中训练候选，再由无标签预测子进程产出 logits/OOD scores，
最后由未导入候选的父进程计算指标。Opacus 从实际 accountant history 独立重算隐私预算。
最终测试校验源码/检查点身份，仅调用推理分支。Linux bubblewrap 模式还对候选子进程
做内层隔离；普通进程模式仍不强制文件不可访问，不能声称抵抗恶意 Python。

无 Docker 的服务器使用 [本地进程后端](SERVER.md)：下文容器路径依次对应
`{candidate}`、`{data}`、`{result}`、`{scratch}`、`{checkpoint}` 占位符。
指标与检查点契约相同；进程模式的只读/不可见/禁网条件需要外部系统权限保证，
或使用并验收可选 bubblewrap，不能直接沿用容器的隔离保证。

## 目录与调用

验证容器：

| 路径 | 权限 | 内容 |
|---|---|---|
| `/candidate` | 只读 | 公开基线文件与 agent 修改后的版本 |
| `/data` | 只读 | 训练数据、开发验证数据；不得包含隐藏测试标签 |
| `/result` | 可写 | 可信评分结果、训练产物、日志 |
| `/tmp` | 临时 | 实验 scratch；结束后删除 |

测试容器额外挂载 `/checkpoint`，是验证阶段选出的精确训练产物，或登记的未改基线
检查点；`/data` 换成隐藏测试数据。不挂载验证数据目录，不向 agent 暴露测试输出。
验证和测试均无网络，使用指定 GPU。镜像必须已经包含完整依赖、上游源码和评估器。
调用为 argv 数组，禁止把 agent 文本拼接成宿主机 shell。

任务适配器将 `/candidate` 的允许文件覆盖到临时上游工作目录，固定其余实现和数据划分。
验证阶段只训练/验证，不能执行最终测试。候选训练子进程退出后，由独立的可信评分过程
计算指标；不要信任候选直接打印的 `val_bpb`、AUROC 或自报的隐私约束。
评估器和测试数据不得被放进公开 source，验证镜像及数据也不能预装隐藏标签。
为保持统计独立性，所有方法/重复采用相同任务随机设置与基线，不允许 agent 修改 split。

成功验证必须写入：

```json
{
  "metric": 0.75,
  "constraints_ok": true,
  "training_seconds": 123.4
}
```

文件为 `/result/metrics.json`；上述数字仅为接口示例。
同时将完整训练检查点写入 `/result/checkpoint/`，不能是空目录或符号链接。
指标统一为 [0,1] 小数。执行异常非零退出；缺失指标、超时、NaN、约束失败都记为无效调用，
不能计为“假设已经证伪”。缺失 training_seconds 会明确记为计时缺失。

Opacus 必须额外提供受信任的隐私会计结果：

```json
{
  "metric": 0.75,
  "constraints_ok": true,
  "training_seconds": 123.4,
  "constraint_checks": {
    "target_epsilon": 8,
    "target_delta": 0.00001,
    "actual_epsilon": 7.99,
    "actual_delta": 0.00001
  }
}
```

测试镜像只载入 `/checkpoint`，以相同模型定义计算隐藏指标，写同样的结果接口。
它不能重新搜索或调整模型，也不重新训练。测试发生一次；失败轨迹保留，不重试测到好分数。

## 各任务的适配要点

- **ColoredMNIST**：固定 env2 的验证/测试划分、种子与 5001 次更新；沿用固定 ERM 接口，
  将 agent 修改的 `domainbed/algorithms.py`、`domainbed/networks.py` 接入训练。
- **OpenOOD**：保持 CIFAR-100 作开发 OOD、SVHN 作隐藏 OOD，分离 CIFAR-10 的隐藏测试
  标签。内置准备器已在训练前物化独立数组，验证不加载 SVHN/隐藏 ID。
  agent 修改 `algorithm.py`，split_config 固定。训练检查点在一次验证后保留，最终测试只加载。
- **Opacus**：agent 修改 `algorithm.py`，split_config 固定。目标 ε=8、δ=1e-5；从实际
  PrivacyEngine/accountant 读取预算，验证模型兼容性和训练配置，不信任 agent 自报。

## 冻结前的服务器集成验收（完整 GPU/数据尚未执行）

1. 检查验证镜像、数据和 public source 内没有隐藏测试标签；保存文件清单与审核证据。
2. 在同一任务基线上复现原始验证和测试指标，保存检查点，并记录准备成本。
3. 验证候选进程拿不到其他任务、其他轨迹和 controller 的私有文件；原生 agent 容器不挂数据。
4. 用已知有效候选检验任务接口；用 NaN、越界指标、非法隐私预算和超时检验拒绝路径。
5. 检查测试模型与验证选中版本/检查点一致；最终测试不重新训练，测试结果不会被后续 agent 读取。
6. 检查原生 CLI 的 session resume、fresh session 与 token 回执格式确实符合已固定镜像。

服务器入口会先执行开发数据的小规模真实模型检查和三任务完整基线，再开始正式评测。
开发机的真实 CPU/小规模数据检查不能替代服务器 CUDA/完整数据的验收；实际检查记录
与边界见 [VERIFICATION.md](VERIFICATION.md)，不会用合成数据指标充当 FML 成绩。
