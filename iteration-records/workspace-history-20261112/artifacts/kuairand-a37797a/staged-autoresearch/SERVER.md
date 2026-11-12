# 无 Docker 服务器：本地进程后端与一键评测

**ColoredMNIST、OpenOOD、Opacus 已有内置真实评估器与安装/数据脚本，直接使用
[FML_SETUP.md](FML_SETUP.md)，无需手写下文的评估脚本。**
本页保留通用后端接口，适合自定义任务或已有评估服务。

这个后端直接调用服务器上的原生 Claude Code、任务 Python 环境和可信评估器。
不启动 Docker、不下载模型/数据、不安装训练依赖。各任务可以用不同的 venv/conda
Python；调度器本身只需 Python 3.9+ 与 POSIX 系统。正式评测建议 Linux GPU 服务器。

一键入口会依次执行：配置与环境预检 → 首次冻结 → 顺序运行/继续 60 条轨迹 → 导出报告。
它保留原四组设计、100 次预算、10 阶段及每轨迹一次最终隐藏测试。
自定义评估器需要先准备依赖、数据和基线；内置任务由 `run_fml.sh` 自动完成这些步骤。

## 1. 首次配置

在仓库的 `staged-autoresearch/` 下生成本地配置：

```sh
python3 -m staged_research init --backend local --root /srv/research/fml-study
```

编辑 `/srv/research/fml-study/study.draft.json`，也可以将它作为服务器配置文件保存到另一
位置。所有路径都使用服务器上的绝对路径；配置中不写 API 密钥。关键字段如下：

```json
{
  "execution": {
    "backend": "local",
    "isolation": "process",
    "environment_id": "fml-fixed-environment-v1",
    "gpu_device": "0",
    "pinned_files": [
      "/opt/agent/bin/claude",
      "/opt/envs/domainbed/bin/python",
      "/opt/evaluators/domainbed_validate.py",
      "/opt/evaluators/domainbed_test.py",
      "/opt/evaluators/environment-lock.txt"
    ],
    "readonly_roots": [],
    "agent_env_passthrough": []
  }
}
```

这是字段片段，不是完整配置。应在生成的完整草案中填写：

- `agent.binary`：原生 Claude Code 的绝对可执行路径；与 `pinned_files` 中登记的路径一致。
- `agent.model`、`agent.version`：明确模型 ID、已核对的 CLI 版本；其余各组共用。
- `sources.fml_commit`：固定 FML 源码版本；已有工作区可先使用 README 的 `prepare-fml` 导出。
- 三个任务的公开源码、验证数据、隐藏测试数据、基线检查点及实测基线指标。
- 各任务的 `validation_argv`、`test_argv`：可信评估器的参数数组，见下节。
- `execution.pinned_files`：全部 CLI/解释器/评估脚本与环境锁文件。上面的列表仅展示一个任务，
  需要补齐另外两个任务。首次冻结保存这些文件的 SHA256，再运行时检查有无改变。
- 每个任务的隔离与外部评分审核说明；不能用填写 `true` 代替实际审核。

`environment_id` 用来标记固定环境。只哈希列出的文件，不会自动冻结全部共享库与依赖；
应保存实际环境锁文件并维护环境。配置变化使用新的 study 目录，不覆盖旧登记。
`gpu_device` 是本机物理卡序号或 GPU UUID，明确写入配置；不会继承外层临时
`CUDA_VISIBLE_DEVICES` 改变已登记的设备。请与服务器调度器分配保持一致。

## 2. 接入任务的本地评估命令

每个任务可以指定独立 Python 环境。参数按数组执行，不经过 shell；支持下列占位符：

| 占位符 | 验证 | 隐藏测试 |
|---|---|---|
| `{candidate}` | 本次候选源码 | 验证集选中的最佳源码 |
| `{data}` | 训练/开发数据目录 | 隐藏测试数据目录 |
| `{result}` | 此次输出目录 | 私有测试输出目录 |
| `{scratch}` | 本次临时工作目录 | 本次临时工作目录 |
| `{checkpoint}` | 不可用 | 验证选中的精确检查点 |

示例（填写到一个任务的配置中）：

```json
{
  "validation_argv": [
    "/opt/envs/domainbed/bin/python", "/opt/evaluators/domainbed_validate.py",
    "--candidate", "{candidate}", "--data", "{data}", "--result", "{result}"
  ],
  "test_argv": [
    "/opt/envs/domainbed/bin/python", "/opt/evaluators/domainbed_test.py",
    "--candidate", "{candidate}", "--data", "{data}", "--result", "{result}",
    "--checkpoint", "{checkpoint}"
  ]
}
```

示例里的脚本名称表示自定义任务适配器，不是内置 FML 入口；内置入口由脚本自动生成。
评估器必须在验证后写 `{result}/metrics.json` 和非空 `{result}/checkpoint/`；隐藏测试
只加载 `{checkpoint}`，不能重新训练或搜索。指标、Opacus 隐私会计和三个任务的约定见
[EVALUATORS.md](EVALUATORS.md)。执行工作目录为 scratch，评估器应在其中准备完整固定
任务源码，再覆盖候选的可编辑文件；不要在公开源码目录原地训练。

训练环境只提供最小 PATH、HOME、语言、临时目录、指定 GPU 等变量，不继承模型 API
密钥、宿主机 PYTHONPATH 或用户 CLI 配置。依赖使用明确的 Python 和库安装位置。
原生 agent 使用单独 HOME/CLAUDE_CONFIG_DIR，保留原生编辑工具和结构化输出。
如代理服务需要地址或 HTTPS 代理，可在 `agent_env_passthrough` 显式登记允许继承的变量名：
`ANTHROPIC_BASE_URL`、`HTTP_PROXY`、`HTTPS_PROXY`、`NO_PROXY`、`SSL_CERT_FILE`。
`ANTHROPIC_API_KEY` 只传给 agent，不落入配置/训练环境；不要在服务地址中嵌入密钥。

## 3. 一键预检与运行

只检查，不冻结、不调用模型、不训练：

```sh
./evaluate.sh \
  --root /srv/research/fml-study \
  --output /srv/research/fml-report \
  --check-only
```

通过检查后，在服务器环境安全配置 `ANTHROPIC_API_KEY`，使用同一条命令去掉
`--check-only` 即可启动正式评测：

```sh
./evaluate.sh \
  --root /srv/research/fml-study \
  --output /srv/research/fml-report
```

如将完整配置保存在独立文件中，可加 `--config /srv/research/server.json`。首次会复制成
study 草案；已有草案或登记时必须逐项相同，不静默覆盖。配置未就绪时失败退出。
默认遇到轨迹异常即停止并尝试导出已有结果；`--keep-going` 会继续其他矩阵单元，
保留失败记录。完整成功退出码为 0；预检不通过或结果未齐备返回非零。
模型/任务异常与参数错误也返回非零。中断后重新执行相同命令继续，不另起免费预算。

可从任意工作目录调用 `evaluate.sh` 的绝对路径；如控制器需特定 Python，设置
`STAGED_RESEARCH_PYTHON=/absolute/python`。独立运行单元仍可用原 `run` 命令；所有 CLI
运行入口使用同用户、同设备的共享锁，避免本框架的不同 study 同时占用同卡。
此锁不管理其他用户、无关程序或外部 GPU 调度器；混用卡序号与 UUID 也不能视为同一锁。

## 4. 恢复、输出与限制

- 训练前先扣除实验次数，再保存进程身份，最后放行评估器。身份记录失败则不执行训练。
- 超时/中断停止整个进程组，包括正常继承进程组的训练子进程。禁止评估器脱离进程组、
  后台守护化或另外提交训练作业；这些应由独立调度后端接管，本地后端不能追杀任意远程作业。
- Linux 恢复用 boot ID 与进程启动时间校验 PID，避免误杀复用的 PID。身份无法核对时
  停止恢复并要求审计，不假装遗留 GPU 工作已经结束。更换主机恢复需额外核对工作进程。
- 失败、超时和恢复时结果未知的训练均不退还次数。隐藏测试未知时只清理遗留进程，不重跑。
- 在原生 agent 调用中断、CLI 会话写入状态不明时，需要核对该固定 CLI 版本的原会话
  恢复行为；不自动更换会话以绕过错误，否则会改变 A/B 的连续上下文条件。
- 已完成的轨迹不会再次调用 agent 或训练；事件/源码/检查点被篡改时拒绝继续。
- study 内有结构化轨迹、私有 CLI 回执、进程记录及逐次失败记录。报告目录输出
  `summary.json`、`curves.csv`、`report.md`；失败/缺失单元不删除。

### 普通进程与可选 Linux 隔离

`isolation=process` 无额外沙箱依赖，适合受信任的本地执行环境。**同一系统用户下，
进程隔离 HOME 与工作目录不能阻止读取其他目录、旧会话、隐藏数据或联网。**
新的 session 仍执行预定的上下文重置，但不能声称证明了“不可访问旧会话”。报告会明确
标注这个限制；正式实验若需要严格的信息边界，必须增加并验收操作系统隔离。

无 Docker 的 Linux 可设置 `isolation=bubblewrap`，安装系统 `bwrap` 并启用用户命名空间。
`readonly_roots` 仅登记确实需要的只读依赖目录，例如 `/opt/envs`；不能登记 `/srv`、
用户主目录等含 study/数据的宽泛目录。后端检查挂载范围与私有资产不重叠。
agent 仅挂载本轮 work、研究证据和当前 session；验证仅挂载候选、验证数据和输出；
测试换成隐藏数据并附选中检查点。评估器关闭网络，agent 保留模型服务所需网络。
GPU 设备节点和系统库按实际服务器准备；如 NVIDIA 库位于 `/usr` 外，应登记依赖目录。

预检会试运行一个无训练的隔离命令，确认 bwrap/用户命名空间可用；缺失时明确失败，
**不会自动降级到普通进程**。这仍不替代可信评估器、数据拆分和 GPU 环境的实际验收。
当前开发机只验证了 bubblewrap 参数构造，尚未完成 Linux GPU/bubblewrap 集成测试。
隔离参数依据 [bubblewrap 官方手册](https://github.com/containers/bubblewrap/blob/main/bwrap.xml)。
