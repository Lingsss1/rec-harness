# 阶段规划、归因与方向选择研究

在 Karpathy 的原生编码 agent 循环外，加入可审计的阶段规划和方向评审。
这是仓库内的独立子项目，不依赖现有 RecHarness 运行时，也不修改历史实验。
Python 3.9+，控制器无第三方 Python 依赖。

**实现已包含调度器、Claude Code 原生编辑适配器、本地进程/Docker 后端、一键评测、
冻结登记、预算与恢复、记录和统计报告。当前未运行任何真实模型调用或 GPU 实验。**
真实任务需要另行准备并审核 FML 的本地评估器或验证/测试镜像、数据和基线检查点；本项目不会
把 FML 的现有评估脚本未经数据隔离直接包装成“安全镜像”。详见 [评估器契约](EVALUATORS.md)。
离线测试使用明确标注的合成响应，不是 FML 成绩，也不能替代真实模型/容器集成测试。

## 无 Docker 服务器一键入口

先生成并补齐服务器的模型、评估命令、数据和基线配置：

```sh
python3 -m staged_research init --backend local --root /srv/research/fml-study
```

配置完成后检查；去掉 `--check-only` 即顺序执行预检、首次冻结、运行/继续完整矩阵和报告：

```sh
./evaluate.sh --root /srv/research/fml-study \
  --output /srv/research/fml-report --check-only
```

支持独立配置文件 `--config /absolute/server.json` 和失败后继续其他单元的 `--keep-going`。
**[服务器部署、配置和评估命令接口](SERVER.md)** 说明完整操作。
普通进程模式不强制文件/网络隔离；严格实验可选 Linux bubblewrap，不能将目录约定视为
沙箱保证。本地后端无需 Docker，下面的镜像准备步骤仅适用于 Docker 后端。

## 设计目标

检验：在相同的训练/验证调用预算下，**提前规划实验、重新建立评审上下文、
按需花预算验证原因**，能否提高验证选出方案的最终隐藏测试成绩。

- B−A：检验阶段规划与阶段复盘这一整体干预。
- C−B：检验清空历史对话、依据保留证据重新评审的增量价值。
- D−C：检验受控归因实验是否值得占用原本可用于优化的两次预算。

方向切换多、报告长、接受改动多，都不作为成功标准。四组不能完整估计清空与归因的
交互效应；D 只优于 A 时，也不能单独归功于归因。
完整的研究问题、控制变量、阶段协议、指标、失败处理与结论边界见
**[实验方案与设计目标](EXPERIMENT_DESIGN.md)**。

## 固定实验设计

| 组 | 阶段计划 | 评审上下文 | 第 9、10 次实验 |
|---|---|---|---|
| A | 不强制 | 原生持续会话 | 自主提出下一实验 |
| B | 有 | 同一持续会话 | 依据记录分析继续优化 |
| C | 有 | 新会话，只读研究证据 | 同 B |
| D | 有 | 同 C | 按需做归因验证，否则继续优化 |

三个任务：ColoredMNIST、OpenOOD、Opacus。每任务/方法做 5 次独立 agent 运行，
每次 100 次验证调用。共 60 条轨迹、6000 次调用，另有每条轨迹一次最终隐藏测试。
基线训练/评估的准备成本应单独记录，不隐含在免费初始状态中。
运行顺序按固定种子 20261015 在每个任务/重复内打乱方法次序。

每阶段十次：先登记包含全部十个槽位和条件分支的方案；执行前八次；做证据评审；
执行两次归因或预定后续实验；做最终方向决策。C/D 在第八次后的评审和第十次后的
最终评审分别建立新会话，阶段间继续使用最后建立的会话。
固定节点只要求复核，不强迫换方向。

方向是瓶颈假设与主要干预机制的组合。continue/revisit 必须复用已有字段；
更换同一机制的参数范围不算新方向。字段身份校验不等于语义新颖性识别，
论文仍需盲审这些方向标注，不能把 agent 自报的切换次数作为成功标准。

保留原始代码编辑和显式 keep/discard 循环：指标严格改善时保留，否则回到 incumbent。
候选完整源码与父版本用内容哈希登记，替代要求 agent 自己操作 git。
原始 program.md 中“同分但更简单也可以接受”的自由裁量例外未纳入自动评分规则；
因此基线名称应为 **Karpathy 原生 agent 循环的受控任务适配**，不能称作原仓库零修改复现。
原指令完整保留在包资源内，来源 commit 为
`228791fb499afffb54b46200aca536f79142f117`，上游为 MIT 项目。

## Docker 后端：不调用模型的准备步骤

在本目录运行，不需要安装即可使用：

```sh
python3 -m staged_research init --root ./runs/fml-study
python3 -m staged_research matrix --root ./runs/fml-study
python3 -m staged_research preflight --root ./runs/fml-study
```

`init` 只生成 `study.draft.json`。默认镜像、模型、数据和基线指标为空，预检会失败；
这避免意外启动不完整或未冻结的实验。`matrix` 应显示 60 条轨迹、6000 次验证调用。

如果已有配置好的 FML checkout，可只导出公开代码和官方任务说明：

```sh
python3 -m staged_research prepare-fml \
  --root ./runs/fml-study \
  --checkout /absolute/path/FML-bench \
  --assets /absolute/path/fml-public-assets
```

这个命令不安装依赖、不下载数据、不训练。它读取既有 `workspace/`，核对官方可编辑
文件范围，登记 FML commit，只复制可编辑源码和官方 TASK.md。资产目录必须是新目录，
且与私有 study 目录分离。尚未建立 FML baseline workspace 时明确失败。

编辑草案，提供同一个明确模型 ID（禁用 sonnet/opus 等漂移别名）、CLI 版本、
所有容器镜像的 `@sha256:` 身份，以及三个任务的：

- `public_source`：只含公开参考代码及可编辑文件，不含 git、会话、数据或评估器。
- `validation_data`：训练/开发所需的数据，必须完全没有隐藏测试标签。
- `test_data`：私有测试数据，不出现在训练或 agent 容器。
- `baseline_checkpoint`：未改基线的检查点，初始方案最终获选时用于隐藏测试。
- `baseline_validation` / `baseline_test`：同一数据和协议下测得的基线指标，范围 [0,1]。
- 验证和测试的独立镜像与命令、各任务固定超时、可信外部评分和数据隔离的审核证据。

Opacus 隐私目标固定 ε=8、δ=1e-5。虽然 FML 原始任务允许部分任务修改数据划分，
本研究为了归因可比性固定 split_config.json，agent 不可修改它。
以源码/数据清单哈希冻结资产和控制器代码；冻结后变更需新建登记。

```sh
python3 -m staged_research preflight --root ./runs/fml-study --check-images
python3 -m staged_research freeze --root ./runs/fml-study
```

`freeze` 要求 Docker 与本地镜像存在，不会拉取镜像、启动模型或训练。
数据、评估器是否真的遵守契约还需独立审核；布尔声明不是隔离证明。

## Docker 后端：实际运行接口（本次未执行）

在准备好的 Docker/NVIDIA GPU 主机使用；Claude Code 镜像支持 `--bare` 和 JSON schema 输出，
通过环境变量 `ANTHROPIC_API_KEY` 认证，不读取宿主机 OAuth、钥匙串或用户配置。
密钥由 Docker 继承，命令参数和研究记录不写入其内容。

```sh
python3 -m staged_research run --root ./runs/fml-study \
  --task colored_mnist --arm D --replicate 1
python3 -m staged_research run-matrix --root ./runs/fml-study
```

第二个命令按登记矩阵串行运行，复用已完成轨迹。多 GPU 可以分别运行不同矩阵单元，
本版同一注册固定 GPU device=0；通过主机/容器设备可见性控制物理卡，不并发占用同一卡。
所有方法必须使用同样的执行方式。模型不自动降级，不自动更换提供商。

### 上下文、预算与恢复

- 原生 agent 只获得公开源码、研究证据和当前会话目录；没有 GPU、数据、宿主机目录或 shell 工具。
- 使用原生 Read/Glob/Grep/Edit/Write 编码，不由另一个 LLM 代理生成 patch。每次调用后检查实际文件差异，越界修改拒绝进入执行器。
- A/B 使用同一 session ID；C/D 的评审获得新的 session ID 和新的挂载目录，旧会话物理上不在容器中。
- 清空组仍可读取方案、实验事实、解释、代码和训练日志；这些是明确保留的研究记忆。
- 方案、修订、评审和方向选择按事件链追加；禁止把已观察实验的计划改写为“预先预测”。
- 每次训练调用先持久化扣款，再启动容器。失败、超时、归因和重复验证同样占次数。
- 中断恢复会停止遗留执行容器；有完整回执则恢复结果，否则记为无效并保留扣款，不偷偷重跑。
- 配对归因必须从被比较实验的精确源码版本出发；repeat 不改代码。训练随机设置沿用任务协议，不宣称提供多种子因果证据。
- 精确父版本和固定条件不能自动证明干预有效性；仍需审查实际代码差异是否只改变了预期因素。报告中的解释和事实始终分别保存。
- 归因候选在独立副本上执行；只有满足同一接受规则才更新 incumbent。
- 最终隐藏测试先持久化预约，只测试验证选出的源码和对应检查点。测试失败或中断不自动重试，不把测试反馈送回 agent。
- 如果恢复发现事件写入与状态提交不一致，保守停止并要求审计，不自动截断或覆盖历史。

隔离保证依赖 Docker 边界、可信镜像和正确的数据拆分。未经容器集成测试，不将该实现描述为
已通过真实环境安全审计。候选在一次被计费的训练进程内消耗多少计算仍由 task timeout/GPU
约束；“100 次调用”不是所有候选的 FLOPs 相同。

## 结果与验证

```sh
python3 -m unittest discover -s tests -v
python3 -m staged_research report --root ./runs/fml-study \
  --output /absolute/path/fml-study-report
```

报告写到 study 目录外，读取前验证事件链、源码和检查点。输出：

- `summary.json`：完整 60 单元，包括失败/未运行项；各任务原始指标、重复分布、B−A/C−B/D−C/D−A 差值及探索性 bootstrap 区间。
- `curves.csv`：每次实验的原始/最佳验证指标及归一化收益。
- `report.md`：可读比较表与证据限制。

统计包括验证 AUC、后半程收益、下一阶段实际收益、精确重复比例、无效比例、
归因调用比例及前后决策变化、训练耗时、整体耗时和原生 CLI token/cost 回执。
缓存读写 tokens 分开列；缺失 usage 不伪装成零。完整测试结果齐备前，不发布该方法的
全套宏平均。5 次重复的区间只是探索性描述，不自动作显著性或因果结论。

本轮四组只能拆解新增机制的逐步作用，不能完整估计清空×归因交互；
原始日志分析可能也会推测原因，D−C 检验的是**额外受控归因实验**的收益。

相关原始资料：[Karpathy](https://github.com/karpathy/autoresearch)、
[FML-bench](https://github.com/qrzou/FML-bench)、
[Fork-and-Flush](https://arxiv.org/html/2610.07447v1)、
[Arbor](https://arxiv.org/html/2606.11926v1)。
