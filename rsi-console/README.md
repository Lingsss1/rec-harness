# RSI Console

本机中文深色双轨研究控制台：React / TypeScript / Vite + Node.js + SQLite，通过 Codex App Server 独立会话驱动 SASRec / Amazon CDs 的受控实验。新项目不修改 `rec-harness` 或另一文档站。

## 启动

首次使用先运行 `bash scripts/setup.sh` 安装依赖并构建，然后：

```bash
cd rsi-console
bash scripts/start.sh
```

打开 **http://127.0.0.1:4317**。端口可用 `PORT` 指定；`RSI_DATA` 指定持久化目录，`RSI_PYTHON` / `RSI_CODEX` 可覆盖本机可执行文件。服务必须能写本地 Codex 状态目录并访问已登录服务。浏览器不接收登录凭据。

新环境需要 Node ≥22.13、Python、Git 和已登录的 Codex CLI。运行 `bash scripts/setup.sh` 下载固定提交及依赖。安装脚本不会启动上游 `self_evolverec.py`。CUDA 需自行配置匹配的 PyTorch 和驱动；CPU 是本次验收的真实运行设备。

开发前端：`npm run dev`（5173），后端仍用 `npm start`（4317）。生产使用 `npm run build && npm start`。

## 实现与协议

- 五个角色：研究负责人、模型工程师、诊断研究员、独立评估员、RSI 改进员。每次工作使用独立 App Server thread/turn，保存身份和结构化回执；固定回放由两次额外评估员调用完成。首版串行执行，满足最多两个 Agent、最多一个训练进程的限制。
- App Server 使用 `initialize → thread/start → turn/start`，读取完成事件后才提交回执。Skill 通过协议的 `skill` input 显式调用；新任务还在 prompt 内附正文，避免模型未读取 Skill。原始项目 Skill 保留，新任务生成自己的完整副本。
- 冻结数据文件哈希、上游提交、种子、用户采样、训练 epoch、指标与保留条件。新任务还保存评估器快照及哈希，校验上游数据处理源文件。验证候选列表直接复用上游；不重新采样负候选。
- 训练直接调用上游 `Model` 和 `SeqDataset`，独立适配器去掉上游 LLM 模拟器与进化调度。固定用户子集、32 维、50 序列长度，每次从同种子初始化。指标是原始逐用户 rank 汇总的 NDCG@5 / HR@5；页面曲线取自评估文件，缺失值显示“未产生”。
- 模型保留要求验证 NDCG 严格提升且 HR 不下降。测试集仅在全部候选轮结束后评估最终保留模型，不用于决策。
- 模型工程师的代码修改首版限定为 ReLU→GELU、1 层或 3 层这三个可审计变换，由宿主应用、保存补丁并训练。它不是任意代码执行平台；Agent 会话只读，不能改调度器或评估器。其他研究假设需扩展受控变换后才能执行。
- `diagnose_collapse.py` 在真实验证表征上执行。外层调用 `coevolve.py propose` 保存结构变化触发；存在已验收的 ready handoff 时先 `check`，评估后 `close`。完整的 `propose / handoff / check / close` 和六级门控脚本保留在本项目 Skill 副本中。
- RSI 策略候选采用相同案例、相同单 turn 上限的原策略/候选回放。固定回放没有证明收益时保持候选；真实任务上的独立同预算效用证据目前尚未自动生成，所以即使回放改善也不自动晋升。诊断工具六级证据未完成时显示 pending，工具通过不等于模型收益。
- 已接受版本的下轮启用与父版本回退由调度器执行并记录事件；当前验收没有接受的 RSI 版本。本轮 Agent 提出的逐用户权衡诊断已通过受控模板生成独立 Python 工具及 Skill 正文草案，验证合成正负控制、异常输入和两个真实模型条件；因文献审核与独立真实任务效用证据不足保持候选。任意新探针代码执行与自动生成完整真实任务对照仍未开放。

## 状态与可靠性

`created → running → completed`；支持 `pausing → paused → resume`、`stopped`、`blocked`、`interrupted` 和 `budget_exhausted`。暂停等待当前完整候选轮结束（基线单独作为边界），停止终止所属进程组并保留产物。运行时间包含 Agent 和审批等待，暂停时间不计入；调用预算按 `turn/start` 计数。

SQLite WAL 保存任务和递增事件序号，SSE 接受 `Last-Event-ID` 或 `?after=`。重复 start 不重复训练；已有且身份匹配的完整评估文件可复用。重启按 PID 和 Linux `/proc` 启动时间核对所属进程，再终止并置为 interrupted，人工点击继续恢复。进程身份无法匹配时不误杀。单实例锁防止两套后端同时调度同一存储目录。

仅监听 loopback，并校验 Host、Origin 和 JSON content type。文件接口仅提供已登记、真实路径在实验目录内的文件；禁止 symlink 越界，大检查点留在本机目录。审批请求在页面呈现；冻结实验禁止 Agent 提权，当前可拒绝或取消，不能放开调度器保护。日志做常见密钥脱敏，不记录 App Server 账户响应和环境变量。

## API

| 接口 | 用途 |
|---|---|
| `GET /api/health` / `GET /api/preflight?device=cpu` | 服务与数据/登录/依赖检查 |
| `GET, POST /api/runs` | 列表与创建（name/mode/device/minutes/callBudget/rounds/epochs/trainUsers/evalUsers/seed） |
| `GET /api/runs/:id` | 任务完整快照 |
| `POST /api/runs/:id/control` | `{action: start/resume/pause/stop/rollback}` |
| `GET /api/runs/:id/agents` / `models` / `rsi` / `metrics` | 关联对象 |
| `GET /api/runs/:id/events?after=序号` / `stream?after=序号` | 历史与可恢复 SSE |
| `GET /api/runs/:id/artifacts?path=已登记相对路径` | 产物内容；省略 path 获取清单 |
| `POST /api/runs/:id/approvals` | `{requestId, decision: decline/cancel}` |

数据位于 `var/console.sqlite` 和 `var/<run-id>/`。实验目录包括 protocol、models、round-N、final、memory、harness 与 skills。事件关联 run/round/agent/model/rsi/protocol/evidence；日志事件无 Agent 时字段为 null。历史回放滑块回放事件流，页面明确标注上方指标卡展示最终状态，不伪装为历史快照。

## 验证

```bash
export PATH="$PWD/.runtime/bin:$PATH"
npm run build
node tests/core.test.mjs
node tests/codex.test.mjs
node tests/runtime.test.mjs
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

浏览器验收从工作区父目录运行：

```bash
PLAYWRIGHT_BROWSERS_PATH="$PWD/rsi-console/.runtime/browsers" \
  rsi-console/.runtime/bin/node rsi-console/scripts/browser-check.mjs
```

真实训练结果、回放证据及验收范围见 [ACCEPTANCE.md](ACCEPTANCE.md)。演示模式只产生明确标注的固定示例数据，不调用 Codex 或训练。

## 来源

- [Codex App Server 官方协议](https://learn.chatgpt.com/docs/app-server)，本机核对 codex-cli 0.153.4 生成的 JSON Schema。
- [Self-EvolveRec](https://github.com/Sein-Kim/self_evolverec)，固定 `edeff0126f237483e2d30651b5179eaea0b0ad22`。模型、数据和 util 在本机 vendor 中保持来源；不要将本项目视作授予上游代码或数据的新许可证。
- `rec-harness` 的本机版本 `b2580b036d21569ce24c6356af1efc5f58110dfa`，研究 Skill 使用独立副本；来源工作树包含既有未提交的闭环扩展，完整文件指纹保存在 `skills.snapshot.json`，不能仅凭 Git HEAD 重建该副本。
