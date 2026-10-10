# Autoresearch 本地监控

监控面板沿用 RSI Console 的中文深色布局，直接读取 Amazon autoresearch 的 campaign 目录。它独立于训练控制器运行，不调用 Codex、不执行恢复、不写研究账本，也不提供启动、停止或修改协议的 HTTP 接口。

## 启动

在仓库根目录运行，`--run` 为包含 `research/`、`study/`、`progress.json` 的运行目录，可重复指定。列表中的第一个任务默认展示。

```bash
python3 -m rec_harness.monitor \
  --run /path/to/current-campaign \
  --run /path/to/previous-campaign \
  --port 4318
```

浏览器访问 `http://127.0.0.1:4318`。远程运行时可通过 SSH 转发端口：

```bash
ssh -L 4318:127.0.0.1:4318 user@training-host
```

然后打开客户端的 `http://127.0.0.1:4318`。默认仅绑定 loopback，并检查 Host；不会发布实验记录到外部网站。查看 GPU 遥测及进程状态需要在具有相应可见性的运行环境中启动服务；无权限时明确显示不可用，而不是显示空闲。

## 无需端口的服务器本地 HTML 面板

服务器无法使用 SSH 转发或 Web 端口时，可以持续生成一个独立 HTML 文件：

```bash
python3 -m rec_harness.monitor \
  --run /path/to/current-campaign \
  --run /path/to/previous-campaign \
  --html /path/to/monitor.html --interval 10
```

在服务器文件管理器中双击 `monitor.html`，或在服务器浏览器中打开该文件。HTML 内置样式、脚本和聚合快照，不请求 HTTP API，不需要登录服务或开放端口。后台每 10 秒原子更新文件，页面每 10 秒重新载入；暂停刷新仅影响页面。切换任务与所选轮次在当前浏览器会话中保留。

下载到其他电脑的副本只包含下载时的快照；只有服务器上的原文件持续更新。文件顶部和底部显示快照时间。省略 `--interval` 仅导出一次。

## 无桌面环境的终端实时面板

```bash
python3 -m rec_harness.monitor --run /path/to/current-campaign --terminal --interval 10
```

在服务器终端每 10 秒刷新进度、模型、调用与 token 用量、GPU 占用、训练 epoch 和最近迭代。`Ctrl+C` 只退出监控，不停止训练。省略 `--interval` 输出一次快照；终端模式不启动 HTTP 服务。

## 展示与统计口径

- 运行概览：控制器 PID 与启动身份校验、实际阶段、计划轮数、完成实验数、失败数和晋升次数。完成轮数取已提交的 `auto_round_completed` 事件，失败轮不算完成实验。已停止进程不能仅凭旧 `progress.json` 显示为运行中。
- 迭代结果：开发集各模型 HR@10、提案、审查意见、返回修改次数和每次角色调用。候选、机制对照和每轮基线分别显示；没有结果的地方显示空值。
- 训练任务：模型、类别、GPU、当前 epoch、最佳 epoch、最终 HR@10 与 GPU 秒。训练中从宿主保存的 stdout 读取已刷新聚合曲线，不进入 jail。候选 epoch 指标不冒充完成回执中的指标。
- GPU：`nvidia-smi` 提供整卡利用率和显存，包含其他任务的占用。高亮表示配置的卡号，不代表当前任务独占该卡。
- Token：以 `resources.json` 的用量回执为准，包括顶层角色、格式拒绝重试和归档的 reviewer 返回修改。运行中尚未写回执时，读取末尾 `turn.completed` 用量。输入已包含缓存输入；思考已包含输出，累计值仅为输入加输出。没有回执不推算实际消耗，不将 token 数换算成 Pro 额度。
- 成本：当前与继承的 GPU 计费/预留分开展示，GPU 秒来自研究账本，与整卡利用率不是同一指标。
- 最近事件：显示两个账本的最后 60 条事件，仅投影选定字段。研究数据库以 SQLite `mode=ro` 单次读事务访问，不通过可写 Store 初始化或修复文件。

前端每 10 秒刷新，支持切换 campaign、查看某轮详情、训练状态筛选和暂停页面刷新。服务端缓存 5 秒；页面刷新不会触发任何模型调用。页面暂停刷新不会暂停训练。

HTTP 仅开放页面静态资源、`GET /api/runs` 和 `GET /api/snapshot?run=已配置的任务名`。不提供任意文件、原始上下文、凭据、checkpoint 或用户级结果的下载接口。常见密钥模式在展示文本中脱敏。

## 验证

```bash
python3 -m unittest discover -s tests -p test_monitor.py -v
node --check rec_harness/monitor_assets/app.js
```

测试覆盖只读账本、失败/完成分离、格式重试与修订用量、继承成本、部分 JSON 写入、工作区同名文件排除、停止的控制器和运行中 epoch。实际浏览器验收覆盖当前任务与上一 campaign 的真实数据、桌面/手机布局、轮次详情、筛选、暂停刷新、Host 校验及路径限制。
