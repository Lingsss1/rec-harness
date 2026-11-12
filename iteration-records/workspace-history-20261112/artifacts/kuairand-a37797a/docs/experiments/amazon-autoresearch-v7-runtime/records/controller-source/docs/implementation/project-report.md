# 项目调用与费用报表

```bash
python3 -m rec_harness project-report /path/to/project --format markdown
python3 -m rec_harness project-report /path/to/project --format json
```

报表读取项目冻结配置和已提交状态，逐个检查项目账本及 episodes 下的嵌套验证账本。不创建 Agent、不执行训练、不恢复或结算运行；未提交到项目终态但已发生的子任务也列入清单。每行展示所属 episode、账本路径、研究角色提交、候选生成提交、前缀完成记录、费用及待结算数。JSON 保留各账本事件头与每次提交的事件身份，便于定位调用来源。

前缀完成记录单列，不伪装成提交事件或完整尝试数。调用频率使用框架提交事件时间，不能推导服务内部 API 请求、并发或计费。费用保留各账本原单位，不把父级验证费用和子级执行费用当作同一项，也不将外部 Amazon study/GPU 费用混入总数。多个数据库各自读取一致快照；正在运行时不保证跨数据库同一时刻，项目状态亦不代表进程是否存活。

`tests/test_project_report.py` 覆盖运行中保留费用、未提交子任务、旧 state 投影、外部账本拒绝、预算投影篡改以及 Markdown CLI。真实只读核验目录 `rec-harness-runs/project-report-acceptance-v1` 对比此前独立终态审计：

| 项目 | 提交事件 | 前缀完成记录 | 待结算 |
|---|---:|---:|---:|
| joint-tool-project-v1 | 20 | 0 | 0 |
| joint-project-acceptance-v1 | 20 | 8 | 0 |

两次读取完全一致；各本地账本原值之和分别与原审计的 51、42 BCU 一致。这两个具体项目的费用单位已由原审计确定，不据此默认所有项目都可合计。核验未新增角色或训练调用。
