# 实验血缘图

打开 [交互血缘图](index.html)。完整机器可读节点与边见 [experiments.json](experiments.json)。

每次实验启动及结束调用 `python3 qformer-loop/lineage/update.py`。失败也更新，不等整轮收尾。历史 parent 字段保持原意；当前研究单独记录结构父节点与指标对照边。
