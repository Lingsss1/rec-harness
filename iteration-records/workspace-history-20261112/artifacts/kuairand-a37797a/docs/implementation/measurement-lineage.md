# 工具晋升后的缺口复测血缘

新项目冻结 `measurement_lineage: true`。工具晋升后，后继模型的测量要求保留 `source_measurement_gap_ids`；同一要求可关联多个源缺口，已有要求的测量合同仍必须一致。旧项目没有该冻结字段时保留原配置生成方式，避免恢复时改变已冻结子研究输入。

实际工具输出包含所需测量时，主机核对已提交的诊断 Observation、模型、实现 hash 和完整输出，再原子提交 `measurement_output_observed` 事件与 `measurement_output_coverage` Observation。新观察通过 DERIVED_FROM 连接原始诊断，保留原缺口 ID；只有本地图中确有对应 MeasurementGap 时才添加 ABOUT 关系，跨研究引用不虚构节点。

重复处理同一诊断及要求不新增事件。输出被替换或缺少匹配的已提交诊断时拒绝；输出仍缺失则保留阻塞，并在新 MeasurementGap 的 `related_gap_ids` 中保留原缺口。旧观察与旧缺口不覆盖。

“观测到所需输出”不等于语义验收通过、缺口被解决或模型机制已证实。新增观察明确不是独立实验单位，不产生 Claim。

验证覆盖无预声明测量的自主项目路径：发现缺口→工具候选验证/晋升→后继模型复测，检查成功记录、来源值与版本、重复无写入、替换输出拒绝，以及错误输出继续阻塞并保留缺口血缘。使用已有脚本角色及 CPU 测量替身，属于工程集成测试，不是新增真实工具效用实验。本项晚于 v14 安装包。

完整源码回归 406 项通过（21 项环境相关跳过）；补充重复与替换输出断言后，项目定向 7 项全部通过。日志分别为 `/tmp/rec-harness-measurement-lineage-full-tests.log` 和 `/tmp/rec-harness-measurement-lineage-tests.log`。
