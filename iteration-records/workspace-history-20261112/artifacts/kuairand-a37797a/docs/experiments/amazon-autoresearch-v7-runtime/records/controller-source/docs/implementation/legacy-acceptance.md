# 旧记录分类型迁移验收

实现入口为 `rec_harness/legacy.py`，依据仓库 `model-optimization-harness-updater/references/evidence-schema.md`。迁移的是已存在的源记录，不重新解释为已证实的新实验。

`python3 -m unittest discover -s tests -p test_legacy.py -v` 的五项测试覆盖：

- results/episode同一experiment_id只生成一个Experiment标识；两份不同指标摘要均保留为Observation。
- failures保留签名、复现命令、避免条件及执行证据属性；lessons封装内的支持/反证关系保留；strategy只作工作状态。
- 原confirmed及valid标签不会升级迁移后证据；即使两个旧Experiment被新Claim引用，仍不能通过独立有效证据门槛。
- 重复迁移不新增节点、边或事件；源文件修改后仍可从CAS恢复旧文件的精确字节。
- 源文件在解析后被另一个写入者替换时，归档仍保存实际解析的同一版本字节，引用hash不会错配。
- 先到的lesson不制造缺失实验；后到真实记录补齐引用，旧图快照仍可读取。
- 显式Hidden字段拒绝；相同记录迁移到不同任务不会发生ID冲突。

这些是schema与数据完整性测试，不证明旧结果本身正确。缺失seed、独立运行身份或验证资格保留为未知；原文所报值另存reported字段。未自动摄取任意外部artifact_path，也未声称支持未定义的旧血缘图格式。
