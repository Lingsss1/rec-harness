# 已封存的 NextItNet v3 候选

这里交付已经完成开发与独立初始化确认的原始源码，不重新训练或修改候选：

- `baseline.py`：该研究的 `legacy_incumbent` 对照。
- `candidate.py`：主机选中的 `amazon-next-v3-round-1-model`。
- `candidate.patch`：相对对照的唯一改动，在 `NextItNetBlock.forward` 第二次卷积后、dropout 前应用已有 GELU。
- `manifest.json`：精确源码哈希、研究身份、类别及冻结协议摘要。

源码保持原始字节，包括原有文档字符串和训练辅助函数。候选哈希为 `ae556df7a0c0808bd7c2dc5dc912eb5546a0abe297cc73683a395bddb8240635`，基线哈希为 `d32d162182c2c156dd24573819361d513e49960f44429ce6f005afa21ba4d553`。可在本目录用 `sha256sum baseline.py candidate.py` 核对。

四类别开发平均 HR@10 从 0.553500 升至 0.566250；相同验证用户、独立初始化确认从 0.549625 升至 0.562750。开发集 CDs_and_Vinyl 的 NDCG@10 下降，因此不声称所有指标改善。这不是最终测试或独立的机制因果证明。完整范围和原始收据位置见[研究记录](../../../docs/experiments/amazon-next-autoresearch-v3.md)。

将这些文件作为 Amazon study 的模型源码使用。实际研究由冻结 worker 控制训练和评分，源码中的辅助函数默认值不等于完整执行配置；只运行 `candidate.py` 不能声称复现上述指标。本目录不包含数据、checkpoint、运行环境或最终测试产物，也不覆盖当前 RecHarness 的默认模型。
