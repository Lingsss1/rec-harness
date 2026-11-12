# v33 安装交付验收

冻结提交 `f4e368b3f033823b970092b3fbc0fbab5eb848e8`，包含候选评审的基线测量身份修复。wheel SHA256 为 `42e094aa36a7e8ec49995efb3a69e88f3bb0164eec5312c5bb0d5c86ac322b5a`。

144个Python模块及原生C源码在冻结源码、wheel和安装目录之间逐字节一致，20个命令入口通过。独立安装环境完整运行645项测试，零跳过，171.394秒；138个已加载项目模块均来自安装包。161份测试/示例文件与冻结提交核对一致。源码环境另完成645项测试，零跳过，176.690秒。

验收目录为 `rec-harness-runs/package-install-acceptance-v33`，`delivery-acceptance.json`保存源码、wheel、测试日志和检查脚本哈希。测试使用独立Python3.12环境，显式复用第三方Torch依赖，启用操作系统隔离测试；包/入口检查使用独立Python3.14环境。unittest完整结果保存在installed-tests.log；普通标准输出保留于工具会话，没有另造full-stdout.log。

本次交付验证没有真实Codex调用或Amazon训练。v4原始测量送达审计失败仍保留；修复后的真实评审输入验证在独立 `amazon-review-scope-v1` 目录运行，不替代原实验，也不建立新的工具效用结论。
