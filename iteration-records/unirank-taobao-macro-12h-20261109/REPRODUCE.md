# 复现入口

环境与版本见 research/environment*；UniRank 固定提交为 90bb99a60724e975ec565836266fd86dcad0039d。
数据 revision、文件校验和、固定切分、特征映射、历史口径见各 protocol 和 dataset-source-manifest。
权重及数据未上传，因此归档适合检查和同协议重训练，不能直接重放已训练模型。

淘宝模型：以 research/final-assessment-top3/suite/rank_1/model.py 和 model-config.json 为最终配置。
原训练器是 research/source/train.py，命令入口为 --campaign CAMPAIGN --model MODEL --config CONFIG --output OUTPUT --seed 20262027。
需要先按 research/config 和 source 中的数据准备程序重建目录，并把原绝对路径替换为本机路径。
选模使用 research/selection-policy.json；测试程序为 research/postprocess/assess_topk.py。

外部数据集：generalization/*/source/train.py 为实际执行代码。
先按 datasets/DATASET/dataset-source-manifest.json 获取完整固定版本，重建 data/、prepared/、parameters.json 中的本地路径。
需要重建协议中的 source_sha256 和 identity；不得改动评分、切分、任务顺序和训练配方。
运行：python source/train.py --root ROOT --dataset DATASET --kind candidate --output FRESH_OUTPUT。
训练器按验证集全任务平均 AUC 冻结 checkpoint 和选择凭据后再测试；不要通过测试成绩调参或选 checkpoint。
source/taobao-original.py 与 interface-adaptation.patch 记录原模型和字段/任务数适配。
历史 DIN 与上一版指标标记为复用；要重新对照应使用同一配方独立训练，不把历史耗时当作本轮新增成本。

每个复制文件的来源和 SHA256 见 copy-sources.json；manifest.json 覆盖整个归档的非 manifest 文件。

只核验归档：`python verify.py .`（在归档目录内执行，不训练、不访问原始数据）。
