UniRank：测试点击AUC最佳模型的多数据集结构泛化

源模型：g1-r0088-membership_full_first_mean_only
源淘宝测试点击AUC：0.6664593526801014
来源：旧48h搜索的预先冻结完整候选；它的已测点击AUC略高于当时按验证集选出的替换对照。
本轮由人明确要求按已观察到的测试点击AUC选源模型。因此这是测试观察后的探索性迁移，不是一次全新的未见测试确认。

执行：只新增4组候选训练（QK-Video、KuaiRand、TAAC-25、MerRec），GPU4..7；GPU0..3的12h研究继续运行。
复用前一批同数据、同划分、同字段、同seed和训练配方的DIN及g3-r0029-context_shared_primary结果，不伪装成重新训练。
每组保留原模型的自定义梯度、首层Dice均值导数替换、历史压缩、任务成员信息及joint真标签训练损失。
接口适配：任务名和数量；主任务为原生元数据的首任务；辅助梯度在原生任务顺序下作左关联FP64求和；三个匹配槽为item_id和两个元数据字段，缺失槽为零。
原淘宝CPU/BF16 GPU上的初始化、预测、损失、完整梯度、运行状态和eval输出精确等价；2/4/5/6任务及缺失元数据测试均通过。
实际原淘宝checkpoint另做完整验证集重放核查，不重训、不读取测试。

协议：固定原生train/valid/test分块，1epoch、batch8192、最近100历史、seed20262027、embedding16、AdamW1e-4 + Adagrad0.05、BF16。
验证全任务平均AUC选checkpoint，随后一次完整测试；报告点击/首任务AUC、全任务平均AUC、全部任务Logloss和成本。
MerRec没有点击标签，其首任务Like不能称作点击AUC。
测试集之前已用于另一架构评估；本轮不据目标测试结果调参、重选或继续搜索。

完整数据已复用并重新逐文件SHA256核验，无需重新下载。
每组先运行20步成本标定，使用相同seed重新初始化后进行完整训练；training_loss_hook_calls逐步记录，必须与优化步数一致。
每次完整运行上限48h；校准上限30min。失败和成本均保留；单seed只形成初步证据。

主要产物：
  source-selection.json：源模型选择依据及原始测试分数
  interface-adaptation.patch：相对原模型的接口适配差异
  interface-checks.* / cached-release-checks.json / taobao-validation-replay/：等价性和数据检查
  datasets/*/protocol.json / candidate-config.json / reference-reuse.json：协议、配置、旧对照来源
  runs/*/candidate/{calibration-01,fit-01}/：完整日志、结果、checkpoint、验证选择凭据
  reference-runs/：旧DIN及旧多任务最佳模型的原始结果，不改写其原协议哈希
  report.json / report.txt：三模型的首任务及全任务测试对照，含所有任务变化

复现某个候选（请换成新输出目录以保留已有结果）：
  CUDA_VISIBLE_DEVICES=4 artifacts/unirank-env/bin/python artifacts/unirank-click-best-generalization-latest/source/train.py --root artifacts/unirank-click-best-generalization-latest --dataset QK-Video --kind candidate --output /tmp/unirank-click-best-qk-repeat

观察：
  tmux -L ctr-research attach -t ctr-research
  进入“点击最佳泛化”窗口。

停止本轮固定迁移测试（不停止另一个12h研究）：
  touch artifacts/unirank-click-best-generalization-latest/STOP
