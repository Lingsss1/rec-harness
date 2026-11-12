# 上游来源、许可与适配说明

- [FML-bench](https://github.com/qrzou/FML-bench/tree/d336651ebea50c622c256f02ded82b68b4451fdc)，固定 commit `d336651ebea50c622c256f02ded82b68b4451fdc`。
  三任务的算法接口、默认配置、数据划分和评分约定来自该版本。
  使用 Apache-2.0 许可，随附 [FML-LICENSE.txt](staged_research/resources/FML-LICENSE.txt)。
- [DomainBed](https://github.com/facebookresearch/DomainBed/tree/b93c22a1cfc3b2428398272c1a116c8de1f4139e)，固定 commit `b93c22a1cfc3b2428398272c1a116c8de1f4139e`。
  Copyright (c) Facebook, Inc. and its affiliates. MIT 许可，随附
  [DOMAINBED-LICENSE.txt](staged_research/resources/DOMAINBED-LICENSE.txt)。
  ColoredMNIST 构造、哈希种子和训练接口按上游实现适配，保留其算法源文件及许可。
- [Opacus 1.5.4](https://github.com/pytorch/opacus/tree/v1.5.4)，Apache-2.0。
  通过包管理器安装，不随本项目复制库源码。隐私会计使用该版本 PRV 实现与 BatchMemoryManager。
- [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html)，安装器版本 2.3.3，
  Linux x86_64 归档 SHA256：`e7274528ceb9c20d048a428d6c22d7e02e268f8ffb762c4c365422347c8b8ba2`。
  仅在服务器准备时下载到指定 prefix。

本项目新增的评估器将训练、预测和外部评分分离，物化固定划分；原生最终测试只加载已选
模型。与直接执行 FML 原脚本的主要差异、随机性及验证边界详见 [FML_SETUP.md](FML_SETUP.md)。
本项目不将这些适配后的基线指标表述为上游原环境的逐位复现。

数据从 torchvision 声明的原始来源获取，适用各数据集自身的许可/使用条件。
MNIST、CIFAR 与 SVHN 数据不随本仓库发布。
