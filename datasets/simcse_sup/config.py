# -*- coding: utf-8 -*-
# @Time : 2023/10/27 22:05
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : config.py
# @Software: VSCode

# 模式
# train:                      训练分类器
# get_embedding:              获取句向量
# predict_one:                预测一句模式
# convert_onnx:               将torch模型保存onnx文件
# mteb:                       跑mteb进行测试
# test:                       目前只支持跑相关性测试
mode = 'train'

# 使用GPU设备
use_cuda = True
cuda_device = 2

configure = {
    # 训练方式
    # 支持的训练方式有cosent、simcse_sup、simcse_unsup
    'train_type': 'simcse_sup',
    # 模型类别，支持Bert、XLMRoberta、GTE
    'model_type': 'GTE',
    # 获取Embedding的方法，支持cls、last-avg、pooler
    'emb_type': 'last-avg',
    # 训练数据集
    'train_file': 'datasets/simcse_sup/cmnli_sup_train_data.csv',
    # 验证数据集，必须是pairdata
    'val_file': '',
    # 测试数据集
    'test_file': '',
    # 模型保存的文件夹
    'checkpoints_dir': 'checkpoints/piccolo/first_version',
    # 模型的名字
    'model_name': 'debug.bin',
    # 预训练模型细分类（直接填huggingface上的模型tag）
    'hf_tag': 'Alibaba-NLP/gte-multilingual-base',
    # 使用fp16混合精度训练
    'use_fp16': False,
    # 句子的token的最大长度，请注意在你不需要扩展长度的时候需要和你用到的模型的config.json文件里面的max_position_embeddings保持一致
    'max_position_embeddings': 1024,
    # 训练迭代的次数
    'epochs': 4,
    # bs设置
    'batch_size': 4,
    # 学习率
    'learning_rate': 4e-5,
    # 梯度累计
    'gradient_accumulation_steps': 4,
    # warmup的步数在所有步数中的前占比
    'warmup_ratio': 0.05,
    # 计算指标的时候的选项
    'metrics_average': 'micro',
    # 微调阶段的patient
    'patience': 8,
    # 训练阶段每print_per_batch打印
    'print_per_batch': 100,
    # 训练是否提前结束微调
    'is_early_stop': True,
    # Cosent方法中的超参数lambda
    'cosent_ratio': 20,
    # SimCSE的超参数tao
    'simcse_tao': 0.05,
    # 判决相似和不相似的阈值
    'decision_threshold': 0.78,
    # 使用层次位置编码扩展相对位置编码的长度
    'hierarchical_position': False,
    # 使用层次位置编码扩展的默认超参数
    'hierarchical_alpha': 0.4,
    # 使用EWC
    'use_ewc': False,
    # EWC损失的超参数lambda
    'ewc_ratio': 10,
    # 使用mteb评测的时候的能力
    # retrieval、reranking、pairclassification、clustering、sts
    'task_class': 'retrieval'
}
