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
mode = 'train'

# 使用GPU设备
use_cuda = True
cuda_device = 2

configure = {
    # 训练方式
    # 支持的训练方式有cosent、simcse_sup、simcse_unsup
    'train_type': 'cosent',
    # 模型类别
    # 支持的有e5、bge、piccolo、simbert、simbert_v2、m3e
    'model_type': 'piccolo',
    # 训练数据集
    'train_file': 'datasets/customer_service/customer_service.csv',
    # 验证数据集，必须是pairdata
    'val_file': '',
    # 测试数据集
    'test_file': '',
    # 模型保存的文件夹
    'checkpoints_dir': 'checkpoints',
    # 模型的名字
    'model_name': 'debug.bin',
    # 预训练模型细分类
    'hf_tag': 'infgrad/stella-base-zh-v2',
    # 句子的token的最大长度
    'max_sequence_length': 1024,
    # 训练迭代的次数
    'epochs': 4,
    # bs设置
    'batch_size': 8,
    # 学习率
    'learning_rate': 4e-5,
    # 梯度累计
    'gradient_accumulation_steps': 2,
    # warmup的步数在所有步数中的前占比
    'warmup_ratio': 0.05,
    # 计算指标的时候的选项
    'metrics_average': 'micro',
    # 微调阶段的patient
    'patience': 8,
    # 训练阶段每print_per_batch打印
    'print_per_batch': 500,
    # 训练是否提前结束微调
    'is_early_stop': True,
    # Cosent方法中的超参数lambda
    'cosent_ratio': 20,
    # SimCSE的超参数tao
    'simcse_tao': 0.05,
    # 判决相似和不相似的阈值
    'decision_threshold': 0.78,
    # 使用EWC
    'use_ewc': False,
    # EWC损失的超参数lambda
    'ewc_ratio': 10,
}
