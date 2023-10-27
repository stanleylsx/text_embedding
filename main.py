# -*- coding: utf-8 -*-
# @Time : 2023/10/27 22:05
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : main.py
# @Software: VSCode
from loguru import logger
from engines.data import DataPrecess
from config import use_cuda, cuda_device, mode, configure
import random
import numpy as np
import os
import torch
import json


def set_env(configures):
    random.seed(configures.seed)
    np.random.seed(configures.seed)


def fold_check(configures):
    datasets_fold = 'datasets_fold'
    assert hasattr(configures, datasets_fold), 'item datasets_fold not configured'

    if not os.path.exists(configures.datasets_fold):
        print('datasets fold not found')
        exit(1)

    checkpoints_dir = 'checkpoints_dir'
    if not os.path.exists(configures.checkpoints_dir) or not hasattr(configures, checkpoints_dir):
        print('checkpoints fold not found, creating...')
        paths = configures.checkpoints_dir.split('/')
        if len(paths) == 2 and os.path.exists(paths[0]) and not os.path.exists(configures.checkpoints_dir):
            os.mkdir(configures.checkpoints_dir)
        else:
            os.mkdir('checkpoints')

    log_dir = 'log_dir'
    if not os.path.exists(configures.log_dir):
        print('log fold not found, creating...')
        if hasattr(configures, log_dir):
            os.mkdir(configures.log_dir)
        else:
            os.mkdir(configures.datasets_fold + '/vocabs')


if __name__ == '__main__':
    log_name = './logs/' + mode + '.log'
    logger.add(log_name, encoding='utf-8')

    if use_cuda:
        if torch.cuda.is_available():
            if cuda_device == -1:
                device = torch.device('cuda')
            else:
                device = torch.device(f'cuda:{cuda_device}')
        else:
            raise ValueError(
                "'use_cuda' set to True when cuda is unavailable."
                " Make sure CUDA is available or set use_cuda=False."
            )
    else:
        device = 'cpu'
    logger.info(f'device: {device}')
    logger.info(json.dumps(configure, indent=2, ensure_ascii=False))
    data_manage = DataPrecess(logger)
    if mode == 'train':
        from engines.train import Train
        logger.info('stage: train')
        trainer = Train(data_manage, device, logger)
        trainer.train()
    elif mode == 'get_embedding':
        from engines.predict import Predictor
        logger.info('stage: get_embedding')
        predict = Predictor(data_manage, device, logger)
        predict.get_embedding('warm')
        while True:
            logger.info('please input a sentence (enter [exit] to exit.)')
            sentence = input()
            if sentence == 'exit':
                break
            logger.info('input:{}'.format(str(sentence)))
            result = predict.get_embedding(sentence)
            logger.info('output:{}'.format(str(result)))
            print(result)
    elif mode == 'predict_one':
        from engines.predict import Predictor
        logger.info('stage: predict_one')
        predict = Predictor(data_manage, device, logger)
        a = """
            怎么使用支付宝
        """
        b = """
            怎么使用微信
        """
        similarity, if_similar = predict.predict_one(a, b)
        print(similarity, if_similar)
    elif mode == 'convert_onnx':
        from engines.predict import Predictor
        logger.info('stage: convert_onnx')
        predict = Predictor(data_manage, device, logger)
        result = predict.convert_onnx()
