# -*- coding: utf-8 -*-
# @Time : 2023/10/27 22:05
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : main.py
# @Software: VSCode
from loguru import logger
from engines.data import DataPrecess
from config import use_cuda, cuda_device, mode, configure
from engines.train import Train
from engines.predict import Predictor
import random
import numpy as np
import os
import torch
import json


def set_env(configure):
    random.seed(configure.seed)
    np.random.seed(configure.seed)


def fold_check(configure):

    if configure['checkpoints_dir'] == '':
        raise Exception('checkpoints_dir did not set...')

    if not os.path.exists(configure['checkpoints_dir']):
        print('checkpoints fold not found, creating...')
        os.makedirs(configure['checkpoints_dir'])


if __name__ == '__main__':
    log_name = './logs/' + mode + '.log'
    logger.add(log_name, encoding='utf-8')
    fold_check(configure)
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
        logger.info('stage: train')
        trainer = Train(data_manage, device, logger)
        trainer.train()
    elif mode == 'get_embedding':
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
    elif mode == 'predict_one':
        logger.info('stage: predict_one')
        predict = Predictor(data_manage, device, logger)
        a = """冬天好冷"""
        b = """这个冬天真的好冷啊，你准备在哪里过年"""
        similarity, if_similar = predict.predict_one(a, b)
        text = '\nsentence A:{}\nsentence B:{}\nsimilarity:{}\nif_similar:{}'.format(a, b, similarity, if_similar)
        logger.info(text)
    elif mode == 'convert_onnx':
        logger.info('stage: convert_onnx')
        predict = Predictor(data_manage, device, logger)
        result = predict.convert_onnx()
    elif mode == 'mteb':
        logger.info('stage: mteb')
        predict = Predictor(data_manage, device, logger)
        result = predict.mteb()
    elif mode == 'test':
        logger.info('stage: test')
        trainer = Train(data_manage, device, logger)
        predict = Predictor(data_manage, device, logger)
        predict.test(trainer)
    elif mode == 'batch_test':
        # 批量转embedding
        logger.info('stage: batch_test')
        predict = Predictor(data_manage, device, logger)
        predict.batch_embedding()
