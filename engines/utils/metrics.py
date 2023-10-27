# -*- coding: utf-8 -*-
# @Time : 2023/10/27 22:05
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : metrics.py
# @Software: VSCode
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from config import configure
import scipy


def compute_corrcoef(x, y):
    """
    Spearman和Pearsonr相关系数
    """
    spearmanr = scipy.stats.spearmanr(x, y).correlation
    pearsonr = scipy.stats.pearsonr(x, y)[0]
    return {'spearmanr': spearmanr, 'pearsonr': pearsonr}


def cal_metrics(predict, targets):
    """
    指标计算
    """
    average = configure['metrics_average']
    precision, recall, f1, _ = precision_recall_fscore_support(targets, predict, average=average, zero_division=0)
    acc = accuracy_score(targets, predict)
    return {'precision': precision, 'recall': recall, 'f1': f1, 'acc': acc}
