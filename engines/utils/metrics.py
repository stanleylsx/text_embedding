# -*- coding: utf-8 -*-
# @Time : 2023/10/27 22:05
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : metrics.py
# @Software: VSCode
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from config import configure
import scipy
import torch


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


class MyModel():
    def __init__(self, data_manage, model, device):
        self.model = model
        self.data_manage = data_manage
        self.device = device

    @torch.inference_mode()
    def encode(self, sentences, batch_size, **kwargs):
        vectors = []
        sentences = [sentences[i: i + batch_size] for i in range(0, len(sentences), batch_size)]
        for sentence in sentences:
            input_ids = self.data_manage.batch_tokenize(sentence).to(self.device)
            vector = self.model(input_ids)
            vector = vector.detach().cpu().tolist()
            vectors.extend(vector)
        return vectors
