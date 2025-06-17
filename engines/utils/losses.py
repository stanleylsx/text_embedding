# -*- coding: utf-8 -*-
# @Time : 2023/10/27 22:05
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : losses.py
# @Software: VSCode
from config import configure
import torch


def cosent_loss(output, y_true, device):
    """
    :param output: 模型的输出对于文本对队里a和队里b的cos值，[a,b,c]--[a1,b1,c1]-->[cos<a,a1>,cos<b,b1>,cos<c,c1>]
    :param y_true: 文本[a,b,c]--[a1,b1,c1]对应的有监督标签<a,a1>/<b,b1>/<c,c1>
    :return: loss
    """
    output = output * configure['cosent_ratio']
    # 4. 取出负例-正例的差值
    # 利用了矩阵计算的广播机制
    y_pred = output[:, None] - output[None, :]  # 这里是算出所有位置 两两之间余弦的差值
    # 矩阵中的第i行j列  表示的是第i个余弦值-第j个余弦值
    y_true = y_true[:, None] < y_true[None, :]  # 取出负例-正例的差值
    y_true = y_true.float()
    y_pred = y_pred - (1 - y_true) * 1e12  # 这里之所以要这么减，是因为公式中所有的正样本对的余弦值减去负样本对的余弦值才计算损失，把不是这些部分通过exp(-inf)忽略掉
    y_pred = y_pred.view(-1)
    y_pred = torch.cat((torch.tensor([0]).float().to(device), y_pred), dim=0)  # 这里加0是因为e^0 = 1相当于在log中加了1
    return torch.logsumexp(y_pred, dim=0)


def simcse_sup_loss(y_pred):
    """
    有监督SimCSE损失函数
    """
    device = y_pred.device
    temperature = configure['simcse_tao']
    assert y_pred.shape[0] % 3 == 0, "Batch size must be divisible by 3 (anchor, positive, negative)"
    batch_size = y_pred.shape[0] // 3
    anchors = y_pred[0::3]
    positives = y_pred[1::3]
    negatives = y_pred[2::3]
    # 拼接 positives 和 negatives 作为候选
    candidates = torch.cat([positives, negatives], dim=0)  # [2*batch_size, hidden_size]
    # 计算余弦相似度: anchors × candidates.T
    similarities = torch.matmul(anchors, candidates.T)  # [batch_size, 2*batch_size]
    # 相似度缩放
    similarities = similarities / temperature
    # 构造标签，正样本在 candidates 中排在前 batch_size 个位置
    labels = torch.arange(batch_size, device=device)
    loss = torch.nn.functional.cross_entropy(similarities, labels)
    return loss


def simcse_unsup_loss(features):
    """
    无监督SimCSE损失函数
    """
    device = features.device
    temperature = configure['simcse_tao']
    y_true = torch.arange(0, features.shape[0], device=device)
    y_true = (y_true - y_true % 2 * 2) + 1
    similarities = torch.matmul(features, features.T)  # cosine sim
    similarities = similarities - torch.eye(features.shape[0], device=device) * 1e12
    similarities = similarities / temperature
    loss = torch.nn.functional.cross_entropy(similarities, y_true)
    return loss


def get_mean_params(model):
    """
    :param model:
    :return:Dict[para_name, para_weight]
    """
    result = {}
    for param_name, param in model.named_parameters():
        result[param_name] = param.data.clone()
    return result


def ewc_loss(model, original_weight):
    losses = []
    ewc_ratio = configure['ewc_ratio']
    for n, p in model.named_parameters():
        # 每个参数都有mean和fisher
        if p.requires_grad:
            mean = original_weight[n.replace('module.', '')]
            if 'position_embeddings.weight' in n:
                losses.append(((p - mean)[:512, :] ** 2).sum())
            else:
                losses.append(((p - mean) ** 2).sum())
    ewc_loss = sum(losses) * ewc_ratio
    return ewc_loss
