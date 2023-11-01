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


def simcse_sup_loss(y_pred, device):
    """
    有监督simcse损失函数
    """
    # [12, 768]
    simcse_tao = configure['simcse_tao']
    # [12, 12]
    similarities = torch.cosine_similarity(y_pred.unsqueeze(0), y_pred.unsqueeze(1), dim=2)
    #  0,  3,  6,  9
    row = torch.arange(0, y_pred.shape[0], 3)
    #  0 ~ 11
    col = torch.arange(0, y_pred.shape[0])
    #  1,  2,  4,  5,  7,  8, 10, 11
    col = col[col % 3 != 0]
    # [4, 12]
    similarities = similarities[row, :]
    # [4, 8]
    similarities = similarities[:, col]
    # [4, 8]
    similarities = similarities / simcse_tao
    #  0,  2,  4,  6
    y_true = torch.arange(0, len(col), 2, device=device)
    loss = torch.nn.functional.cross_entropy(similarities, y_true)
    return loss


def simcse_unsup_loss(y_pred, device):
    """无监督的损失函数
    y_pred (tensor): bert的输出, [batch_size * 2, 768]

    """
    simcse_tao = configure['simcse_tao']
    # 得到y_pred对应的label, [1, 0, 3, 2, ..., batch_size-1, batch_size-2]
    y_true = torch.arange(y_pred.shape[0], device=device)
    y_true = (y_true - y_true % 2 * 2) + 1
    # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
    sim = torch.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    # 将相似度矩阵对角线置为很小的值, 消除自身的影响
    sim = sim - torch.eye(y_pred.shape[0], device=device) * 1e12
    # 相似度矩阵除以温度系数
    sim = sim / simcse_tao
    # 计算相似度矩阵与y_true的交叉熵损失
    # 计算交叉熵，每个case都会计算与其他case的相似度得分，得到一个得分向量，目的是使得该得分向量中正样本的得分最高，负样本的得分最低
    loss = torch.nn.functional.cross_entropy(sim, y_true)
    loss = torch.mean(loss)
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
