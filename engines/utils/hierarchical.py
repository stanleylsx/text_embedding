# -*- coding: utf-8 -*-
# @Time : 2023/11/21 22:05
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : hierarchical.py
# @Software: VSCode
from torch.nn.modules.sparse import Embedding
from config import configure
import torch


class BertHierarchicalPositionEmbedding(Embedding):
    """
    分层位置编码PositionEmbedding
    """
    def __init__(self, num_embeddings, embedding_dim=768):
        super().__init__(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.input_dim = 512
        self.alpha = configure['hierarchical_alpha']

    def forward(self, input_data):
        # 输入的序列[batch, seq_len]
        input_shape = input_data.shape
        seq_len = input_shape[1]
        # 根据序列的长度随机生成一个pytorch的张量
        position_ids = torch.arange(0, seq_len, dtype=torch.int64, device=self.weight.device)
        # 计算i, j的权重值，分子：当前的权重减去 alph乘权重中第一个位置内容(CLS标签)的权重值
        embeddings = self.weight - self.alpha * self.weight[:1]
        embeddings = embeddings / (1 - self.alpha)
        # 计算组内的权重
        embeddings_x = torch.index_select(embeddings, 0, torch.div(position_ids, self.input_dim, rounding_mode='trunc'))
        # 计算组间的权重
        embeddings_y = torch.index_select(embeddings, 0, position_ids % self.input_dim)
        embeddings = self.alpha * embeddings_x + (1 - self.alpha) * embeddings_y
        return embeddings


def hierarchical_position(model):
    """
    通过bert预训练权重创建BertHierarchicalPositionEmbedding并返回
    """
    max_sequence_length = configure['max_sequence_length']
    # 加载bert预训练文件中的position embedding的weight
    embedding_weight = model.embeddings.position_embeddings.weight
    # 先创建一个分层的embedding
    hierarchical_position = BertHierarchicalPositionEmbedding(num_embeddings=max_sequence_length)
    # 把已经训练好的embedding替换创建的embedding
    hierarchical_position.weight.data.copy_(embedding_weight)
    # 不参与模型训练
    hierarchical_position.weight.requires_grad = False
    return hierarchical_position
