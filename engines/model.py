# -*- coding: utf-8 -*-
# @Time : 2023/10/27 22:05
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : model.py
# @Software: VSCode
from transformers import BertModel, XLMRobertaModel, RoFormerModel
from config import configure
import torch


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model_type = configure['model_type']
        self.emb_type = configure['emb_type']
        if self.model_type == 'XLMRoberta':
            self.model = XLMRobertaModel.from_pretrained(configure['hf_tag'])
        elif self.model_type == 'RoFormer':
            self.tokenizer = RoFormerModel.from_pretrained(configure['hf_tag'])
        elif self.model_type == 'Bert':
            self.model = BertModel.from_pretrained(configure['hf_tag'])
        else:
            raise ValueError('model_type must be in [XLMRoberta, RoFormer, Bert]')

    def forward(self, input_ids):
        attention_mask = torch.where(input_ids > 0, 1, 0)
        model_output = self.model(input_ids, attention_mask=attention_mask)
        if self.emb_type == 'last-avg':
            token_embeddings = model_output[0]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            vectors = sum_embeddings / sum_mask
        elif self.emb_type == 'cls':
            vectors = model_output.last_hidden_state[:, 0]
        vectors = torch.nn.functional.normalize(vectors, 2.0, dim=1)
        return vectors
