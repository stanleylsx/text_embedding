# -*- coding: utf-8 -*-
# @Time : 2023/10/27 22:05
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : data.py
# @Software: VSCode
from transformers import AutoTokenizer, XLMRobertaTokenizer, BertTokenizer, RoFormerTokenizer
from config import configure
import torch


class DataPrecess:
    """
    文本处理
    """

    def __init__(self, logger):
        super(DataPrecess, self).__init__()
        self.logger = logger
        self.max_position_embeddings = configure['max_position_embeddings']
        self.decision_threshold = configure['decision_threshold']
        self.train_type = configure['train_type']
        if configure['model_type'] == 'XLMRoberta':
            self.tokenizer = XLMRobertaTokenizer.from_pretrained(configure['hf_tag'])
        elif configure['model_type'] == 'RoFormer':
            self.tokenizer = RoFormerTokenizer.from_pretrained(configure['hf_tag'])
        elif configure['model_type'] == 'Bert':
            self.tokenizer = BertTokenizer.from_pretrained(configure['hf_tag'])
        elif configure['model_type'] == 'GTE':
            self.tokenizer = AutoTokenizer.from_pretrained(configure['hf_tag'])

    def prepare_pair_data(self, df_values):
        inputs_a, inputs_b, labels = [], [], []
        for sentence1, sentence2, label in df_values:
            inputs_a.append(sentence1)
            inputs_b.append(sentence2)
            labels.append(label)
        inputs_a = self.tokenizer.batch_encode_plus(inputs_a,
                                                    padding='longest',
                                                    truncation=True,
                                                    max_length=self.max_position_embeddings,
                                                    return_tensors='pt')
        inputs_b = self.tokenizer.batch_encode_plus(inputs_b,
                                                    padding='longest',
                                                    truncation=True,
                                                    max_length=self.max_position_embeddings,
                                                    return_tensors='pt')
        token_ids_a, token_ids_b = inputs_a['input_ids'], inputs_b['input_ids']
        return token_ids_a, token_ids_b, torch.tensor(labels)

    def prepare_simcse_sup_data(self, df_values):
        triple_sentences = []
        for sentence, entailment, contradiction in df_values:
            triple_sentences.extend([sentence, entailment, contradiction])
        inputs = self.tokenizer.batch_encode_plus(triple_sentences,
                                                  max_length=self.max_position_embeddings,
                                                  truncation=True,
                                                  padding='longest',
                                                  return_tensors='pt')
        token_ids = inputs['input_ids']
        return token_ids

    def prepare_simcse_unsup_data(self, df_values):
        sentences = []
        for sentence in df_values:
            sentence = sentence[0]
            sentences.extend([sentence, sentence])
        inputs = self.tokenizer.batch_encode_plus(sentences,
                                                  max_length=self.max_position_embeddings,
                                                  truncation=True,
                                                  padding='longest',
                                                  return_tensors='pt')
        token_ids = inputs['input_ids']
        return token_ids

    def get_dataset(self, df_values):
        """
        构建Dataset
        """
        if self.train_type == 'cosent':
            inputs_a, inputs_b, labels = self.prepare_pair_data(df_values)
            dataset = (inputs_a, inputs_b, labels)
        elif self.train_type == 'simcse_sup':
            dataset = self.prepare_simcse_sup_data(df_values)
        elif self.train_type == 'simcse_unsup':
            dataset = self.prepare_simcse_unsup_data(df_values)
        return dataset

    def get_eval_dataset(self, df_values):
        """
        构建验证集Dataset
        """
        inputs_a, inputs_b, labels = self.prepare_pair_data(df_values)
        dataset = (inputs_a, inputs_b, labels)
        return dataset

    def batch_tokenize(self, sentences):
        token_ids = self.tokenizer.batch_encode_plus(sentences,
                                                     max_length=self.max_position_embeddings,
                                                     truncation=True,
                                                     padding='longest',
                                                     return_tensors='pt').input_ids
        return token_ids
