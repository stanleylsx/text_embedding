# -*- coding: utf-8 -*-
# @Time : 2023/10/27 22:05
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : predict.py
# @Software: VSCode
from config import configure
from engines.model import Model
from engines.utils.metrics import MyModel
from mteb import MTEB
import torch
import os


class Predictor:
    def __init__(self, data_manage, device, logger):
        self.logger = logger
        self.data_manage = data_manage
        self.device = device
        self.checkpoints_dir = configure['checkpoints_dir']
        self.model_name = configure['model_name']
        self.model = Model().to(device)
        if not os.path.exists(os.path.join(self.checkpoints_dir, self.model_name)):
            logger.info('Local checkpoint not found, load raw HF model.')
        else:
            self.model.load_state_dict(torch.load(os.path.join(self.checkpoints_dir, self.model_name)))
        self.model.eval()

    @torch.inference_mode()
    def predict_one(self, sentence_a, sentence_b):
        token_ids_a = self.data_manage.tokenizer(sentence_a).input_ids
        token_ids_b = self.data_manage.tokenizer(sentence_b).input_ids
        token_ids_a = torch.tensor([token_ids_a]).to(self.device)
        token_ids_b = torch.tensor([token_ids_b]).to(self.device)
        vector_a = self.model(token_ids_a)
        vector_b = self.model(token_ids_b)
        similarity = float(torch.cosine_similarity(vector_a, vector_b, dim=1).detach().cpu().squeeze(0))
        if_similar = 'similar' if similarity >= self.data_manage.decision_threshold else 'dissimilar'
        return similarity, if_similar

    @torch.inference_mode()
    def get_embedding(self, sentence):
        """
        获取句向量
        """
        token_ids = self.data_manage.batch_tokenize([sentence]).to(self.device)
        vector = self.model(token_ids)
        vector = vector.detach().cpu().squeeze(0).numpy()
        return vector

    def convert_onnx(self):
        max_sequence_length = self.data_manage.max_sequence_length
        dummy_input = torch.ones([1, max_sequence_length]).to('cpu').int()
        onnx_path = self.checkpoints_dir + '/model.onnx'
        torch.onnx.export(self.model.to('cpu'), dummy_input,
                          f=onnx_path,
                          input_names=['input'],
                          output_names=['vector'],
                          dynamic_axes={'input': {0: 'batch_size', 1: 'max_sequence_length'},
                                        'vector': {0: 'batch_size'}})

    def mteb(self):
        model = MyModel(self.data_manage, self.model, self.device)
        task_class = configure['task_class']
        match task_class:
            case 'reranking':
                task_names = ['T2Reranking', 'MMarcoRetrieval', 'CMedQAv1', 'CMedQAv2']
            case 'pairclassification':
                task_names = ['Cmnli', 'Ocnli']
            case 'clustering':
                task_names = ['CLSClusteringS2S', 'CLSClusteringP2P', 'ThuNewsClusteringS2S', 'ThuNewsClusteringP2P']
            case 'sts':
                task_names = ['ATEC', 'BQ', 'LCQMC', 'PAWSX', 'STSB', 'AFQMC', 'QBQTC']
            case 'retrieval':
                task_names = ['T2Retrieval', 'MMarcoRetrieval', 'DuRetrieval', 'CovidRetrieval', 
                              'CmedqaRetrieval', 'EcomRetrieval', 'MedicalRetrieval', 'VideoRetrieval']
        output_dir = os.path.join(self.checkpoints_dir, 'generic_test/' + task_class)
        self.logger.info(f'Total tasks: {task_names}')
        for task in task_names:
            MTEB(tasks=[task], task_langs=['zh', 'zh-CN']).run(model, output_folder=output_dir)
