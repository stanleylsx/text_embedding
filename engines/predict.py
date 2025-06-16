# -*- coding: utf-8 -*-
# @Time : 2023/10/27 22:05
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : predict.py
# @Software: VSCode
from config import configure
from engines.model import Model
from engines.utils.metrics import MyModel
from torch.utils.data import DataLoader
from mteb import MTEB
import pandas as pd
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
        max_position_embeddings = self.data_manage.max_position_embeddings
        dummy_input = torch.ones([1, max_position_embeddings], dtype=torch.int32).to(self.device).int()
        onnx_path = self.checkpoints_dir + '/model.onnx'
        torch.onnx.export(self.model, dummy_input,
                          f=onnx_path,
                          opset_version=17,
                          do_constant_folding=True,
                          input_names=['input'],
                          output_names=['vector'],
                          dynamic_axes={'input': {0: 'batch_size', 1: 'max_position_embeddings'},
                                        'vector': {0: 'batch_size'}})

    def mteb(self):
        model = MyModel(self.data_manage, self.model, self.device)
        task_class = configure['task_class']
        if task_class == 'reranking':
            task_names = ['T2Reranking', 'MMarcoRetrieval', 'CMedQAv1', 'CMedQAv2']
        elif task_class == 'pairclassification':
            task_names = ['Cmnli', 'Ocnli']
        elif task_class == 'clustering':
            task_names = ['CLSClusteringS2S', 'CLSClusteringP2P', 'ThuNewsClusteringS2S', 'ThuNewsClusteringP2P']
        elif task_class == 'sts':
            task_names = ['ATEC', 'BQ', 'LCQMC', 'PAWSX', 'STSB', 'AFQMC', 'QBQTC']
        elif task_class == 'retrieval':
            task_names = ['T2Retrieval', 'MMarcoRetrieval', 'DuRetrieval', 'CovidRetrieval',
                          'CmedqaRetrieval', 'EcomRetrieval', 'MedicalRetrieval', 'VideoRetrieval']
        output_dir = os.path.join(self.checkpoints_dir, 'generic_test/' + task_class)
        self.logger.info(f'Total tasks: {task_names}')
        for task in task_names:
            MTEB(tasks=[task], task_langs=['zh', 'zh-CN']).run(model, output_folder=output_dir)

    def test(self, trainer):
        test_file = configure['test_file']
        batch_size = configure['batch_size']
        if test_file != '':
            test_data = pd.read_csv(test_file, encoding='utf-8')
            if test_data.columns.tolist() != ['sentence1', 'sentence2', 'label']:
                raise ValueError('test_file format error')
            self.logger.info('test_data_length:{}'.format(len(test_data)))
            test_loader = DataLoader(dataset=test_data.values,
                                     collate_fn=self.data_manage.get_eval_dataset,
                                     shuffle=False,
                                     batch_size=batch_size)
            trainer.evaluate(self.model, test_loader)

    def batch_embedding(self):
        test_file = configure['test_file']
        if test_file != '':
            indices = []
            vectors = []
            sentences = []
            test_data = pd.read_csv(test_file, encoding='utf-8')
            for _, row in test_data.iterrows():
                index = row['index']
                indices.append(index)
                sentence = row['sentence']
                sentences.append(sentence)
                vector = self.get_embedding(sentence)
                vectors.append(vector.tolist())
        test_result = pd.DataFrame({'index': indices, 'sentence': sentences, 'vector': vectors})
        test_result.to_csv('batch_test_result.csv', index=False)
