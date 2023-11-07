# -*- coding: utf-8 -*-
# @Time : 2023/10/27 22:05
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : train.py
# @Software: VSCode
from engines.model import Model
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from engines.utils.metrics import cal_metrics, compute_corrcoef
from engines.utils.losses import cosent_loss, get_mean_params, ewc_loss, simcse_sup_loss, simcse_unsup_loss
from config import configure
import pandas as pd
import torch
import time
import math
import os


class Train:
    def __init__(self, data_manage, device, logger):
        self.logger = logger
        self.device = device
        self.data_manage = data_manage
        self.decision_threshold = data_manage.decision_threshold
        self.train_type = data_manage.train_type
        self.use_fp16 = configure['use_fp16']

    @torch.inference_mode()
    def evaluate(self, model, val_loader):
        """
        验证集评估函数，分别计算f1、precision、recall和spearmanr相关系数
        """
        model.eval()
        start_time = time.time()
        loss_sum = 0.0
        all_predicts = []
        all_labels = []
        preds_sims = []
        for _, batch in enumerate(tqdm(val_loader)):
            input_a, input_b, labels = batch
            input_a, input_b, labels = input_a.to(self.device), input_b.to(self.device), labels.to(self.device)
            vectors_a, vectors_b = model(input_a), model(input_b)
            pred_sims = torch.cosine_similarity(vectors_a, vectors_b, dim=1)
            loss = cosent_loss(pred_sims, labels, self.device)
            loss_sum += loss.item()
            predicts = torch.where(pred_sims >= self.decision_threshold, 1, 0)
            preds_sims.extend(pred_sims.cpu().numpy())
            all_predicts.extend(predicts.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        val_time = time.time() - start_time
        val_loss = loss_sum / len(val_loader)
        val_measures = cal_metrics(all_predicts, all_labels)
        val_measures |= compute_corrcoef(all_labels, preds_sims)
        # 打印验证集上的指标
        res_str = ''
        for k, v in val_measures.items():
            res_str += (k + ': %.3f ' % v)
        self.logger.info('loss: %.5f, %s' % (val_loss, res_str))
        self.logger.info('time consumption of evaluating:%.2f(min)' % val_time)
        return val_measures

    def train(self):
        batch_size = 256
        epoch = configure['epochs']
        learning_rate = configure['learning_rate']
        batch_size = configure['batch_size']
        gradient_accumulation_steps = configure['gradient_accumulation_steps']
        print_per_batch = configure['print_per_batch']
        train_file = configure['train_file']
        val_file = configure['val_file']
        train_data = pd.read_csv(train_file, encoding='utf-8')

        patience = configure['patience']
        is_early_stop = configure['is_early_stop']
        checkpoints_dir = configure['checkpoints_dir']
        model_name = configure['model_name']
        best_f1 = 0.0
        best_at_epoch = 0
        patience_counter = 0

        very_start_time = time.time()
        self.logger.info('train_data_length:{}\n'.format(len(train_data)))
        train_loader = DataLoader(dataset=train_data.values,
                                  collate_fn=self.data_manage.get_dataset,
                                  shuffle=True,
                                  batch_size=batch_size)

        if val_file != '':
            val_data = pd.read_csv(val_file, encoding='utf-8')
            if val_data.columns.tolist() != ['sentence1', 'sentence2', 'label']:
                raise ValueError('val_file format error')
            self.logger.info('val_data_length:{}\n'.format(len(val_data)))
            val_loader = DataLoader(dataset=val_data.values,
                                    collate_fn=self.data_manage.get_eval_dataset,
                                    shuffle=False,
                                    batch_size=batch_size)

        total_steps = len(train_loader) * epoch
        num_train_optimization_steps = int(len(train_data) / batch_size / gradient_accumulation_steps) * epoch
        self.logger.info(f'Num steps:{num_train_optimization_steps}')
        model = Model().to(self.device)
        params = list(model.parameters())
        optimizer = AdamW(params, lr=learning_rate)
        if self.use_fp16:
            scaler = GradScaler()

        if os.path.exists(os.path.join(checkpoints_dir, model_name)):
            self.logger.info('Resuming from checkpoint...')
            model.load_state_dict(torch.load(os.path.join(checkpoints_dir, model_name)))
            optimizer_checkpoint = torch.load(os.path.join(checkpoints_dir, model_name + '.optimizer'))
            optimizer.load_state_dict(optimizer_checkpoint['optimizer'])
        else:
            self.logger.info('Initializing from scratch.')

        if configure['use_ewc']:
            original_weight = get_mean_params(model)

        # 定义梯度策略
        warmup_steps = math.ceil(total_steps * configure['warmup_ratio'])
        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                    num_warmup_steps=warmup_steps,
                                                    num_training_steps=total_steps)

        self.logger.info(('+' * 20) + 'training starting' + ('+' * 20))

        for i in range(epoch):
            train_start = time.time()
            self.logger.info('epoch:{}/{}'.format(i + 1, epoch))
            loss, loss_sum = 0.0, 0.0
            model.train()

            for step, batch in enumerate(tqdm(train_loader)):
                if self.train_type == 'cosent':
                    input_a, input_b, labels = batch
                    input_a, input_b, labels = input_a.to(self.device), input_b.to(self.device), labels.to(self.device)
                    if self.use_fp16:
                        with autocast():
                            vectors_a, vectors_b = model(input_a), model(input_b)
                            pred_sims = torch.cosine_similarity(vectors_a, vectors_b, dim=1)
                            loss = cosent_loss(pred_sims, labels, self.device)
                    else:
                        vectors_a, vectors_b = model(input_a), model(input_b)
                        pred_sims = torch.cosine_similarity(vectors_a, vectors_b, dim=1)
                        loss = cosent_loss(pred_sims, labels, self.device)
                else:
                    batch = batch.to(self.device)
                    if self.use_fp16:
                        with autocast():
                            out = model(batch)
                            if self.train_type == 'simcse_sup':
                                loss = simcse_sup_loss(out, self.device)
                            elif self.train_type == 'simcse_unsup':
                                loss = simcse_unsup_loss(out, self.device)
                    else:
                        out = model(batch)
                        if self.train_type == 'simcse_sup':
                            loss = simcse_sup_loss(out, self.device)
                        elif self.train_type == 'simcse_unsup':
                            loss = simcse_unsup_loss(out, self.device)

                if configure['use_ewc']:
                    loss = loss + ewc_loss(model, original_weight)

                loss_sum += loss.item()
                if self.use_fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                if (step + 1) % gradient_accumulation_steps == 0:
                    if self.use_fp16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                # 打印训练过程中的指标
                if step % print_per_batch == 0 and step != 0:
                    if self.train_type == 'cosent':
                        out_classes = torch.where(pred_sims >= self.decision_threshold, 1, 0)
                        measures = cal_metrics(out_classes.cpu(), labels.cpu())
                        measures |= compute_corrcoef(labels.cpu().numpy(), pred_sims.cpu().detach().numpy())
                        res_str = ''
                        for k, v in measures.items():
                            res_str += (k + ': %.3f ' % v)
                        self.logger.info('training step: %5d, loss: %.5f, %s' % (step, loss, res_str))
                    else:
                        self.logger.info('training step: %5d, loss: %.5f' % (step, loss))

            train_time = (time.time() - train_start) / 60
            self.logger.info('time consumption of training:%.2f(min)' % train_time)
            if val_file != '':
                self.logger.info('start evaluate model...')
                val_measures = self.evaluate(model, val_loader)

                if val_measures['f1'] > best_f1:
                    patience_counter = 0
                    best_f1 = val_measures['f1']
                    best_at_epoch = i + 1
                    optimizer_checkpoint = {'optimizer': optimizer.state_dict()}
                    torch.save(optimizer_checkpoint, os.path.join(checkpoints_dir, model_name + '.optimizer'))
                    torch.save(model.state_dict(), os.path.join(checkpoints_dir, model_name))
                    self.logger.info('saved the new best model with f1: %.3f' % best_f1)
                else:
                    patience_counter += 1

                if is_early_stop:
                    if patience_counter >= patience:
                        self.logger.info('early stopped, no progress obtained within {} epochs'.format(patience))
                        self.logger.info('overall best f1 is {} at {} epoch'.format(best_f1, best_at_epoch))
                        self.logger.info('total training time consumption: %.3f(min)' % ((time.time() - very_start_time) / 60))
                        return
            else:
                optimizer_checkpoint = {'optimizer': optimizer.state_dict()}
                torch.save(optimizer_checkpoint, os.path.join(checkpoints_dir, model_name + '.optimizer'))
                torch.save(model.state_dict(), os.path.join(checkpoints_dir, model_name))
                self.logger.info('saved the current model')
        if val_file != '':
            self.logger.info('overall best f1 is {} at {} epoch'.format(best_f1, best_at_epoch))
        self.logger.info('total training time consumption: %.3f(min)' % ((time.time() - very_start_time) / 60))
        self.logger.info('total training time consumption: %.3f(min)' % ((time.time() - very_start_time) / 60))
