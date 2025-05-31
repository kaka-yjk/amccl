import os
import time
import logging
import argparse
import numpy as np
import pickle as plk
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim
from models.contrastive_loss import contrastive_loss
from utils.functions import dict_to_str
from utils.metricsTop import MetricsTop
from transformers import get_cosine_schedule_with_warmup
import matplotlib.pyplot as plt

logger = logging.getLogger('MSA')

class AMCCL:
    def __init__(self, args):
        assert args.train_mode == 'regression'
        self.args = args
        self.args.tasks = "M"
        self.metrics = MetricsTop(args.train_mode).getMetics(args.datasetName)

        # 初始化特征映射和标签映射
        self.feature_map = {
            'fusion': torch.zeros(args.train_samples, args.post_fusion_dim, requires_grad=False).to(args.device),
            'text': torch.zeros(args.train_samples, args.post_text_dim, requires_grad=False).to(args.device),
            'audio': torch.zeros(args.train_samples, args.post_audio_dim, requires_grad=False).to(args.device),
            'vision': torch.zeros(args.train_samples, args.post_video_dim, requires_grad=False).to(args.device),
        }
        self.label_map = {
            'fusion': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'text': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'audio': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'vision': torch.zeros(args.train_samples, requires_grad=False).to(args.device)
        }
        self.name_map = {'M': 'fusion', 'T': 'text', 'A': 'audio', 'V': 'vision'}

    def do_train(self, model, dataloader):
        # 记录训练参数（仅写入文件日志）
        logger.info(
            f"Training for up to {self.args.num_epochs} epochs, warm_up_epochs={self.args.warm_up_epochs}, early_stop={self.args.early_stop}")

        # 参数分组
        bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        bert_params = list(model.Model.text_model.named_parameters())
        audio_params = list(model.Model.audio_model.named_parameters())
        video_params = list(model.Model.video_model.named_parameters())

        bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
        bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
        audio_params = [p for n, p in audio_params]
        video_params = [p for n, p in video_params]
        model_params_other = [p for n, p in list(model.named_parameters()) if 'text_model' not in n and \
                              'audio_model' not in n and 'video_model' not in n]

        optimizer_grouped_parameters = [
            {'params': bert_params_decay, 'weight_decay': self.args.weight_decay_bert,
             'lr': self.args.learning_rate_bert},
            {'params': bert_params_no_decay, 'weight_decay': 0.0, 'lr': self.args.learning_rate_bert},
            {'params': audio_params, 'weight_decay': self.args.weight_decay_audio, 'lr': self.args.learning_rate_audio},
            {'params': video_params, 'weight_decay': self.args.weight_decay_video, 'lr': self.args.learning_rate_video},
            {'params': model_params_other, 'weight_decay': self.args.weight_decay_other,
             'lr': self.args.learning_rate_other}
        ]
        optimizer = optim.AdamW(optimizer_grouped_parameters)
        num_training_steps = len(dataloader['train']) * self.args.num_epochs
        num_warmup_steps = int(num_training_steps * 0.1)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_training_steps)

        # 初始化标签
        logger.info("Initializing labels...")
        with tqdm(dataloader['train'], desc="Init Labels") as td:
            for batch_data in td:
                labels_m = batch_data['labels']['M'].view(-1).to(self.args.device)
                indexes = batch_data['index'].view(-1)
                self.init_labels(indexes, labels_m)

        # 开始训练
        logger.info("Start training...")
        epochs, best_epoch = 0, 0
        train_losses, valid_losses, valid_metrics, learning_rates = [], [], [], []
        epoch_metrics = []  # 新增：存储每个 Epoch 的指标
        task_loss_sum, cl_loss_sum, num_batches = 0.0, 0.0, 0
        min_or_max = 'min' if self.args.KeyEval in ['MAE', 'Loss'] else 'max'
        best_valid = float('inf') if min_or_max == 'min' else float('-inf')

        while epochs < self.args.num_epochs:
            epochs += 1
            model.train()
            epoch_train_loss, epoch_cl_loss = 0.0, 0.0
            y_pred, y_true = {'M': []}, {'M': []}
            ids = []

            with tqdm(dataloader['train'], desc=f"Epoch {epochs}/{self.args.num_epochs}") as td:
                optimizer.zero_grad()
                for i, batch_data in enumerate(td):
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    indexes = batch_data['index'].view(-1)
                    ids.extend(batch_data['id'])

                    audio_lengths = batch_data['audio_lengths'].to(
                        self.args.device) if not self.args.need_data_aligned else None
                    vision_lengths = batch_data['vision_lengths'].to(
                        self.args.device) if not self.args.need_data_aligned else None
                    outputs = model(text, (audio, audio_lengths), (vision, vision_lengths))

                    outputs['Feature_v'] = outputs['Feature_v'].float()
                    outputs['Feature_a'] = outputs['Feature_a'].float()
                    outputs['Feature_t'] = outputs['Feature_t'].float()
                    outputs['M'] = outputs['M'].float()

                    y_pred['M'].append(outputs['M'].detach().cpu())
                    current_labels = self.label_map['fusion'][indexes].float()
                    if current_labels.dim() > 1:
                        current_labels = current_labels.squeeze()
                    if current_labels.dim() == 0:
                        current_labels = current_labels.unsqueeze(0)

                    y_true['M'].append(current_labels.cpu())

                    regression_loss = self.l1_loss(outputs['M'], current_labels)
                    contrastive_loss_fn = contrastive_loss(
                        dataset_name=self.args.datasetName,
                        device=self.args.device,
                        k_threshold=self.args.k_threshold,
                        gain=self.args.gain,
                        temperature=self.args.temperature,
                        lambda_refine=self.args.lambda_refine,
                        alpha=self.args.alpha,
                        inter_weight=self.args.inter_weight,
                        intra_weight=self.args.intra_weight,
                        gamma=self.args.gamma,
                        seed=self.args.seed
                    )
                    contrastive_loss_val = contrastive_loss_fn(outputs, current_labels, task_loss=regression_loss)
                    total_loss = regression_loss + self.args.gamma * contrastive_loss_val

                    total_loss.backward()
                    epoch_train_loss += regression_loss.item()
                    epoch_cl_loss += contrastive_loss_val.item()
                    task_loss_sum += regression_loss.item()
                    cl_loss_sum += contrastive_loss_val.item()
                    num_batches += 1
                    learning_rates.append(optimizer.param_groups[0]['lr'])

                    if (i + 1) % self.args.update_epochs == 0 or (i + 1) == len(dataloader['train']):
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

            avg_train_loss = epoch_train_loss / len(dataloader['train'])
            avg_cl_loss = epoch_cl_loss / len(dataloader['train'])
            train_losses.append(avg_train_loss + self.args.gamma * avg_cl_loss)
            logger.info(
                f"TRAIN-{self.args.modelName} Epoch {epochs} >> Total Loss: {train_losses[-1]:.4f}, Reg Loss: {avg_train_loss:.4f}, CL Loss: {avg_cl_loss:.4f}")

            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            valid_losses.append(val_results['Loss'])
            valid_metrics.append(val_results[self.args.KeyEval])

            # 记录当前 Epoch 的指标
            epoch_metrics.append({
                'epoch': epochs,
                'Total Train Loss': train_losses[-1],
                'Valid Loss': val_results['Loss'],
                'Has0_acc_2': val_results.get('Has0_acc_2', 0.0),
                'Has0_F1_score': val_results.get('Has0_F1_score', 0.0),
                'Non0_acc_2': val_results.get('Non0_acc_2', 0.0),
                'Non0_F1_score': val_results.get('Non0_F1_score', 0.0),
                'Mult_acc_7': val_results.get('Mult_acc_7', 0.0),
                'MAE': val_results.get('MAE', float('inf')),
                'Corr': val_results.get('Corr', 0.0),
            })

            # 在控制台输出 epoch 指标
            print(
                f"Epoch {epochs}: Total Train Loss: {train_losses[-1]:.4f}, Valid Loss: {val_results['Loss']:.4f}, Valid {self.args.KeyEval}: {val_results[self.args.KeyEval]:.4f}")

            isBetter = val_results[self.args.KeyEval] < best_valid if min_or_max == 'min' else val_results[
                                                                                                   self.args.KeyEval] > best_valid
            if isBetter:
                best_valid, best_epoch = val_results[self.args.KeyEval], epochs
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)

            if epochs - best_epoch >= self.args.early_stop:
                logger.info(f"Training stopped at epoch {epochs}")
                break

        avg_task_loss_total = task_loss_sum / num_batches if num_batches > 0 else 0.0
        avg_cl_loss_total = cl_loss_sum / num_batches if num_batches > 0 else 0.0
        return {}, avg_task_loss_total

    def do_test(self, model, dataloader, mode="VAL"):
        model.eval()
        y_pred, y_true = {'M': []}, {'M': []}
        eval_loss = 0.0
        with torch.no_grad():
            with tqdm(dataloader, desc=f"{mode} Evaluation") as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    audio_lengths = batch_data['audio_lengths'].to(
                        self.args.device) if not self.args.need_data_aligned else None
                    vision_lengths = batch_data['vision_lengths'].to(
                        self.args.device) if not self.args.need_data_aligned else None
                    labels_m = batch_data['labels']['M'].to(self.args.device).view(-1)
                    outputs = model(text, (audio, audio_lengths), (vision, vision_lengths))
                    loss = self.l1_loss(outputs['M'], labels_m)
                    eval_loss += loss.item()
                    y_pred['M'].append(outputs['M'].cpu())
                    y_true['M'].append(labels_m.cpu())
        avg_eval_loss = eval_loss / len(dataloader)
        # 仅在文件日志中记录详细评估信息
        logger.info(f"{mode}-{self.args.modelName} >> Loss: {avg_eval_loss:.4f}")
        pred, true = torch.cat(y_pred['M']), torch.cat(y_true['M'])
        eval_results = self.metrics(pred, true)
        logger.info(f"{mode} Metrics: M: >> {dict_to_str(eval_results)}")
        eval_results['Loss'] = avg_eval_loss
        return eval_results

    def l1_loss(self, y_pred, y_true):
        return torch.mean(torch.abs(y_pred.view(-1) - y_true.view(-1)))

    def init_labels(self, indexes, m_labels):
        for modality in self.label_map:
            self.label_map[modality][indexes] = m_labels.float()