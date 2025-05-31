import os
import argparse
from utils.functions import Storage


class ConfigRegression():
    def __init__(self, args):
        # hyper parameters for models
        HYPER_MODEL_MAP = {
            'amccl': self.__AMCCL
        }
        # hyper parameters for datasets
        HYPER_DATASET_MAP = self.__datasetCommonParams()

        # normalize
        model_name = str.lower(args.modelName)
        dataset_name = str.lower(args.datasetName)
        # load params
        commonArgs = HYPER_MODEL_MAP[model_name]()['commonParas']
        dataArgs = HYPER_DATASET_MAP[dataset_name]
        dataArgs = dataArgs['aligned'] if (commonArgs['need_data_aligned'] and 'aligned' in dataArgs) else dataArgs[
            'unaligned']
        # integrate all parameters
        self.args = Storage(dict(vars(args),
                                 **dataArgs,
                                 **commonArgs,
                                 **HYPER_MODEL_MAP[model_name]()['datasetParas'][dataset_name],
                                 ))

    def __datasetCommonParams(self):
        root_dataset_dir = '/data/'
        tmp = {
            'mosi': {
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSI/Processed/unaligned_50.pkl'),
                    'seq_lens': (50, 50, 50),
                    'feature_dims': (768, 5, 20),
                    'train_samples': 1284,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'MAE'
                }
            },
            'mosei': {
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSEI/Processed/unaligned_50.pkl'),
                    'seq_lens': (50, 500, 375),
                    'feature_dims': (768, 74, 35),
                    'train_samples': 16326,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'MAE'
                }
            },
            'sims': {
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'SIMS/Processed/unaligned_39.pkl'),
                    'seq_lens': (39, 400, 55),
                    'feature_dims': (768, 33, 709),
                    'train_samples': 1368,
                    'num_classes': 3,
                    'language': 'cn',
                    'KeyEval': 'MAE',
                }
            },
            'simsv2': {
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'SIMS_V2/ch-simsv2s.pkl'),
                    'seq_lens': (50, 925, 232),
                    'feature_dims': (768, 25, 177),
                    'train_samples': 2722,
                    'num_classes': 3,
                    'language': 'cn',
                    'KeyEval': 'MAE',
                }
            }
        }
        return tmp

    def __AMCCL(self):
        tmp = {
            'commonParas': {
                'need_data_aligned': False,
                'need_model_aligned': False,
                'need_normalized': False,
                'use_bert': True,
                'use_finetune': True,
                'save_labels': False,
            },
            'datasetParas': {
                'mosi': {
                    'batch_size': 64,  # 64
                    'learning_rate_bert': 5e-5,
                    'learning_rate_audio': 1e-3,
                    'learning_rate_video': 5e-4,
                    'learning_rate_other': 5e-4,
                    'weight_decay_bert': 0.01,
                    'weight_decay_audio': 0.01,
                    'weight_decay_video': 0.01,
                    'weight_decay_other': 0.001,
                    'fusion_filter_nums': 16,
                    'a_encoder_heads': 1,
                    'v_encoder_heads': 4,
                    'a_encoder_layers': 2,
                    'v_encoder_layers': 2,
                    'text_out': 768,
                    'audio_out': 5,
                    'video_out': 20,
                    't_bert_dropout': 0.1,
                    'post_fusion_dim': 128,
                    'post_text_dim': 64,
                    'post_audio_dim': 64,
                    'post_video_dim': 64,
                    'post_fusion_dropout': 0.2,
                    'post_text_dropout': 0.1,
                    'post_audio_dropout': 0.1,
                    'post_video_dropout': 0.1,
                    'skip_net_reduction': 2,
                    'warm_up_epochs': 15,
                    'gamma': 0.01,
                    'inter_weight': 0.2,
                    'intra_weight': 1,
                    'num_epochs': 150,
                    'update_epochs': 1,
                    'early_stop': 30,
                    'H': 3.0,
                    'k_threshold': 0.5,
                    'gain': 1.5,
                    'temperature': 0.03,
                    'lambda_refine': 4,
                    'alpha': 0.8
                },
                'mosei': {
                    'batch_size': 128,
                    'learning_rate_bert': 5e-5,
                    'learning_rate_audio': 5e-4,
                    'learning_rate_video': 5e-4,
                    'learning_rate_other': 25e-4,
                    'weight_decay_bert': 0.001,
                    'weight_decay_audio': 0.0,
                    'weight_decay_video': 0.0,
                    'weight_decay_other': 0.01,
                    'fusion_filter_nums': 16,
                    'a_encoder_heads': 2,
                    'v_encoder_heads': 5,
                    'a_encoder_layers': 2,
                    'v_encoder_layers': 2,
                    'text_out': 768,
                    'audio_out': 74,
                    'video_out': 35,
                    't_bert_dropout': 0.1,
                    'post_fusion_dim': 256,
                    'post_text_dim': 128,
                    'post_audio_dim': 128,
                    'post_video_dim': 128,
                    'post_fusion_dropout': 0.1,
                    'post_text_dropout': 0.01,
                    'post_audio_dropout': 0.01,
                    'post_video_dropout': 0.01,
                    'skip_net_reduction': 8,
                    'warm_up_epochs': 20,
                    'gamma': 0.005,
                    'inter_weight': 0.2,
                    'intra_weight': 1,
                    'num_epochs': 100,
                    'update_epochs': 1,
                    'early_stop': 20,
                    'H': 3.0,
                    'k_threshold': 0.5,
                    'gain': 1.5,
                    'temperature': 0.03,
                    'lambda_refine': 4,
                    'alpha': 0.4
                },
                'sims': {
                    'batch_size': 64,
                    'learning_rate_bert': 5e-5,
                    'learning_rate_audio': 5e-4,
                    'learning_rate_video': 5e-4,
                    'learning_rate_other': 5e-4,
                    'weight_decay_bert': 0.001,
                    'weight_decay_audio': 0.01,
                    'weight_decay_video': 0.01,
                    'weight_decay_other': 0.001,
                    'fusion_filter_nums': 16,
                    'a_encoder_heads': 3,
                    'v_encoder_heads': 1,
                    'a_encoder_layers': 2,
                    'v_encoder_layers': 2,
                    'text_out': 768,
                    'audio_out': 33,
                    'video_out': 709,
                    't_bert_dropout': 0.1,
                    'post_fusion_dim': 256,
                    'post_text_dim': 128,
                    'post_audio_dim': 128,
                    'post_video_dim': 128,
                    'post_fusion_dropout': 0.1,
                    'post_text_dropout': 0.4,
                    'post_audio_dropout': 0.4,
                    'post_video_dropout': 0.4,
                    'skip_net_reduction': 8,
                    'warm_up_epochs': 15,
                    'update_epochs': 1,
                    'early_stop': 30,
                    'gamma': 0.0025,
                    'inter_weight': 1,
                    'intra_weight': 1,
                    'num_epochs': 150,
                    'H': 1.0,
                    'k_threshold': 0.5,
                    'gain': 1.5,
                    'temperature': 0.03,
                    'lambda_refine': 3,
                    'alpha': 0.4
                }
            },
        }
        return tmp

    def get_config(self):
        return self.args