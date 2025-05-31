import os
import gc
import time
import random
import torch
import pynvml
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from models.AMIO import AMIO
from trains.multiTask.AMCCL import AMCCL
from data.load_data import MMDataLoader
from config.config_regression import ConfigRegression

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def run(args):
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    args.model_save_path = os.path.join(args.model_save_dir,
                                        f'{args.modelName}-{args.datasetName}-{args.train_mode}.pth')

    if len(args.gpu_ids) == 0 and torch.cuda.is_available():
        pynvml.nvmlInit()
        dst_gpu_id, min_mem_used = 0, 1e16
        for g_id in [0, 1, 2, 3]:
            handle = pynvml.nvmlDeviceGetHandleByIndex(g_id)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used = meminfo.used
            if mem_used < min_mem_used:
                min_mem_used = mem_used
                dst_gpu_id = g_id
        logger.info(f'Find gpu: {dst_gpu_id}, with memory: {min_mem_used} left!')
        args.gpu_ids.append(dst_gpu_id)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.device = device
    dataloader = MMDataLoader(args)
    model = AMIO(args).to(device)

    def count_parameters(model):
        answer = 0
        for p in model.parameters():
            if p.requires_grad:
                answer += p.numel()
        return answer

    logger.info(f'The model has {count_parameters(model)} trainable parameters')
    atio = AMCCL(args)
    results, task_loss_avg = atio.do_train(model, dataloader)
    assert os.path.exists(args.model_save_path)
    model.load_state_dict(torch.load(args.model_save_path))
    model.to(device)

    if args.tune_mode:
        test_results = atio.do_test(model, dataloader['valid'], mode="VALID")
    else:
        test_results = atio.do_test(model, dataloader['test'], mode="TEST")

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return test_results

def run_normal(args):
    args.res_save_dir = os.path.join(args.res_save_dir, 'normals')
    init_args = args
    model_results = []
    seeds = args.seeds
    for i, seed in enumerate(seeds):
        args = init_args
        if args.train_mode == "regression":
            config = ConfigRegression(args)
        args = config.get_config()

        setup_seed(seed)
        args.seed = seed
        logger.info(f'Start running {args.modelName}...')
        args.cur_time = i + 1
        test_results = run(args)
        model_results.append(test_results)

        criterions = list(model_results[0].keys())
        save_path = os.path.join(args.res_save_dir,
                                 f'{args.datasetName}-{args.train_mode}.csv')
        if not os.path.exists(args.res_save_dir):
            os.makedirs(args.res_save_dir)
        if os.path.exists(save_path):
            df = pd.read_csv(save_path)
        else:
            df = pd.DataFrame(columns=["Model", "Seed", "gamma"] + criterions)
        for i, test_results in enumerate(model_results):
            res = [args.modelName, f'{seed}', f'{args.gamma}']
            for c in criterions:
                res.append(round(test_results[c] * 100, 2))
            df.loc[len(df)] = res

        df.to_csv(save_path, index=None)
        logger.info(f'Results are added to {save_path}...')
        df = df.iloc[0:0]
        model_results = []

def set_log(args):
    if not os.path.exists('logs'):
        os.makedirs('logs')
    log_file_path = f'logs/{args.modelName}-{args.datasetName}.log'
    logger = logging.getLogger('MSA')  # 使用与 AMCCL 一致的 logger
    logger.setLevel(logging.DEBUG)

    # 移除现有处理器
    for ph in logger.handlers:
        logger.removeHandler(ph)

    # 文件日志：记录所有 DEBUG 及以上级别
    formatter_file = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter_file)
    logger.addHandler(fh)

    # 控制台日志：仅输出 INFO 级别，且由 print 控制
    formatter_stream = logging.Formatter('%(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter_stream)
    logger.addHandler(ch)

    return logger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_mode', type=str, default="regression",
                        help='regression')
    parser.add_argument('--modelName', type=str, default='amccl',
                        help='support AMCCL')
    parser.add_argument('--datasetName', type=str, default='sims',
                        help='support mosi/mosei/sims/simsv2')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num workers of loading data')
    parser.add_argument('--model_save_dir', type=str, default='results/models',
                        help='path to save results.')
    parser.add_argument('--res_save_dir', type=str, default='results/results',
                        help='path to save results.')
    parser.add_argument('--gpu_ids', type=list, default=[0],
                        help='indicates the gpus will be used.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    logger = set_log(args)
    for data_name in ['mosi']:
        args.datasetName = data_name
        args.seeds = [8921]
        run_normal(args)