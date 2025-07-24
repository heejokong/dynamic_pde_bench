import os
import sys
from tqdm import tqdm
import pprint
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import umap

from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from skimage.filters import threshold_otsu

sys.path.append('..')
from operators.core.utils import get_dataset, over_write_args_from_file
from operators.core.criterions import SimpleLpLoss
from operators.algorithms.dpot.model import DPOT2D
from einops import rearrange

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--c', type=str, default='')


def load_model_at(args, step='best'):
    args.step = step
    if step == 'best':
        args.load_path = '/'.join(args.load_path.split('/')[:-1]) + "/model_best.pth"
    elif step == 'latest':
        args.load_path = '/'.join(args.load_path.split('/')[:-1]) + "/latest_model.pth"
    else:
        args.load_path = '/'.join(args.load_path.split('/')[:-1]) + "/model_at_{args.step}_step.pth"
    print(args.load_path)
    checkpoint_path = os.path.join(args.load_path)
    checkpoint = torch.load(checkpoint_path)
    load_model = checkpoint['model']

    load_state_dict = {}
    for key, item in load_model.items():
        if key.startswith('module'):
            new_key = '.'.join(key.split('.')[1:])
            load_state_dict[new_key] = item
        else:
            load_state_dict[key] = item
    save_dir = '/'.join(checkpoint_path.split('/')[:-1])
    if step == 'best':
        args.save_dir = os.path.join(save_dir, f"model_best")
    else:
        args.save_dir = os.path.join(save_dir, f"step_{args.step}")
    # os.makedirs(args.save_dir, exist_ok=True)

    net = DPOT2D(
        img_size=args.res, patch_size=args.patch_size, in_channels=args.n_channels, 
        out_channels=args.n_channels, in_timesteps=args.T_in, out_timesteps=args.T_bundle, 
        n_blocks=args.n_blocks, embed_dim=args.width, out_layer_dim=args.out_layer_dim,
        depth=args.n_layers, modes=args.modes, mlp_ratio=args.mlp_ratio, act=args.act,
        time_agg='mlp',
    )
    net.load_state_dict(load_state_dict, strict=True)
    print(f'Model at step {args.step} loaded!')
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    return net


def evaluate(args, net, dataset):
    eval_loader = DataLoader(dataset, batch_size=args.eval_batch_size, drop_last=False, shuffle=False, num_workers=4)
    criterion = SimpleLpLoss(size_average=False, reduction=False)

    total_num = 0.
    l2_steps = 0.
    l2_fulls = 0.

    net.eval()
    with torch.no_grad():
        loss_all = [[] for _ in range(args.T_in)]
        for data in tqdm(eval_loader):
            loss = 0.

            x = data['x_lb']
            y = data['y_lb']

            if isinstance(x, dict):
                _x = {k: v.cuda() for k, v in x.items()}
            else:
                _x = x.cuda()
            y = y.cuda()

            num_batch = y.shape[0]
            total_num += num_batch

            for t in range(0, y.shape[-2], args.T_bundle):
                _y = y[..., t:t + args.T_bundle, :]
                y_hat = net(_x)
                _loss = criterion(y_hat, _y)
                loss += torch.sum(_loss)

                loss_all[t].append(_loss)
                if t == 0:
                    pred = y_hat
                else:
                    pred = torch.cat((pred, y_hat), -2)

                _x = torch.cat((_x[..., args.T_bundle:,:], y_hat), dim=-2)

            l2_steps += loss
            l2_fulls += torch.sum(criterion(pred, y))

    l2_steps = l2_steps / total_num / (y.shape[-2] / args.T_bundle)
    l2_fulls = l2_fulls / total_num

    eval_dict = {'/loss_step': l2_steps.item(), '/loss_full': l2_fulls.item()}

    # 
    print(eval_dict)

    # 
    losses = []
    for i in range(args.T_in):
        _l = torch.cat(loss_all[i])
        losses.append(_l)
    losses = torch.stack(losses)
    mean_losses = torch.mean(losses, dim=1)
    var_losses = torch.var(losses, dim=1)
    print(mean_losses)
    print(var_losses)

    return eval_dict

"""  """
seed_list = [1]
dataset_list = ['ns2d_fno_1e-5']
algorithm_list = ['dpot']

for seed in seed_list:
    for aidx in algorithm_list:
        for didx in dataset_list:
            args = parser.parse_args(args=['--c', f'config/{aidx}/fnobench/{aidx}_tiny_{didx}_{seed}.yaml'])
            over_write_args_from_file(args, args.c)

            dataset_dict = get_dataset(args.dataset, args.dataset_type, normalize=args.normalize,)
            base_root = "./saved_models"
            # args.load_path = f"{base_root}/fnobench/{aidx}/{aidx}_tiny_{didx}_{seed}/model_best.pth"
            # args.load_path = f"{base_root}/fnobench_bu/{aidx}_ar/{aidx}_tiny_{didx}_{seed}/model_best.pth"
            args.load_path = f"{base_root}/fnobench_bu/{aidx}_tf/{aidx}_tiny_{didx}_{seed}/model_best.pth"
            if not os.path.exists(args.load_path):
                continue
            best_net = load_model_at('best')
            # best_net = load_model_at('latest')
            # evaluate(args, best_net, dataset_dict['test'])
            evaluate(args, best_net, dataset_dict['train'])
