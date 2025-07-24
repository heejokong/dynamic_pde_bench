import os
import random
import h5py
import math
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from numpy.lib.stride_tricks import sliding_window_view
from einops import rearrange

from operators.datasets.pde_datasets.data_configs import DATASET_DICT


class TemporalGridDataset(Dataset):
    def __init__(self, data_name, normalize=False, train=True):
        super(TemporalGridDataset, self).__init__()
        self.train = train
        self.normalize = normalize
        data_path = DATASET_DICT[data_name]['train_path'] if self.train else DATASET_DICT[data_name]['test_path']

        self.downsample = DATASET_DICT[data_name]['downsample']
        self.downsample_t = DATASET_DICT[data_name]['downsample_t']
        self.in_seq = math.ceil(DATASET_DICT[data_name]['in_seq'] / self.downsample_t)
        if self.train:
            # self.pred_seq = min(self.pred_seq, self.in_seq)
            self.pred_seq = math.ceil(DATASET_DICT[data_name]['pred_seq'] / self.downsample_t)
            self.num_data = DATASET_DICT[data_name]['num_train']
        else:
            self.pred_seq = math.ceil(DATASET_DICT[data_name]['test_seq'] / self.downsample_t)
            self.num_data = DATASET_DICT[data_name]['num_test']
        assert self.pred_seq > 0

        self.spatial_dim = len(DATASET_DICT[data_name]['raw_res'])
        self.raw_res = DATASET_DICT[data_name]['raw_res']
        self.in_res = DATASET_DICT[data_name]['in_res']

        #
        self.scatter_storage = DATASET_DICT[data_name]['scatter_storage']
        if self.scatter_storage:
            data = []
            for i, fname in tqdm(enumerate(os.listdir(data_path))):
                if i >= self.num_data:
                    break
                data_idx = f'{data_path}/{fname}'
                data.append(data_idx)
            # self.data = np.concatenate(data, axis=0)
            self.data = data
        else:
            self.data = h5py.File(data_path, 'r')['data'][..., ::self.downsample_t, :]
            self.data = self.data[:self.num_data]

        # 
        # self.grid = self.get_grid_coord()
        self.grid = self.get_flat_grid_coord()
        # 
        if self.normalize:
            self.mean = np.mean(self.data, axis=(0, 1, 2), keepdims=True).squeeze(0)
            self.std  = np.std(self.data, axis=(0, 1, 2), keepdims=True).squeeze(0) + 1e-8

        print("Done preparing datasets")

    def get_grid_coord(self,):
        if self.spatial_dim == 1:
            size_x = self.in_res
            grid = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float).reshape(size_x, 1)

        elif self.spatial_dim == 2:
            (size_x, size_y) = self.in_res
            gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float) \
                .reshape(size_x, 1, 1).repeat([1, size_y, 1])
            gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float) \
                .reshape(1, size_y, 1).repeat([size_x, 1, 1])
            grid = torch.cat((gridx, gridy), dim=-1)

        elif self.spatial_dim == 3:
            (size_x, size_y, size_z) = self.in_res
            gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float) \
                .reshape(size_x, 1, 1, 1).repeat([1, size_y, size_z, 1])
            gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float) \
                .reshape(1, size_y, 1, 1).repeat([size_x, 1, size_z, 1])
            gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float) \
                .reshape(1, 1, size_z, 1).repeat([size_x, size_y, 1, 1])
            grid = torch.cat((gridx, gridy, gridz), dim=-1)
        return grid

    def get_flat_grid_coord(self,):
        if self.spatial_dim == 1:
            (size_x,) = self.in_res
            gridx = np.linspace(0, 1, size_x)
            grid = torch.tensor(gridx, dtype=torch.float)

        elif self.spatial_dim == 2:
            (size_x, size_y) = self.in_res
            gridx = np.linspace(0, 1, size_x)
            gridy = np.linspace(0, 1, size_y)

            gridx, gridy = np.meshgrid(gridx, gridy, indexing="ij")
            grid = torch.tensor(np.hstack([gridx.flatten()[:, None], gridy.flatten()[:, None]]), dtype=torch.float)

        elif self.spatial_dim == 3:
            (size_x, size_y, size_z) = self.in_res
            gridx = np.linspace(0, 1, size_x)
            gridy = np.linspace(0, 1, size_y)
            gridz = np.linspace(0, 1, size_z)

            gridx, gridy, gridz = np.meshgrid(gridx, gridy, gridz, indexing="ij")
            grid = torch.tensor(np.hstack([gridx.flatten()[:, None], gridy.flatten()[:, None], gridz.flatten()[:, None]]), dtype=torch.float)
        return grid

    def transform_fn(self, x):
        ## Normalizing
        if self.normalize:
            x = (x - self.mean) / self.std
        return x

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.scatter_storage:
            path = self.data[idx]
            if self.spatial_dim == 1:
                data = h5py.File(path, 'r')['data'][::self.downsample[0], ::self.downsample_t, :]
            elif self.spatial_dim == 2:
                data = h5py.File(path, 'r')['data'][::self.downsample[0], ::self.downsample[1], ::self.downsample_t, :]
            elif self.spatial_dim == 3:
                data = h5py.File(path, 'r')['data'][::self.downsample[0], ::self.downsample[1], ::self.downsample[2], ::self.downsample_t, :]
            data = torch.from_numpy(data).to(dtype=torch.float32)
        else:
            data = torch.from_numpy(self.data[idx]).to(dtype=torch.float32)
        data = self.transform_fn(data)

        # Data spliting
        if self.train:
            # T = data.shape[-2]
            # bos = np.random.randint(max(T-(self.in_seq + self.pred_seq)+1, 1))
            # eos = bos+self.in_seq
            T = data.shape[-2]
            bos = 0
            eos = self.in_seq
        else:
            bos = 0
            eos = self.in_seq

        x = data[..., bos:eos, :]
        y = data[..., eos:eos + self.pred_seq, :]
        coord = self.grid.clone()

        return {'idx_lb': idx, 'x_lb': x, 'y_lb': y, 'coord': coord}
