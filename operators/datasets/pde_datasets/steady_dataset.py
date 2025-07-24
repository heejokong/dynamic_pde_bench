import os
import random
import h5py
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from numpy.lib.stride_tricks import sliding_window_view
from einops import rearrange

from operators.datasets.pde_datasets.data_configs import DATASET_DICT


class SteadyGridDataset(Dataset):
    def __init__(self, data_name, x_normalize=False, y_normalize=False, train=True):
        super(SteadyGridDataset, self).__init__()
        self.train = train
        self.x_normalize = x_normalize
        self.y_normalize = y_normalize
        data_path = DATASET_DICT[data_name]['train_path'] if self.train else DATASET_DICT[data_name]['test_path']

        self.downsample = DATASET_DICT[data_name]['downsample']
        self.spatial_dim = len(DATASET_DICT[data_name]['raw_res'])
        self.raw_res = DATASET_DICT[data_name]['raw_res']
        self.in_res = DATASET_DICT[data_name]['in_res']

        #
        if DATASET_DICT[data_name]['scatter_storage']:
            x_data = []
            y_data = []
            for fname in os.listdir(data_path):
                data_idx = h5py.File(f'{data_path}/{fname}', 'r')
                x_idx = data_idx['input'][:]
                y_idx = data_idx['output'][:]
                x_data.append(np.expand_dims(x_idx, axis=0))
                y_data.append(np.expand_dims(y_idx, axis=0))
            x_data = np.concatenate(x_data, axis=0)
            y_data = np.concatenate(y_data, axis=0)
        else:
            data = h5py.File(data_path, 'r')
            x_data = data['input'][:]
            y_data = data['output'][:]

        # 
        self.x_data = self.get_downsample_data(x_data)
        self.y_data = self.get_downsample_data(y_data)
        self.grid = self.get_grid_coord()

        # 
        if self.x_normalize:
            self.x_mean = np.mean(self.x_data, axis=(0,), keepdims=True)
            self.x_std  = np.std(self.x_data, axis=(0,), keepdims=True) + 1e-8

        if self.y_normalize:
            self.y_mean = np.mean(self.y_data, axis=(0,), keepdims=True)
            self.y_std  = np.std(self.y_data, axis=(0,), keepdims=True) + 1e-8

    def get_downsample_data(self, data):
        if self.spatial_dim == 2:
            data = data[:, ::self.downsample[0], ::self.downsample[1]]
            data = data[:, :self.in_res[0], :self.in_res[1]]

        elif self.spatial_dim == 3:
            data = data[:, ::self.downsample[0], ::self.downsample[1], ::self.downsample[2]]
            data = data[:, :self.in_res[0], :self.in_res[1], :self.in_res[2]]

        return data

    def get_grid_coord(self,):
        if self.spatial_dim == 2:
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

    def transform_fn(self, x, y):
        ## Normalizing
        if self.x_normalize:
            x = (x - self.x_mean) / self.x_std

        if self.y_normalize:
            y = (y - self.y_mean) / self.y_std

        return x, y


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.x_data[idx])
        y = torch.from_numpy(self.y_data[idx])
        x, y = self.transform_fn(x, y)

        coord = self.grid.clone()

        return {'idx_lb': idx, 'x_lb': x, 'y_lb': y, 'coord': coord}
