import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# %%
class ManualScheduledHook():
    def __init__(self, 
                 max_train_ter, warmup_iter, max_seq_len, 
                 strategy='train', decay='exponential', 
                 linear_slope_f=1.0, exp_radix_f=0.9999, sigmoid_k_f=3000, 
                 linear_slope_g=1.0, exp_radix_g=0.9, sigmoid_k_g=5, 
                 ):
        self.max_train_iter = max_train_ter
        self.warmup_iter = warmup_iter
        self.max_seq_len = max_seq_len

        self.linear_slope_f = linear_slope_f
        self.exp_radix_f = exp_radix_f
        self.sigmoid_k_f = sigmoid_k_f
        self.linear_slope_g = linear_slope_g
        self.exp_radix_g = exp_radix_g
        self.sigmoid_k_g = sigmoid_k_g

        self.strategy = strategy
        self.decay = decay

        self.eps = 1e-12

    def get_f_i(self, curr_iter):
        if self.decay == 'linear':
            threshold = torch.max(self.eps, 1. - (curr_iter / self.max_train_iter) * self.linear_slope_f)
        elif self.decay == 'exponential':
            threshold = self.exp_radix_f ** curr_iter
        elif self.decay == 'sigmoid':
            threshold = self.sigmoid_k_f / (self.sigmoid_k_f + torch.exp(curr_iter / self.sigmoid_k_f))
        return threshold

    def get_g_i(self, curr_seq):
        if self.decay == 'linear':
            threshold = torch.max(self.eps, 1. - (curr_seq / self.max_seq_len) * self.linear_slope_g)
        elif self.decay == 'exponential':
            threshold = self.exp_radix_g ** curr_seq
        elif self.decay == 'sigmoid':
            threshold = self.sigmoid_k_g / (self.sigmoid_k_g + torch.exp(curr_seq / self.sigmoid_k_g))
        return threshold

    def sample(self, curr_iter, curr_seq, u_target, u_pred):
        batch_size = u_target.shape[0]

        if self.strategy == 'train':
            threshold = self.get_f_i(curr_iter)

        elif self.strategy == 'decode':
            threshold = self.get_g_i(curr_seq)

        elif self.strategy == 'composite':
            threshold = self.get_f_i(curr_iter)
            threshold = self.get_g_i(curr_seq * (1. - threshold))

        random_prob = torch.randn([batch_size])
        pos_mask = torch.where(random_prob < threshold, torch.ones_like(random_prob), torch.zeros_like(random_prob)).bool()

        next_input = u_pred.clone()
        next_input[pos_mask] = u_target[pos_mask]
        return next_input


# %%
class CurriculumScheduledHook():
    def __init__(self, total_epoch, warmup_epoch, seq_len):
        self.total_epoch = total_epoch
        self.warmup_epoch = warmup_epoch
        self.seq_len = seq_len

    def sample(self, curr_epoch, curr_seq, u_target, u_pred):
        ep_norm = (curr_epoch - self.warmup_epoch) / (self.total_epoch - self.warmup_epoch)
        t_norm = 0.5 * (1. + math.tanh( (ep_norm - 0.5) / 0.2))
        seq_ar = int(self.seq_len * t_norm)
        seq_ar = min(self.seq_len, max(1, seq_ar))

        if curr_seq < seq_ar:
            next_u = u_pred.clone()
        else:
            next_u = u_target.clone()

        return next_u

