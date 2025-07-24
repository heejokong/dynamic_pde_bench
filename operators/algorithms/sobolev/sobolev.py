import numpy as np
import torch
from operators.algorithms.utils import Add_Argument, str2bool
from operators.core.algorithmbase import AlgorithmBase
from operators.core.utils import ALGORITHMS
from operators.core.utils import get_optimizer, get_scheduler

from .utils import HsLoss


@ALGORITHMS.register('sobolev')
class Sobolev_temporal(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        self.init(T_bundle=args.T_bundle, noise_scale=args.noise_scale)

    def init(self, T_bundle, noise_scale):
        self.T_bundle = T_bundle
        self.noise_scale = noise_scale
        self.sobolev_criterion = HsLoss(
            k=self.args.loss_k, group=self.args.loss_group, size_average=False
            )

    def train_step(self, x_lb, y_lb):
        T_ar = y_lb.shape[-2]

        # inference and calculate auto-regressive losses
        with self.amp_cm():
            loss_mse = 0.
            for t in range(0, T_ar, self.T_bundle):
                y = y_lb[..., t:t + self.T_bundle, :]

                # 
                ar_noise = self.noise_scale * torch.sum(x_lb**2, dim=(1,2,3), keepdim=True) ** 0.5 * torch.randn_like(x_lb)
                x_lb = x_lb + ar_noise

                # 
                y_hat = self.model(x_lb)
                loss_idx = self.sobolev_criterion(y_hat, y)
                loss_mse += loss_idx

                # 
                if self.args.use_tf:
                    x_lb = torch.cat((x_lb[..., self.T_bundle:, :], y), dim=-2)
                else:
                    if self.args.use_pushforward:
                        y_hat = y_hat.detach()
                    x_lb = torch.cat((x_lb[..., self.T_bundle:, :], y_hat), dim=-2)

                # logging
                if t == 0:
                    pred = y_hat.detach()
                else:
                    pred = torch.cat((pred, y_hat), dim=-2)

            l2_steps = loss_mse / self.args.batch_size / (T_ar / self.T_bundle)
            l2_fulls = self.criterion(pred, y_lb) / self.args.batch_size

        # 
        self.call_hook("param_update", "ParamUpdateHook", loss=loss_mse)
        tb_dict = {'train/loss_step': l2_steps.item(), 'train/loss_full': l2_fulls.item(),}
        return tb_dict

    @staticmethod
    def get_argument():
        return [
            Add_Argument('--use_tf', str2bool, 'false'), # True: Teacher-forcing, False: Autoregression
            Add_Argument('--use_pushforward', str2bool, 'false'),
            Add_Argument('--loss_group', str2bool, 'true'), 
            # Add_Argument('--loss_k', float, 0), # H0 Sobolev loss = L2 loss
            Add_Argument('--loss_k', int, 0), # H0 Sobolev loss = L2 loss
        ]
