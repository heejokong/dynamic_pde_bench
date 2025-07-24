import numpy as np
import torch
from operators.algorithms.utils import Add_Argument, str2bool
from operators.core.algorithmbase import AlgorithmBase
from operators.core.utils import ALGORITHMS
from operators.core.utils import get_optimizer, get_scheduler

from .utils import ManualScheduledHook, CurriculumScheduledHook


@ALGORITHMS.register('standard_mse')
class MSE_temporal(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        self.init(T_bundle=args.T_bundle, noise_scale=args.noise_scale)

    def init(self, T_bundle, noise_scale):
        self.T_bundle = T_bundle
        self.noise_scale = noise_scale
        self.use_sampler = self.args.use_sampler
        if self.args.use_sampler:
            if self.args.sampler_type == 'curriculum':
                self.sampler = CurriculumScheduledHook(self.args.epochs, self.args.warmup_epochs, self.args.T_ar)
            else:
                warmup_iter = int(self.args.warmup_epochs * self.args.num_eval_iter)
                self.sampler = ManualScheduledHook(
                    self.args.num_train_iter, warmup_iter, self.args.T_ar, strategy=self.args.sampler_type, decay=self.args.decay_type
                    )
        else:
            self.sampler = None

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
                loss_idx = self.criterion(y_hat, y)
                loss_mse += loss_idx

                # 
                if self.sampler is not None:
                    y_next = self.sampler.sample(self.it, t, y, y_hat)
                    x_lb = torch.cat((x_lb[..., self.T_bundle:, :], y_next.detach()), dim=-2)

                else:
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
            Add_Argument('--use_sampler', str2bool, 'false'),
            Add_Argument('--sampler_type', str, 'train'), # 4 types of sampling strategies ['train', 'decode', 'composite', 'curriculum']
            Add_Argument('--decay_type', str, 'exponential'), # 3 types of sampling strategies ['linear', 'exponential', 'sigmoid']
        ]
