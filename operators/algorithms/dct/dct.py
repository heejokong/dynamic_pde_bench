import numpy as np
import torch
from operators.algorithms.utils import Add_Argument, str2bool
from operators.core.algorithmbase import AlgorithmBase
from operators.core.utils import ALGORITHMS
from operators.core.utils import get_optimizer, get_scheduler

from .utils import DCM


@ALGORITHMS.register('dct')
class DCT_temporal(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        self.init(T_bundle=args.T_bundle, noise_scale=args.noise_scale)

    def init(self, T_bundle, noise_scale):
        self.T_bundle = T_bundle
        self.noise_scale = noise_scale
        self.sampler = None

    def set_optimizer(self):
        self.print_fn("Create optimizer and scheduler")
        optimizer = get_optimizer(
            self.model, self.args.optim, self.args.lr, beta1=self.args.beta1, beta2=self.args.beta2,
            weight_decay=self.args.weight_decay, bn_wd_skip=False
        )
        self.use_dcm = self.args.use_dcm
        if self.use_dcm:
            optimizer = DCM(
                base_optimizer=optimizer,
                m=self.args.dcm_momentum, 
                warmup_step=self.args.dcm_warmup_iter, 
                warmup_p=self.args.dcm_warmup_p, 
                use_inter=self.args.use_dcm_inter,
                use_intra=self.args.use_dcm_intra,
            )
        scheduler = get_scheduler(
            optimizer=optimizer, 
            lr_method=self.args.lr_method, 
            lr=self.args.lr, 
            epochs=self.args.epochs, 
            steps_per_epoch=self.num_iter_per_epoch,
            warmup_epochs=self.args.warmup_epochs, 
            step_size=self.args.step_size, 
            step_gamma=self.args.step_gamma,
        )
        return optimizer, scheduler

    def train_step(self, x_lb, y_lb):
        T_ar = y_lb.shape[-2]

        x_lb_ar = x_lb.clone()
        # inference and calculate auto-regressive losses
        with self.amp_cm():
            loss_log = 0.
            loss_mse = 0.
            for t in range(0, T_ar, self.T_bundle):
                y = y_lb[..., t:t + self.T_bundle, :]

                # 
                ar_noise = self.noise_scale * torch.sum(x_lb**2, dim=(1,2,3), keepdim=True) ** 0.5 * torch.randn_like(x_lb)
                x_lb = x_lb + ar_noise

                # (1)
                y_hat_tf = self.model(x_lb)
                loss_tf = self.criterion(y_hat_tf, y)

                if self.use_dcm:
                    y_hat_ar = self.model(x_lb_ar)
                    loss_ar = self.criterion(y_hat_ar, y)

                    indicator = True if t < (T_ar-1) else False
                    self.optimizer.step_sequence(self.it, t, loss_tf, loss_ar, zero_grad=indicator)

                else:
                    loss_mse += loss_tf

                # (2)
                if self.sampler is not None:
                    y_next = self.sampler.sample(self.it, t, y, y_hat_tf)
                    x_lb = torch.cat((x_lb[..., self.T_bundle:, :], y_next.detach()), dim=-2)
                else:
                    x_lb = torch.cat((x_lb[..., self.T_bundle:, :], y), dim=-2)

                if self.use_dcm:
                    x_lb_ar = torch.cat((x_lb_ar[..., self.T_bundle:, :], y_hat_ar.detach()), dim=-2)

                # logging
                loss_log += loss_tf.detach()
                if t == 0:
                    pred = y_hat_tf.detach()
                else:
                    pred = torch.cat((pred, y_hat_tf), dim=-2)

            l2_steps = loss_log / self.args.batch_size / (T_ar / self.T_bundle)
            l2_fulls = self.criterion(pred, y_lb) / self.args.batch_size

        # (3)
        if self.use_dcm:
            if self.use_amp:
                if (self.clip_grad > 0):
                    self.loss_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.loss_scaler.step(self.optimizer)
                self.loss_scaler.update()
            else:
                if (self.clip_grad > 0):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.optimizer.step()
            self.scheduler.step()
            self.model.zero_grad()

        else:
            self.call_hook("param_update", "ParamUpdateHook", loss=loss_mse)

        # 
        tb_dict = {'train/loss_step': l2_steps.item(), 'train/loss_full': l2_fulls.item(),}
        return tb_dict

    @staticmethod
    def get_argument():
        return [
            Add_Argument('--use_tf', str2bool, 'false'), # True: Teacher-forcing, False: Autoregression
            Add_Argument('--use_pushforward', str2bool, 'false'),
            Add_Argument('--use_sampler', str2bool, 'false'),
            # 
            Add_Argument('--use_dcm', str2bool, 'false'),
            Add_Argument('--dcm_momentum', float, 0.2),
            Add_Argument('--dcm_warmup_p', float, 0.5),
            Add_Argument('--dcm_warmup_iter', int, 1000),
            # 
            Add_Argument('--use_dcm_inter', str2bool, 'true'),
            Add_Argument('--use_dcm_intra', str2bool, 'true'),
        ]
