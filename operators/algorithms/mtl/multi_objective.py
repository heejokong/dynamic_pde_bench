import numpy as np
import torch
from operators.algorithms.utils import Add_Argument, str2bool
from operators.core.algorithmbase import AlgorithmBase
from operators.core.utils import ALGORITHMS
from operators.core.utils import get_optimizer, get_scheduler

from .utils import PCGrad, CAGrad


@ALGORITHMS.register('mtl')
class MTL_temporal(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        self.init(T_bundle=args.T_bundle, noise_scale=args.noise_scale)

    def init(self, T_bundle, noise_scale):
        self.T_bundle = T_bundle
        self.noise_scale = noise_scale

    def set_optimizer(self):
        self.print_fn("Create optimizer and scheduler")
        optimizer = get_optimizer(
            self.model, self.args.optim, self.args.lr, beta1=self.args.beta1, beta2=self.args.beta2,
            weight_decay=self.args.weight_decay, bn_wd_skip=False
        )
        self.use_proj_grad = self.args.use_proj_grad
        if self.use_proj_grad:
            if self.args.proj_type == 'pcgrad':
                optimizer = PCGrad(
                    base_optimizer=optimizer,
                )
            elif self.args.proj_type == 'cagrad':
                optimizer = CAGrad(
                    base_optimizer=optimizer,
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

        x_lb_tf = x_lb.clone()
        x_lb_ar = x_lb.clone()

        # inference and calculate auto-regressive losses
        with self.amp_cm():
            loss_tf = 0.
            loss_ar = 0.
            for t in range(0, T_ar, self.T_bundle):
                y = y_lb[..., t:t + self.T_bundle, :]

                ### (1) COMPUTE TEACHER-FORCING OBJECTIVE ### 
                ar_noise = self.noise_scale * torch.sum(x_lb_tf**2, dim=(1,2,3), keepdim=True) ** 0.5 * torch.randn_like(x_lb_tf)
                x_lb_tf = x_lb_tf + ar_noise

                y_hat_tf = self.model(x_lb_tf)
                loss_tf_idx = self.criterion(y_hat_tf, y)
                loss_tf += loss_tf_idx

                ### (2) COMPUTE AUTOREGRESSION OBJECTIVE ### 
                ar_noise = self.noise_scale * torch.sum(x_lb_ar**2, dim=(1,2,3), keepdim=True) ** 0.5 * torch.randn_like(x_lb_ar)
                x_lb_ar = x_lb_ar + ar_noise

                y_hat_ar = self.model(x_lb_ar)
                loss_ar_idx = self.criterion(y_hat_ar, y)
                loss_ar += loss_ar_idx

                ### (3) UPDATE INPUT & OUTPUT SIGNALS ###
                x_lb_tf = torch.cat((x_lb_tf[..., self.T_bundle:, :], y), dim=-2)
                x_lb_ar = torch.cat((x_lb_ar[..., self.T_bundle:, :], y_hat_ar), dim=-2)

                # logging
                if t == 0:
                    pred = y_hat_tf.detach()
                else:
                    pred = torch.cat((pred, y_hat_tf), dim=-2)

            l2_steps = loss_tf / self.args.batch_size / (T_ar / self.T_bundle)
            l2_fulls = self.criterion(pred, y_lb) / self.args.batch_size

        ### REMOVE CONFLICT GRADIENTS ###
        if self.use_proj_grad:
            self.optimizer.update_grads(loss_tf, loss_ar)

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
            loss = loss_tf + loss_ar
            self.call_hook("param_update", "ParamUpdateHook", loss=loss)

        tb_dict = {'train/loss_step': l2_steps.item(), 'train/loss_full': l2_fulls.item(),}
        return tb_dict

    @staticmethod
    def get_argument():
        return [
            Add_Argument('--use_proj_grad', str2bool, 'false'),
            Add_Argument('--proj_type', str, 'pcgrad'),
        ]
