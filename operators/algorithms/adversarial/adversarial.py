import numpy as np
import torch
import torch.nn as nn
from operators.algorithms.utils import Add_Argument, str2bool
from operators.core.algorithmbase import AlgorithmBase
from operators.core.utils import ALGORITHMS
from operators.core.utils import get_optimizer, get_scheduler

from .utils import WarmStartGradientReverseLayer


class adversarial_net(nn.Module):
    def __init__(self, base_net, in_channel, hidden_dim, grl_factor=0.1, warmup_iterations=5000):
        super(adversarial_net, self).__init__()
        self.base_net = base_net
        # 
        self.grl_layer = WarmStartGradientReverseLayer(
            alpha=1.0, lo=0.0, hi=grl_factor, max_iters=warmup_iterations, auto_step=True
        )
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        )
        self.discriminator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, return_logit=False):
        if return_logit:
            x_out, x_feat = self.base_net(x, return_feat=True)
            x_feat = self.grl_layer(x_feat)
            x_logit = self.discriminator(self.encoder(x_feat))
            return x_out, x_logit
        else:
            x_out = self.base_net(x)
            return x_out


@ALGORITHMS.register('adversarial')
class Adversarial_temporal(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        self.init(T_bundle=args.T_bundle, noise_scale=args.noise_scale)

    def init(self, T_bundle, noise_scale):
        self.T_bundle = T_bundle
        self.noise_scale = noise_scale
        self.adv_criterion = nn.BCEWithLogitsLoss(reduction='mean')

    def set_model(self,):
        model = super().set_model()
        model = adversarial_net(model, in_channel=self.args.width, hidden_dim=32)
        return model

    def train_step(self, x_lb, y_lb):
        T_ar = y_lb.shape[-2]

        x_lb_tf = x_lb.clone()
        x_lb_ar = x_lb.clone()

        # inference and calculate auto-regressive losses
        with self.amp_cm():
            loss_tf = 0.
            loss_adv = 0.
            for t in range(0, T_ar, self.T_bundle):
                y = y_lb[..., t:t + self.T_bundle, :]

                ### (1) COMPUTE TEACHER-FORCING OUTPUTS ### 
                ar_noise = self.noise_scale * torch.sum(x_lb_tf**2, dim=(1,2,3), keepdim=True) ** 0.5 * torch.randn_like(x_lb_tf)
                x_lb_tf = x_lb_tf + ar_noise

                y_hat_tf, tf_logits = self.model(x_lb_tf, return_logit=True)

                ### (2) COMPUTE AUTOREGRESSION OUTPUTS ### 
                ar_noise = self.noise_scale * torch.sum(x_lb_ar**2, dim=(1,2,3), keepdim=True) ** 0.5 * torch.randn_like(x_lb_ar)
                x_lb_ar = x_lb_ar + ar_noise

                y_hat_ar, ar_logits = self.model(x_lb_ar, return_logit=True)

                ### (3) COMPUTE TRAINING LOSSES ### 
                # RECON LOSS
                loss_tf_idx = self.criterion(y_hat_tf, y)
                loss_tf += loss_tf_idx

                # ADV LOSS
                label_tf = torch.ones_like(tf_logits).to(tf_logits.device)
                label_ar = torch.zeros_like(ar_logits).to(ar_logits.device)
                loss_adv += self.adv_criterion(tf_logits, label_tf) + self.adv_criterion(ar_logits, label_ar)

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

        loss = loss_tf + self.args.lambda_adv * loss_adv
        self.call_hook("param_update", "ParamUpdateHook", loss=loss)

        tb_dict = {'train/loss_step': l2_steps.item(), 'train/loss_full': l2_fulls.item(),}
        return tb_dict

    @staticmethod
    def get_argument():
        return [
            Add_Argument('--lambda_adv', float, 1.0),
        ]
