import copy
import math
import numpy as np
import torch

# %%
class DCM(torch.optim.Optimizer):
    def __init__(self, base_optimizer, m=0.5, warmup_step=0, warmup_p=1.0, use_inter=True, use_intra=True, **kwargs):

        defaults = dict(**kwargs)
        super(DCM, self).__init__(base_optimizer.param_groups, defaults)

        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

        self.m = m
        self.use_inter = use_inter
        self.use_intra = use_intra
        self.warmup_step = warmup_step
        self.warmup_p = warmup_p

    def _project_conflict(self, tgt_grad, ref_grad):
        _tgt_grad = copy.deepcopy(tgt_grad)
        dot_grad = torch.vdot(_tgt_grad, ref_grad)
        if dot_grad.real < 0:
            _tgt_grad -= (dot_grad) * ref_grad / (ref_grad.norm() ** 2)
        merged_grad = _tgt_grad.clone()
        return merged_grad

    def _get_gradient(self, loss):
        self.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        grad, shape = [], []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
        return grad, shape

    @torch.no_grad()
    def step_sequence(self, curr_step, curr_seq, loss_tf, loss_ar, zero_grad=False):
        # (1) Capture AR & TF Gradients
        ar_grad, _ = self._get_gradient(loss_ar)
        tf_grad, grad_shape = self._get_gradient(loss_tf)
        ar_grad = self._flatten_grad(ar_grad)
        tf_grad = self._flatten_grad(tf_grad)

        # (2)
        if curr_seq != 0:
            if self.use_inter:
                tf_grad = self._project_conflict(tf_grad, ar_grad)

        # (3)
        if curr_seq == 0:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None: continue
                    self.state[p]['ema_grad'] = p.grad.clone()
        else:
            ema_grad = []
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        ema_grad.append(torch.zeros_like(p).to(p.device))
                        continue
                    e_g = self.m * self.state[p]['ema_grad'] + (1-self.m) * p.grad
                    ema_grad.append(e_g)
                    self.state[p]['ema_grad'] = e_g.clone() # update EMA gradients
            ema_grad = self._flatten_grad(ema_grad)

        # (4)
        if curr_seq != 0:
            if self.use_intra:
                tf_grad = self._project_conflict(tf_grad, ema_grad)
        tf_grad = self._unflatten_grad(tf_grad, grad_shape)

        # (5)
        _coef = 1.0
        if self.warmup_step != 0:
            _coef = np.clip((curr_step / self.warmup_step) ** self.warmup_p, a_min=0., a_max=1.)

        # (6)
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                _g = (1. - _coef) * p.grad + _coef * tf_grad[idx].to(p.grad.dtype)
                p.grad = _g.clone()
                idx += 1
                if curr_seq == 0:
                    self.state[p]['agg_grad'] = p.grad.clone()
                else:
                    self.state[p]['agg_grad'].add_(p.grad.clone())
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self,):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.grad = self.state[p]['agg_grad'].clone()
        self.base_optimizer.step()

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

    def _flatten_grad(self, grads):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad
