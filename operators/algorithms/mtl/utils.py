import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize


# %%
class PCGrad(torch.optim.Optimizer):
    def __init__(self, base_optimizer, reduction='mean', **kwargs):

        defaults = dict(**kwargs)
        super(PCGrad, self).__init__(base_optimizer.param_groups, defaults)

        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

        self._reduction = reduction

    def _project_conflict(self, grad_1, grad_2, has_grad):
        shared = has_grad.bool()

        _grad_1 = copy.deepcopy(grad_1)
        _grad_2 = copy.deepcopy(grad_2)

        dot_grad = torch.vdot(_grad_1, _grad_2)
        if dot_grad.real < 0:
            _grad_1 -= (dot_grad) * grad_2 / (grad_2.norm()**2)
            _grad_2 -= (dot_grad) * grad_1 / (grad_1.norm()**2)
        pc_grads = [_grad_1, _grad_2]

        merged_grad = torch.zeros_like(pc_grads[0]).to(pc_grads[0].device)
        if self._reduction == 'mean':
            merged_grad[shared] = torch.stack([g[shared] for g in pc_grads]).mean(dim=0)
        elif self._reduction == 'sum':
            merged_grad[shared] = torch.stack([g[shared] for g in pc_grads]).sum(dim=0)
        else: exit('invalid reduction method')
        merged_grad[~shared] = torch.stack([g[~shared] for g in pc_grads]).sum(dim=0)

        return merged_grad

    def _get_gradient(self, loss):
        self.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        grad, shape, has_grad = [], [], []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad

    def _set_grad(self, grads):
        idx = 0
        for group in self.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                p.grad = grads[idx].to(p.grad.dtype)
                idx += 1

    @torch.no_grad()
    def step(self,):
        self.base_optimizer.step()

    @torch.no_grad()
    def update_grads(self, loss_tf, loss_ar, zero_grad=False):
        # (1) Capture AR & TF Gradients
        ar_grad, _ , _= self._get_gradient(loss_ar)
        tf_grad, grad_shape, has_grad = self._get_gradient(loss_tf)
        ar_grad = self._flatten_grad(ar_grad)
        tf_grad = self._flatten_grad(tf_grad)
        has_grad = self._flatten_grad(has_grad)

        pc_grads = self._project_conflict(tf_grad, ar_grad, has_grad)
        pc_grads = self._unflatten_grad(pc_grads, grad_shape)
        self._set_grad(pc_grads)

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


# %%
class CAGrad(torch.optim.Optimizer):
    def __init__(self, base_optimizer, calpha=0.5, rescale=1, reduction='mean', **kwargs):

        defaults = dict(**kwargs)
        super(CAGrad, self).__init__(base_optimizer.param_groups, defaults)

        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

        self.calpha = calpha
        self.rescale = rescale
        self._reduction = reduction

    def _project_conflict(self, grad_1, grad_2, has_grad):
        grads = torch.stack([grad_1, grad_2])
        self.num_loss = 2
        # 
        GG = torch.matmul(grads, grads.t()).cpu() # [num_tasks, num_tasks]
        g0_norm = (GG.mean()+1e-8).sqrt() # norm of the average gradient
        x_start = np.ones(self.num_loss) / self.num_loss
        bnds = tuple((0,1) for x in x_start)
        cons=({'type':'eq','fun':lambda x:1-sum(x)})
        A = GG.numpy()
        b = x_start.copy()
        c = (self.calpha*g0_norm+1e-8).item()
        def objfn(x):
            return (x.reshape(1,-1).dot(A).dot(b.reshape(-1,1))+c*np.sqrt(x.reshape(1,-1).dot(A).dot(x.reshape(-1,1))+1e-8)).sum()
        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w_cpu = res.x
        ww = torch.Tensor(w_cpu).to(grads.device)
        gw = (grads * ww.view(-1, 1)).sum(0)
        gw_norm = gw.norm()
        lmbda = c / (gw_norm+1e-8)
        g = grads.mean(0) + lmbda * gw
        if self.rescale == 0:
            new_grads = g
        elif self.rescale == 1:
            new_grads = g / (1 + self.calpha ** 2)
        elif self.rescale == 2:
            new_grads = g / (1 + self.calpha)
        else:
            raise ValueError('No support rescale type {}'.format(self.rescale))
        return new_grads

    def _get_gradient(self, loss):
        self.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        grad, shape, has_grad = [], [], []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad

    def _set_grad(self, grads):
        idx = 0
        for group in self.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                p.grad = grads[idx].to(p.grad.dtype)
                idx += 1

    @torch.no_grad()
    def step(self,):
        self.base_optimizer.step()

    @torch.no_grad()
    def update_grads(self, loss_tf, loss_ar, zero_grad=False):
        # (1) Capture AR & TF Gradients
        ar_grad, _ , _= self._get_gradient(loss_ar)
        tf_grad, grad_shape, has_grad = self._get_gradient(loss_tf)
        ar_grad = self._flatten_grad(ar_grad)
        tf_grad = self._flatten_grad(tf_grad)
        has_grad = self._flatten_grad(has_grad)

        ca_grads = self._project_conflict(tf_grad, ar_grad, has_grad)
        ca_grads = self._unflatten_grad(ca_grads, grad_shape)
        self._set_grad(ca_grads)

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
