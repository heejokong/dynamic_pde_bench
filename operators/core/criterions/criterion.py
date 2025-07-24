import math

import torch
from torch.nn.modules.loss import _WeightedLoss


def get_loss_func(name, component, normalizer):
    if name == 'rel2':
        return RelLpLoss(p=2,component=component, normalizer=normalizer)
    elif name == "rel1":
        return RelLpLoss(p=1,component=component, normalizer=normalizer)
    elif name == 'l2':
        return LpLoss(p=2, component=component, normalizer=normalizer)
    elif name == "l1":
        return LpLoss(p=1, component=component, normalizer=normalizer)
    else:
        raise NotImplementedError


class SimpleLpLoss(_WeightedLoss):
    def __init__(self, d=2, p=2, size_average=True, reduction=True, return_comps = False):
        super(SimpleLpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average
        self.return_comps = return_comps

    def forward(self, x, y, mask=None):
        num_examples = x.size()[0]

        # Lp loss 1
        if mask is not None:##TODO: will be meaned by n_channels for single channel data
            x = x * mask
            y = y * mask

            ## compute effective channels
            # msk_channels = mask.sum(dim=(1,2,3),keepdim=False).count_nonzero(dim=-1) # B, 1
            msk_channels = mask.sum(dim=list(range(1, mask.ndim-1)),keepdim=False).count_nonzero(dim=-1) # B, 1
        else:
            msk_channels = x.shape[-1]

        diff_norms = torch.norm(x.reshape(num_examples,-1, x.shape[-1]) - y.reshape(num_examples,-1,x.shape[-1]), self.p, dim=1)    ##N, C
        y_norms = torch.norm(y.reshape(num_examples,-1, y.shape[-1]), self.p, dim=1) + 1e-8

        if self.reduction:
            if self.size_average:
                    return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(torch.sum(diff_norms/y_norms, dim=-1) / msk_channels)
        else:
            return torch.sum(diff_norms/y_norms, dim=-1) / msk_channels


class LpLoss(_WeightedLoss):
    def __init__(self, d=2, p=2, component=0, regularizer=False, normalizer=None):
        super(LpLoss, self).__init__()

        self.d = d
        self.p = p
        self.component = component if component in ['all' , 'all-reduce'] else int(component)

        self.regularizer = regularizer
        self.normalizer = normalizer

    def _lp_losses(self, pred, target):
        if self.component == 'all':
            losses = ((pred - target).view(pred.shape[0],-1,pred.shape[-1]).abs() ** self.p).mean(dim=1) ** (1 / self.p)
            metrics = losses.mean(dim=0).clone().detach().cpu().numpy()

        else:
            assert self.component <= target.shape[1]
            losses = ((pred - target).view(pred.shape[0],-1,pred.shape[-1]).abs() ** self.p).mean(dim=1) ** (1 / self.p)
            metrics = losses.mean().clone().detach().cpu().numpy()

        loss = losses.mean()

        return loss, metrics

    def forward(self, pred, target):
        #### only for computing metrics
        loss, metrics = self._lp_losses(pred, target)

        if self.normalizer is not None:
            ori_pred, ori_target = self.normalizer.transform(pred,component=self.component, inverse=True), self.normalizer.transform(target, inverse=True)
            _, metrics = self._lp_losses(ori_pred, ori_target)

        if self.regularizer:
            raise NotImplementedError
        else:
            reg = torch.zeros_like(loss)

        return loss, reg, metrics


class RelLpLoss(_WeightedLoss):
    def __init__(self, d=2, p=2, component=0, regularizer=False, normalizer=None):
        super(RelLpLoss, self).__init__()

        self.d = d
        self.p = p
        self.component = component if component in ['all' , 'all-reduce'] else int(component)
        self.regularizer = regularizer
        self.normalizer = normalizer

    ### all reduce is used in temporal cases, use only one metric for all components
    def _lp_losses(self, pred, target):
        if (self.component == 'all') or (self.component == 'all-reduce'):
            err_pool = ((pred - target).view(pred.shape[0], -1, pred.shape[-1]).abs()**self.p).sum(dim=1,keepdim=False)
            target_pool = (target.view(target.shape[0], -1, target.shape[-1]).abs()**self.p).sum(dim=1,keepdim=False)
            losses = (err_pool / target_pool)**(1/ self.p)
            if self.component == 'all':
                # metrics = losses.mean(dim=0).clone().detach().cpu().numpy()
                metrics = losses.mean(dim=0).unsqueeze(0).clone().cpu().detach().numpy()  # 1, n
            else:
                # metrics = losses.mean().clone().detach().cpu().numpy()
                metrics = losses.mean().unsqueeze(0).clone().cpu().detach().numpy()   # 1, 1

        else:
            assert self.component <= target.shape[1]
            err_pool = ((pred - target[...,self.component]).view(pred.shape[0], -1, pred.shape[-1]).abs() ** self.p).sum(dim=1,keepdim=False)
            target_pool = (target.view(target.shape[0], -1, target.shape[-1])[...,self.component].abs() ** self.p).sum(dim=1, keepdim=False)
            losses = (err_pool / target_pool)**(1/ self.p)
            # metrics = losses.mean().clone().detach().cpu().numpy()
            metrics = losses.mean().unsqueeze(0).clone().cpu().detach().numpy()

        loss = losses.mean()

        return loss, metrics

    ### pred, target: B, N1, N2..., Nm, C-> B, C
    def forward(self, pred, target):
        loss, metrics = self._lp_losses(pred, target)

        ### only for computing metrics
        if self.normalizer is not None:
            ori_pred, ori_target = self.normalizer.transform(pred,component=self.component,inverse=True), self.normalizer.transform(target, inverse=True)
            _ , metrics = self._lp_losses(ori_pred, ori_target)

        if self.regularizer:
            raise NotImplementedError
        else:
            reg = torch.zeros_like(loss)

        return loss, reg, metrics
