import os
import math
import logging
import random
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader

from operators.datasets import TemporalGridDataset, SteadyGridDataset
from operators.datasets.utils import name2sampler


def get_net_builder(net_name,):
    """
    built network according to network name
    return **class** of backbone network (not instance).
    """
    import operators.nets as nets
    builder = getattr(nets, net_name)
    return builder


def get_logger(name, save_path=None, level='INFO'):
    """
    create logger function
    """
    logger = logging.getLogger(name)
    logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s', level=getattr(logging, level))

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        log_format = logging.Formatter('[%(asctime)s %(levelname)s] %(message)s')
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger


def get_dataset(dataset, dataset_type, normalize=False, y_normalize=False):
    """
    create dataset

    Args
        dataset: dataset name 
        dataset_type: type of PDE datasets, ['temporal', 'steady']
        normalize: 
        random_query: 
        num_query: 
    """
    if dataset_type == 'temporal':
        dset_train = TemporalGridDataset(dataset, normalize=normalize, train=True)
        dset_test  = TemporalGridDataset(dataset, normalize=normalize, train=False)

    elif dataset_type == 'steady':
        dset_train = SteadyGridDataset(dataset, x_normalize=normalize, y_normalize=y_normalize, train=True)
        dset_test  = SteadyGridDataset(dataset, x_normalize=normalize, y_normalize=y_normalize, train=False)

    else:
        raise NotImplementedError

    dataset_dict = {'train': dset_train, 'test': dset_test}
    return dataset_dict


def get_data_loader(
        dset, batch_size=None, shuffle=False, num_workers=4, pin_memory=False, data_sampler='RandomSampler',
        num_epochs=None, num_iters=None, generator=None, drop_last=True, distributed=False):
    """
    get_data_loader returns torch.utils.data.DataLoader for a Dataset.
    All arguments are comparable with those of pytorch DataLoader.
    However, if distributed, DistributedProxySampler, which is a wrapper of data_sampler, is used.
    
    Args
        num_epochs: total batch -> (# of batches in dset) * num_epochs 
        num_iters: total batch -> num_iters
    """
    assert batch_size is not None

    if data_sampler is None:
        data_loader = DataLoader(
            dset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, 
            drop_last=drop_last, pin_memory=pin_memory)

    elif isinstance(data_sampler, str):
        data_sampler = name2sampler[data_sampler]

        if distributed:
            assert dist.is_available()
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()
        else:
            num_replicas = 1
            rank = 0

        per_epoch_steps = num_iters // num_epochs
        num_samples = per_epoch_steps * batch_size * num_replicas

        data_loader = DataLoader(
            dset, batch_size=batch_size, shuffle=False, num_workers=num_workers, 
            drop_last=drop_last, pin_memory=pin_memory, generator=generator,
            sampler=data_sampler(dset, num_replicas=num_replicas, rank=rank, num_samples=num_samples))

    elif isinstance(data_sampler, torch.utils.data.Sampler):
        data_loader = DataLoader(
            dset, batch_size=batch_size, shuffle=False, num_workers=num_workers, 
            drop_last=drop_last, pin_memory=pin_memory,
            sampler=data_sampler, generator=generator)

    else:
        raise Exception(f"unknown data sampler {data_sampler}.")

    return data_loader


def get_optimizer(model, optim_name='adam', lr=1e-3, beta1=0.9, beta2=0.999, weight_decay=0., bn_wd_skip=True):
    '''
    return optimizer (name) in torch.optim.
    If bn_wd_skip, the optimizer does not apply
    weight decay regularization on parameters in batch normalization.
    '''
    no_decay = {}
    if hasattr(model, 'no_weight_decay') and bn_wd_skip:
        no_decay = model.no_weight_decay()

    def param_groups_weight_decay(
            model: nn.Module,
            weight_decay=1e-5,
            no_weight_decay_list=()
    ):
        # Ref: https://github.com/rwightman/pytorch-image-models
        no_weight_decay_list = set(no_weight_decay_list)
        decay = []
        no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay_list:
                no_decay.append(param)
            else:
                decay.append(param)

        return [
            {'params': no_decay, 'weight_decay': 0.},
            {'params': decay, 'weight_decay': weight_decay}]

    per_param_args = param_groups_weight_decay(model, weight_decay, no_weight_decay_list=no_decay)

    if optim_name == 'adam':
        optimizer = torch.optim.Adam(per_param_args, lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)

    elif optim_name == 'adamw':
        optimizer = torch.optim.AdamW(per_param_args, lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)

    return optimizer


def get_scheduler(
        optimizer, lr_method, lr, epochs, steps_per_epoch,
        warmup_epochs, step_size=1024, step_gamma=0.1):
    from torch.optim.lr_scheduler import OneCycleLR, StepLR, LambdaLR

    def get_cosine_schedule_with_warmup(optimizer,
                                        num_training_steps,
                                        num_cycles=7. / 16.,
                                        num_warmup_steps=0,
                                        last_epoch=-1):
        '''
        Get cosine scheduler (LambdaLR).
        if warmup is needed, set num_warmup_steps (int) > 0.
        '''
        def _lr_lambda(current_step):
            '''
            _lr_lambda returns a multiplicative factor given an interger parameter epochs.
            Decaying criteria: last_epoch
            '''

            if current_step < num_warmup_steps:
                _lr = float(current_step) / float(max(1, num_warmup_steps))
            else:
                num_cos_steps = float(current_step - num_warmup_steps)
                num_cos_steps = num_cos_steps / float(max(1, num_training_steps - num_warmup_steps))
                _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
            return _lr

        return LambdaLR(optimizer, _lr_lambda, last_epoch)

    if lr_method == 'cycle':
        scheduler = OneCycleLR(optimizer, max_lr=lr, div_factor=1e4, pct_start=(warmup_epochs / epochs), final_div_factor=1e4, steps_per_epoch=steps_per_epoch, epochs=epochs)
        # scheduler = OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=epochs)
    elif lr_method == 'step':
        scheduler = StepLR(optimizer, step_size=step_size, gamma=step_gamma)
    elif lr_method == 'cosine':
        num_train_iter = epochs * steps_per_epoch
        num_warmup_iter = warmup_epochs * steps_per_epoch
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_train_iter, num_warmup_steps=num_warmup_iter)
    else:
        raise NotImplementedError

    return scheduler


def get_port():
    """
    find a free port to used for distributed learning
    """
    pscmd = "netstat -ntl |grep -v Active| grep -v Proto|awk '{print $4}'|awk -F: '{print $NF}'"
    procs = os.popen(pscmd).read()
    procarr = procs.split("\n")
    tt= random.randint(15000, 30000)
    if tt not in procarr:
        return tt
    else:
        return get_port()
