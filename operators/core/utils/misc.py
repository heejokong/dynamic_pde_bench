import os
import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import ruamel.yaml as yaml


def over_write_args_from_dict(args, dict):
    """
    overwrite arguments acocrding to config file
    """
    for k in dict:
        setattr(args, k, dict[k])

def over_write_args_from_file(args, yml):
    """ overwrite arguments acocrding to config file """
    if yml == '':
        return
    with open(yml, 'r', encoding='utf-8') as f:
        dic = yaml.load(f.read(), Loader=yaml.Loader)
        for k in dic:
            setattr(args, k, dic[k])

def str2bool(v):
    """
    str to bool
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_parameters(model):
    # count trainable parameters
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def send_model_cuda(args, model):
    if not torch.cuda.is_available():
        raise Exception('ONLY GPU TRAINING IS SUPPORTED')
    
    elif args.distributed:
        find_unsued_parameters = False if "osp" in args.algorithm else True
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)

            model.cuda(args.gpu)
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, broadcast_buffers=False,
                                                            find_unused_parameters=find_unsued_parameters,
                                                            device_ids=[args.gpu])
        else:
            # if arg.gpu is None, DDP will divide and allocate batch_size
            # to all available GPUs if device_ids are not set.
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model, broadcast_buffers=False,
                                                            find_unused_parameters=find_unsued_parameters)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()
    return model


class TBLog:
    """
    Construc tensorboard writer (self.writer).
    The tensorboard is saved at os.path.join(tb_dir, file_name).
    """

    def __init__(self, tb_dir, file_name, use_tensorboard=False):
        self.tb_dir = tb_dir
        self.use_tensorboard = use_tensorboard
        if self.use_tensorboard:
            self.writer = SummaryWriter(os.path.join(self.tb_dir, file_name))

    def update(self, tb_dict, it, suffix=None, mode="train"):
        """
        Args
            tb_dict: contains scalar values for updating tensorboard
            it: contains information of iteration (int).
            suffix: If not None, the update key has the suffix.
        """
        if suffix is None:
            suffix = ''
        if self.use_tensorboard:
            for key, value in tb_dict.items():
                self.writer.add_scalar(suffix + key, value, it)


class Bn_Controller:
    """
    Batch Norm controler
    """

    def __init__(self):
        """
        freeze_bn and unfreeze_bn must appear in pairs
        """
        self.backup = {}

    def freeze_bn(self, model):
        assert self.backup == {}
        for name, m in model.named_modules():
            if isinstance(m, nn.SyncBatchNorm) or isinstance(m, nn.BatchNorm2d):
                self.backup[name + '.running_mean'] = m.running_mean.data.clone()
                self.backup[name + '.running_var'] = m.running_var.data.clone()
                self.backup[name + '.num_batches_tracked'] = m.num_batches_tracked.data.clone()

    def unfreeze_bn(self, model):
        for name, m in model.named_modules():
            if isinstance(m, nn.SyncBatchNorm) or isinstance(m, nn.BatchNorm2d):
                m.running_mean.data = self.backup[name + '.running_mean']
                m.running_var.data = self.backup[name + '.running_var']
                m.num_batches_tracked.data = self.backup[name + '.num_batches_tracked']
        self.backup = {}

