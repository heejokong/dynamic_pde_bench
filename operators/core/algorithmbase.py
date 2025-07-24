import os
import contextlib
import numpy as np
from inspect import signature
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from operators.core.hooks import Hook, CheckpointHook, EvaluationHook, LoggingHook, ParamUpdateHook, \
    get_priority, DistSamplerSeedHook, TimerHook
from operators.core.utils import get_dataset, get_data_loader, get_optimizer, get_scheduler, Bn_Controller
from operators.core.criterions import SimpleLpLoss


class AlgorithmBase:
    """
        Base class for algorithms
        init algorithm specific parameters and common parameters
        
        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
    """

    def __init__(
            self,
            args,
            net_builder,
            tb_log=None,
            logger=None,
            **kwargs):

        # common arguments
        self.tb_dict = None
        self.args = args

        self.epochs = args.epochs
        self.num_train_iter = args.num_train_iter
        self.num_eval_iter = args.num_eval_iter
        self.num_log_iter = args.num_log_iter
        self.num_iter_per_epoch = int(self.num_train_iter // self.epochs)

        self.use_amp = args.use_amp
        self.clip_grad = args.clip_grad
        self.save_name = args.save_name
        self.save_dir = args.save_dir
        self.resume = args.resume
        self.algorithm = args.algorithm

        # commaon utils arguments
        self.tb_log = tb_log
        self.print_fn = print if logger is None else logger.info
        self.ngpus_per_node = torch.cuda.device_count()
        self.loss_scaler = GradScaler()
        self.amp_cm = autocast if self.use_amp else contextlib.nullcontext
        self.gpu = args.gpu
        self.rank = args.rank
        self.distributed = args.distributed
        self.world_size = args.world_size

        # common model related parameters
        self.epoch = 0
        self.it = 0
        self.best_eval_l2_step, self.best_eval_l2_full, self.best_it = 10.0, 10.0, 0
        self.bn_controller = Bn_Controller()
        self.net_builder = net_builder

        # build dataset
        self.dataset_dict = self.set_dataset()

        # build data loader
        self.loader_dict = self.set_data_loader()

        # cv, nlp, speech builder different arguments
        self.model = self.set_model()

        # build optimizer and scheduler
        self.optimizer, self.scheduler = self.set_optimizer()

        # build supervised loss
        self.criterion = SimpleLpLoss(size_average=False)

        # other arguments specific to the algorithm
        # self.init(**kwargs)

        # set common hooks during training
        self._hooks = []  # record underlying hooks 
        self.hooks_dict = OrderedDict()  # actual object to be used to call hooks
        self.set_hooks()


    def init(self, **kwargs):
        """
        algorithm specific init function, to add parameters into class
        """
        raise NotImplementedError

    def set_dataset(self):
        if self.rank != 0 and self.distributed:
            torch.distributed.barrier()
        dataset_dict = get_dataset(
            self.args.dataset, self.args.dataset_type, normalize=self.args.normalize, 
            )
        self.args.trainset_len = len(dataset_dict['train'])
        self.args.testset_len = len(dataset_dict['test'])
        self.print_fn(
            "Train data number: {}, Test data number {}".format(self.args.trainset_len, self.args.testset_len))
        if self.rank == 0 and self.distributed:
            torch.distributed.barrier()
        return dataset_dict

    def set_data_loader(self):
        self.print_fn("Create train and test data loaders")
        loader_dict = {}
        loader_dict['train'] = get_data_loader(
            self.dataset_dict['train'],
            self.args.batch_size,
            num_workers=self.args.num_workers,
            data_sampler=self.args.train_sampler,
            num_epochs=self.epochs,
            num_iters=self.num_train_iter,
            distributed=self.distributed
        )
        loader_dict['test'] = get_data_loader(
            self.dataset_dict['test'],
            self.args.eval_batch_size,
            num_workers=self.args.num_workers,
            data_sampler=None,
            shuffle=False,
            drop_last=False,
        )
        self.print_fn(f'[!] data loader keys: {loader_dict.keys()}')
        return loader_dict

    def set_model(self,):
        model = self.net_builder(args=self.args)
        return model

    def set_optimizer(self):
        self.print_fn("Create optimizer and scheduler")
        optimizer = get_optimizer(
            self.model, self.args.optim, self.args.lr, beta1=self.args.beta1, beta2=self.args.beta2,
            weight_decay=self.args.weight_decay, bn_wd_skip=False
        )
        scheduler = get_scheduler(
            optimizer, self.args.lr_method, self.args.lr, self.args.epochs, 
            self.num_iter_per_epoch, self.args.warmup_epochs, self.args.step_size, self.args.step_gamma)

        return optimizer, scheduler

    def set_hooks(self):
        """
        register necessary training hooks
        """
        # parameter update hook is called inside each train_step
        self.register_hook(TimerHook(), None, "HIGHEST")
        self.register_hook(EvaluationHook(), None, "HIGHEST")
        self.register_hook(CheckpointHook(), None, "VERY_HIGH")
        self.register_hook(DistSamplerSeedHook(), None, "HIGH")
        self.register_hook(LoggingHook(), None, "LOW")

        # for hooks to be called in train_step, name it for simpler calling
        self.register_hook(ParamUpdateHook(), "ParamUpdateHook")

    def process_batch(self, **kwargs):
        """
        process batch data, send data to cuda
        NOTE **kwargs should have the same arguments to train_step function as keys to work properly
        """
        input_args = signature(self.train_step).parameters
        input_args = list(input_args.keys())
        input_dict = {}

        for arg, var in kwargs.items():
            if not arg in input_args:
                continue

            if var is None:
                continue

            # send var to cuda
            if isinstance(var, dict):
                var = {k: v.cuda(self.gpu) for k, v in var.items()}
            else:
                var = var.cuda(self.gpu)
            input_dict[arg] = var
        return input_dict

    def train_step(self, *args, **kwargs):
        """
        train_step specific to each algorithm
        """
        # implement train step for each algorithm
        # compute loss
        # update model 
        # record tb_dict
        # return tb_dict
        raise NotImplementedError

    def train(self):
        """
        train function
        """
        self.model.train()
        self.call_hook("before_run")

        for epoch in range(self.epoch, self.epochs):
            self.epoch = epoch

            # prevent the training iterations exceed args.num_train_iter
            if self.it >= self.num_train_iter:
                break

            self.call_hook("before_train_epoch")

            for data_lb in self.loader_dict['train']:
                # prevent the training iterations exceed args.num_train_iter
                if self.it >= self.num_train_iter:
                    break

                self.call_hook("before_train_step")
                self.tb_dict = self.train_step(**self.process_batch(**data_lb))
                self.call_hook("after_train_step")
                self.it += 1

            self.call_hook("after_train_epoch")
            if self.epoch % 100 == 0 and self.epoch != 0:
                save_path = os.path.join(self.save_dir, self.save_name)
                self.save_model(f'model_checkpoint_{self.epoch}.pth', save_path)

        self.call_hook("after_run")

    def evaluate(self, eval_dest='test'):
        """
        evaluation function
        """
        self.model.eval()
        eval_loader = self.loader_dict[eval_dest]

        total_num = 0.
        l2_steps = 0.
        l2_fulls = 0.
        with torch.no_grad():
            for data in eval_loader:
                loss = 0.

                x = data['x_lb']
                y = data['y_lb']

                if isinstance(x, dict):
                    _x = {k: v.cuda(self.gpu) for k, v in x.items()}
                else:
                    _x = x.cuda(self.gpu)
                y = y.cuda(self.gpu)

                num_batch = y.shape[0]
                total_num += num_batch

                for t in range(0, y.shape[-2], self.args.T_bundle):
                    _y = y[..., t:t + self.args.T_bundle, :]
                    y_hat = self.model(_x)
                    loss += self.criterion(y_hat, _y)

                    if t == 0:
                        pred = y_hat
                    else:
                        pred = torch.cat((pred, y_hat), -2)

                    _x = torch.cat((_x[..., self.args.T_bundle:,:], y_hat), dim=-2)

                l2_steps += loss
                l2_fulls += self.criterion(pred, y)

        l2_steps = l2_steps / total_num / (y.shape[-2] / self.args.T_bundle)
        l2_fulls = l2_fulls / total_num

        self.model.train()

        eval_dict = {eval_dest + '/loss_step': l2_steps.item(), eval_dest + '/loss_full': l2_fulls.item()}
        return eval_dict

    def get_save_dict(self):
        """
        make easier for saving model when need save additional arguments
        """
        # base arguments for all models
        save_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'loss_scaler': self.loss_scaler.state_dict(),
            'epoch': self.epoch + 1,
            'it': self.it + 1,
            'best_it': self.best_it,
            'best_eval_l2_step': self.best_eval_l2_step,
            'best_eval_l2_full': self.best_eval_l2_full,
        }
        return save_dict

    def save_model(self, save_name, save_path):
        """
        save model and specified parameters for resume
        """
        save_filename = os.path.join(save_path, save_name)
        save_dict = self.get_save_dict()
        torch.save(save_dict, save_filename)
        self.print_fn(f"model saved: {save_filename}")

    def load_model(self, load_path):
        """
        load model and specified parameters for resume
        """
        checkpoint = torch.load(load_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.loss_scaler.load_state_dict(checkpoint['loss_scaler'])
        self.epoch = checkpoint['epoch']
        self.it = checkpoint['it']
        self.best_it = checkpoint['best_it']
        self.best_eval_l2_step = checkpoint['best_eval_l2_step']
        self.best_eval_l2_full = checkpoint['best_eval_l2_full']
        self.print_fn('model loaded')
        return checkpoint

    def check_prefix_state_dict(self, state_dict):
        """
        remove prefix state dict in ema model
        """
        new_state_dict = dict()
        for key, item in state_dict.items():
            if key.startswith('module'):
                new_key = '.'.join(key.split('.')[1:])
            else:
                new_key = key
            new_state_dict[new_key] = item
        return new_state_dict

    def process_out_dict(self, out_dict=None, **kwargs):
        """
        process the out_dict as return of train_step
        """
        if out_dict is None:
            out_dict = {}

        for arg, var in kwargs.items():
            out_dict[arg] = var
        
        # process res_dict, add output from res_dict to out_dict if necessary
        return out_dict

    def process_log_dict(self, log_dict=None, prefix='train', **kwargs):
        """
        process the tb_dict as return of train_step
        """
        if log_dict is None:
            log_dict = {}

        for arg, var in kwargs.items():
            log_dict[f'{prefix}/' + arg] = var
        return log_dict

    def register_hook(self, hook, name=None, priority='LOWEST'):
        """
        Ref: https://github.com/open-mmlab/mmcv/blob/a08517790d26f8761910cac47ce8098faac7b627/mmcv/runner/base_runner.py#L263
        Register a hook into the hook list.
        The hook will be inserted into a priority queue, with the specified
        priority (See :class:`Priority` for details of priorities).
        For hooks with the same priority, they will be triggered in the same
        order as they are registered.
        Args:
            hook (:obj:`Hook`): The hook to be registered.
            name (:str, default to None): Name of the hook to be registered. Default is the hook class name.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
        """
        assert isinstance(hook, Hook)
        if hasattr(hook, 'priority'):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority  # type: ignore
        hook.name = name if name is not None else type(hook).__name__

        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:  # type: ignore
                self._hooks.insert(i + 1, hook)
                inserted = True
                break

        if not inserted:
            self._hooks.insert(0, hook)

        # call set hooks
        self.hooks_dict = OrderedDict()
        for hook in self._hooks:
            self.hooks_dict[hook.name] = hook

    def call_hook(self, fn_name, hook_name=None, *args, **kwargs):
        """Call all hooks.
        Args:
            fn_name (str): The function name in each hook to be called, such as
                "before_train_epoch".
            hook_name (str): The specific hook name to be called, such as
                "param_update" or "dist_align", uesed to call single hook in train_step.
        """

        if hook_name is not None:
            return getattr(self.hooks_dict[hook_name], fn_name)(self, *args, **kwargs)

        for hook in self.hooks_dict.values():
            if hasattr(hook, fn_name):
                getattr(hook, fn_name)(self, *args, **kwargs)

    def registered_hook(self, hook_name):
        """
        Check if a hook is registered
        """
        return hook_name in self.hooks_dict

    @staticmethod
    def get_argument():
        """
        Get specificed arguments into argparse for each algorithm
        """
        return {}
