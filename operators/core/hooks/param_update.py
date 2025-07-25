import torch

from .hook import Hook


class ParamUpdateHook(Hook):
    def __init__(self) -> None:
        super().__init__()
    
    # specific param_update function, called inside train_step of each algorithm
    def param_update(self, algorithm, loss):
        # algorithm.optimizer.zero_grad()
        # update parameters
        if algorithm.use_amp:
            algorithm.loss_scaler.scale(loss).backward()
            if (algorithm.clip_grad > 0):
                algorithm.loss_scaler.unscale_(algorithm.optimizer)
                torch.nn.utils.clip_grad_norm_(algorithm.model.parameters(), algorithm.clip_grad)
            algorithm.loss_scaler.step(algorithm.optimizer)
            algorithm.loss_scaler.update()
        else:
            loss.backward()
            if (algorithm.clip_grad > 0):
                torch.nn.utils.clip_grad_norm_(algorithm.model.parameters(), algorithm.clip_grad)
            algorithm.optimizer.step()

        algorithm.scheduler.step()
        algorithm.model.zero_grad()

    # specific param_update function, called inside train_step of each algorithm
    def diverse_param_update(self, algorithm, loss):
        # algorithm.optimizer.zero_grad()
        # update parameters
        if algorithm.use_amp:
            algorithm.loss_scaler.scale(loss).backward()
            if (algorithm.clip_grad > 0):
                algorithm.loss_scaler.unscale_(algorithm.diverse_optimizer)
                torch.nn.utils.clip_grad_norm_(algorithm.diverse_model.parameters(), algorithm.clip_grad)
            algorithm.loss_scaler.step(algorithm.diverse_optimizer)
            algorithm.loss_scaler.update()
        else:
            loss.backward()
            if (algorithm.clip_grad > 0):
                torch.nn.utils.clip_grad_norm_(algorithm.diverse_model.parameters(), algorithm.clip_grad)
            algorithm.diverse_optimizer.step()

        algorithm.diverse_scheduler.step()
        algorithm.diverse_model.zero_grad()
