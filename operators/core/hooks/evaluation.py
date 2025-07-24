import os
from .hook import Hook


class EvaluationHook(Hook):
    def __init__(self) -> None:
        super().__init__()

    def before_run(self, algorithm):
        return

    def after_train_step(self, algorithm, eval_type='test'):
        if self.every_n_iters(algorithm, algorithm.num_eval_iter) or self.is_last_iter(algorithm):
            algorithm.print_fn("validating...")
            eval_dict = algorithm.evaluate(eval_type)
            algorithm.tb_dict.update(eval_dict)

            # update best metrics
            if algorithm.tb_dict[f'{eval_type}/loss_full'] < algorithm.best_eval_l2_full:
                algorithm.best_eval_l2_full = algorithm.tb_dict[f'{eval_type}/loss_full']
                algorithm.best_eval_l2_step = algorithm.tb_dict[f'{eval_type}/loss_step']
                algorithm.best_it = algorithm.it

    def after_run(self, algorithm):
        return
