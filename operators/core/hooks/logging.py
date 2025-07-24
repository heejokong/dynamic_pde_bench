from .hook import Hook


class LoggingHook(Hook):
    def __init__(self) -> None:
        super().__init__()

    def after_train_step(self, algorithm, float_tick=8):
        log_dict = {}
        for arg, var in algorithm.tb_dict.items():
            log_dict[arg] = round(var, float_tick)

        """must be called after evaluation"""
        if self.every_n_iters(algorithm, algorithm.num_eval_iter):
            if not algorithm.distributed or (algorithm.distributed and algorithm.rank % algorithm.ngpus_per_node == 0):
                log_str = f"{algorithm.it + 1} iteration, {log_dict}, BEST_EVAL_L2_FULL: {algorithm.best_eval_l2_full:.8f}, " \
                    + f"BEST_EVAL_L2_STEP: {algorithm.best_eval_l2_step:.8f}, at {algorithm.best_it + 1} iters"
                algorithm.print_fn(log_str)

            if algorithm.tb_log is not None:
                algorithm.tb_log.update(algorithm.tb_dict, algorithm.it)

        elif self.every_n_iters(algorithm, algorithm.num_log_iter):
            if not algorithm.distributed or (algorithm.distributed and algorithm.rank % algorithm.ngpus_per_node == 0):
                algorithm.print_fn(f"{algorithm.it + 1} iteration, {log_dict}")
