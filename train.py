import os
import argparse
import logging
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from operators.algorithms import get_algorithm, name2alg
from operators.core.utils import get_net_builder, get_port, get_logger, TBLog, count_parameters, \
    send_model_cuda, over_write_args_from_file, str2bool


def get_config():
    parser = argparse.ArgumentParser(description='PyTorch benchmarks for Time-dependent PDEs Training [FNO_NS2D, PDEBench, PDEArena, CFDBench]')
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--seed', default=1, type=int, help='seed for initializing training. ')

    ################################################################
    # DISTRIBUTED TRAINING
    ################################################################
    parser.add_argument('--world_size', default=1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int, help='**node rank** for distributed training')
    parser.add_argument('-du', '--dist-url', default='tcp://127.0.0.1:11111', type=str, help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--multiprocessing-distributed', type=str2bool, default=False,
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    ################################################################
    # DATASET CONFIGS
    ################################################################
    parser.add_argument('--dataset', default='ns2d', type=str,
                        choices=['ns2d_fno_1e-3', 'ns2d_fno_1e-4', 'ns2d_fno_1e-5', # FNO_NS2D
                                 'ns2d_pdb_M1_eta1e-1_zeta1e-1', 'ns2d_pdb_M1_eta1e-2_zeta1e-2', 
                                 'ns2d_pdb_M1e-1_eta1e-1_zeta1e-1', 'ns2d_pdb_M1e-1_eta1e-2_zeta1e-2', # PDEBench, NS2D
                                 'swe_pdb', 'dr_pdb', # PDEBench, Shallow_Water_Equations & Diffusion_Reactions
                                 'ns2d_pdb_M1e-1_eta1e-8_zeta1e-8_turb_512', 'ns2d_pdb_M1_eta1e-8_zeta1e-8_turb_512', # PDEBench, 2D Turbulence
                                 'ns2d_pdb_M1e-1_eta1e-8_zeta1e-8_rand_512', 'ns2d_pdb_M1_eta1e-8_zeta1e-8_rand_512', # PDEBench, (?)
                                 'ns3d_pdb_M1_rand', 'ns3d_pdb_M1e-1_rand', 'ns3d_pdb_M1_turb', # PDEBench, 3D CFD
                                 'ns2d_cond_pda', 'ns2d_pda', 'sw2d_pda', 'cfdbench', # PDEArena & CFDBench
                                 ],)
    parser.add_argument('--dataset_type', type=str, default='temporal')
    parser.add_argument('--n_channels', type=int, default=1)
    parser.add_argument('--res', type=int, default=128)
    parser.add_argument('--T_in', type=int, default=10)
    parser.add_argument('--T_ar', type=int, default=10)
    parser.add_argument('--T_bundle', type=int, default=1)
    parser.add_argument('--normalize', type=str2bool, default=False)
    parser.add_argument('--img_size', type=str, default='128, 128')

    ################################################################
    # ALGORITHM CONFIGS
    ################################################################
    parser.add_argument('--algorithm', type=str, default='fno', choices=['fno', 'dpot', 'hiermixer'])
    parser.add_argument('--noise_scale', type=float, default=0.0)
    parser.add_argument('--use_amp', type=str2bool, default=False, help='use mixed precision training or not')
    parser.add_argument('--clip_grad', type=float, default=0.1)

    ################################################################
    # MODEL CONFIGS
    ################################################################
    # COMMMON
    parser.add_argument('--width', type=int, default=20)
    parser.add_argument('--n_blocks', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--use_ln', type=str2bool, default=False)
    parser.add_argument('--act', type=str, default='gelu')

    # FNO
    parser.add_argument('--mode1', type=int, default=12)
    parser.add_argument('--mode2', type=int, default=12)
    parser.add_argument('--mode3', type=int, default=12)
    parser.add_argument('--padding', type=str, default='0,0')

    # DPOT
    parser.add_argument('--patch_size', type=int, default=8)
    parser.add_argument('--mlp_ratio', type=int, default=1)
    parser.add_argument('--modes', type=int, default=32)
    parser.add_argument('--out_layer_dim', type=int, default=32)
    parser.add_argument('--mixing_type', type=str, default='afno')
    parser.add_argument('--time_agg', type=str, default='exp_mlp')

    # Galerkin Transformer
    # parser.add_argument('--patch_size', type=int, default=8)

    ################################################################
    # TRAINING CONFIGS
    ################################################################
    # 
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--warmup_epochs',type=int, default=100)
    parser.add_argument('--num_train_iter', type=int, default=524288, help='total number of training iterations')
    parser.add_argument('--num_eval_iter', type=int, default=1024, help='evaluation frequency')
    parser.add_argument('--num_log_iter', type=int, default=128, help='logging frequency')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--eval_batch_size', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--train_sampler', type=str, default='RandomSampler')

    # Optimization
    parser.add_argument('--optim', type=str, default='adam', choices=['adam', 'adamw'])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_method', type=str, default='cycle')
    parser.add_argument('--beta1', type=float,default=0.9)
    parser.add_argument('--beta2', type=float,default=0.999)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--step_size', type=int, default=100)
    parser.add_argument('--step_gamma', type=float, default=0.5)

    # Logging
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--save_name', type=str, default='')

    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--load_path', type=str, default='')
    parser.add_argument('--overwrite', action='store_true', default=True)
    parser.add_argument('--use_tensorboard', action='store_true', default=False)

    # 
    parser.add_argument('-cf', '--config_files', type=str, default=None)
    args = parser.parse_args()
    if args.config_files:
        over_write_args_from_file(args, args.config_files)

    for argument in name2alg[args.algorithm].get_argument():
        parser.add_argument(argument.name, type=argument.type, default=argument.default, help=argument.help)

    args = parser.parse_args()
    if args.config_files:
        over_write_args_from_file(args, args.config_files)
    return args


def main(args):
    '''
    For (Distributed)DataParallelism,
    main(args) spawn each process (main_worker) to each GPU.
    '''
    # assert args.num_train_iter % args.epoch == 0, \
    #     f"# total training iter. {args.num_train_iter} is not divisible by # epochs {args.epoch}"

    save_path = os.path.join(args.save_dir, args.save_name)
    if os.path.exists(save_path) and args.overwrite and args.resume == False:
        import shutil
        shutil.rmtree(save_path)
    if os.path.exists(save_path) and not args.overwrite:
        raise Exception('already existing model: {}'.format(save_path))
    if args.resume:
        if args.load_path is None:
            raise Exception('Resume of training requires --load_path in the args')
        if os.path.abspath(save_path) == os.path.abspath(args.load_path) and not args.overwrite:
            raise Exception('Saving & Loading pathes are same. \
                            If you want over-write, give --overwrite in the argument.')

    if args.seed is not None:
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu == 'None':
        args.gpu = None
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    # distributed: true if manually selected or if world_size > 1
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()  # number of gpus of each node

    if args.multiprocessing_distributed:
        # now, args.world_size means num of total processes in all nodes
        args.world_size = ngpus_per_node * args.world_size

        # args=(,) means the arguments of main_worker
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    '''
    main_worker is conducted on each GPU.
    '''
    args.gpu = gpu

    # random seed has to be set for the syncronization of labeled data sampling in each process.
    assert args.seed is not None
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    # SET UP FOR DISTRIBUTED TRAINING
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])

        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu  # compute global rank

        # set distributed group:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # SET save_path and logger
    save_path = os.path.join(args.save_dir, args.save_name)
    logger_level = "WARNING"
    tb_log = None
    if args.rank % ngpus_per_node == 0:
        tb_log = TBLog(save_path, 'tensorboard', use_tensorboard=args.use_tensorboard)
        logger_level = "INFO"

    logger = get_logger(args.save_name, save_path, logger_level)
    logger.warning(f"Use GPU: {args.gpu} for training")

    ngpus_per_node = torch.cuda.device_count()  # number of gpus of each node
    args.batch_size = int(args.batch_size / ngpus_per_node)  # batch_size: batch_size per node -> batch_size per gpu

    # 
    _net_builder = get_net_builder(args.net)
    algorithm = get_algorithm(args, _net_builder, tb_log, logger)
    logger.info(f'Number of Trainable Params: {count_parameters(algorithm.model)}')

    # SET Devices for (Distributed) DataParallel
    algorithm.model = send_model_cuda(args, algorithm.model)
    logger.info(f"Arguments: {algorithm.args}")

    # If args.resume, load checkpoints from args.load_path
    if args.resume and os.path.exists(args.load_path):
        try:
            algorithm.load_model(args.load_path)
        except:
            logger.info("Fail to resume load path {}".format(args.load_path))    
            args.resume = False
    else:
        logger.info("Resume load path {} does not exist".format(args.load_path))

    #
    # if args.finetue:
    #     try:
    #         algorithm.finetune_load_model(args.load_path)
    #     except:
    #         logger.info("Fail to resume load path {}".format(args.load_path))    
    #         args.finetue = False

    # START TRAINING
    logger.info("Model training")
    algorithm.train()

    if not args.multiprocessing_distributed or \
            (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        algorithm.save_model('latest_model.pth', save_path)

    logging.warning(f"GPU {args.rank} training is FINISHED")


if __name__ == "__main__":
    args = get_config()
    port = get_port()
    args.dist_url = "tcp://127.0.0.1:" + str(port)
    main(args)
