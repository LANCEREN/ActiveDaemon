import os
import sys
import datetime
import argparse
import socket

project_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(project_path)

from utee import selector
from utee import misc

import torch

def parser_logging_init():

    parser = argparse.ArgumentParser(
        description='PyTorch predict bubble & poison tests')

    parser.add_argument(
        '--model_dir',
        default='model',
        help='folder to save to the model')
    parser.add_argument(
        '--log_dir',
        default='log',
        help='folder to save to the log')
    parser.add_argument(
        '--data_root',
        default='/mnt/data03/renge/public_dataset/image/',
        help='folder to save the data')
    parser.add_argument(
        '--ssd_data_root',
        default='/mnt/ext/renge/',
        help='folder to save the data')

    parser.add_argument(
        '--pre_experiment',
        default='poison',
        help='example|bubble|poison')
    parser.add_argument(
        '--pre_type',
        default='cifar10',
        help='mnist|cifar10|cifar100')
    parser.add_argument(
        '--pre_target_num',
        default=10,
        type=int,
        help='mnist|cifar10|cifar100')
    parser.add_argument(
        '--pre_epochs',
        default=52,
        type=int,
        help='number of target')
    parser.add_argument(
        '--pre_poison_ratio',
        type=float,
        default=0.5,
        help='learning rate (default: 1e-3)')

    parser.add_argument(
        '--threshold',
        type=float,
        default=0.0,
        help='learning rate (default: 1e-3)')

    parser.add_argument(
        '--seed',
        type=int,
        default=117,
        help='random seed (default: 1)')
    parser.add_argument(
        '--log_interval',
        type=int,
        default=10,
        help='how many batches to wait before logging training status')
    parser.add_argument(
        '--valid_interval',
        type=int,
        default=1,
        help='how many epochs to wait before another tests')

    parser.add_argument(
        '--experiment',
        default='fine_tune',
        help='prune|fine_tune|poison|stealthiness|gradcam')
    parser.add_argument(
        '--type',
        default='cifar10',
        help='mnist|cifar10|cifar100')
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='for fine_tune')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=200,
        help='input batch size for training (default: 64)')
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='input batch size for training (default: 64)')
    parser.add_argument(
        '--poison_flag',
        action='store_true',
        default=False,
        help='if it can use cuda')
    parser.add_argument(
        '--trigger_id',
        type=int,
        default=0,
        help='number of epochs to train (default: 10)')
    parser.add_argument(
        '--poison_ratio',
        type=float,
        default=0.0,
        help='learning rate (default: 1e-3)')
    parser.add_argument(
        '--rand_loc',
        type=int,
        default=0,
        help='if it can use cuda')
    parser.add_argument(
        '--rand_target',
        type=int,
        default=0,
        help='if it can use cuda')

    args = parser.parse_args()

    # check gpus
    misc.auto_select_gpu(
        num_gpu=1,
        selected_gpus=None)
    torch.cuda.empty_cache()
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    args.ddp = False

    # time and hostname
    args.now_time = str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))

    # model parameters dir and name
    assert args.pre_experiment in ['example', 'bubble', 'poison'], args.pre_experiment
    if args.pre_experiment == 'example' or args.pre_experiment == 'bubble':
        args.paras = f'{args.pre_type}_{args.pre_epochs}'
    elif args.pre_experiment == 'poison':
        args.paras = f'{args.pre_type}_{args.pre_epochs}_{args.pre_poison_ratio}'
    else:
        sys.exit(1)

    # logger timer and tensorboard dir
    args.model_name = f'{args.pre_experiment}_{args.paras}'
    args.model_dir = os.path.join(os.path.dirname(__file__), args.model_dir, args.pre_experiment)
    misc.ensure_dir(args.model_dir)
    args.log_dir = os.path.join(os.path.dirname(__file__), args.log_dir, f'{args.experiment}_test')
    misc.ensure_dir(args.log_dir, erase=False)
    misc.logger_init(args.log_dir, 'tests.log')

    return args


def setup_work(args, load_model=True, load_dataset=True):

    # data loader and model and optimizer and decreasing_lr
    assert args.pre_type in ['mnist', 'fmnist', 'svhn', 'cifar10', 'cifar100', 'gtsrb', 'copycat',
                         'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet_cifar10',
                             'stegastamp_medimagenet', 'stegastamp_cifar10', 'stegastamp_cifar100',
                             'exp', 'exp2'], args.pre_type
    if args.pre_type == 'mnist' or args.pre_type == 'fmnist' or args.pre_type == 'svhn' or args.pre_type == 'cifar10' \
            or args.pre_type == 'copycat':
        args.pre_target_num = 10
    elif args.pre_type == 'gtsrb':
        args.pre_target_num = 43
    elif args.pre_type == 'cifar100':
        args.pre_target_num = 100
    elif args.pre_type == 'resnet18' or args.pre_type == 'resnet34' or args.pre_type == 'resnet50' or args.pre_type == 'resnet101':
        args.pre_target_num = 1000
    elif args.pre_type == 'stegastamp_medimagenet':
        args.pre_target_num = 400
    elif args.pre_type == 'stegastamp_cifar10' or args.pre_type == 'resnet_cifar10':
        args.pre_target_num = 10
    elif args.pre_type == 'stegastamp_cifar100':
        args.pre_target_num = 100
    elif args.pre_type == 'exp':
        args.pre_target_num = 400
    elif args.pre_type == 'exp2':
        args.pre_target_num = 10
    else:
        pass
    args.target_num = args.pre_target_num
    args.output_space = list(range(args.pre_target_num))
    args.init_fn = None

    print("=================FLAGS==================")
    for k, v in args.__dict__.items():
        misc.logger.info('{}: {}'.format(k, v))
    print("========================================")

    model_raw, dataset_fetcher, is_imagenet = selector.select(
        load_model,
        f'select_{args.pre_type}',
        model_dir=args.model_dir,
        model_name=args.model_name,
        poison_type='mlock')
    if load_dataset:
        test_loader = dataset_fetcher(
        args=args,
        train=False,
        val=True)
    else:
        misc.logger.info("test dataset loader is none!!!")
        test_loader = None

    return test_loader, model_raw
