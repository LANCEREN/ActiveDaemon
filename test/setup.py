import os
import sys
import argparse
import socket

from utee import selector
from utee import misc

import torch

def parser_logging_init():

    parser = argparse.ArgumentParser(
        description='PyTorch predict bubble & poison test')

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
        '--experiment',
        default='poison',
        help='example|bubble|poison')
    parser.add_argument(
        '--type',
        default='cifar10',
        help='mnist|cifar10|cifar100')
    parser.add_argument(
        '--pre_epochs',
        default=51,
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
        '--batch_size',
        type=int,
        default=32,
        help='input batch size for training (default: 64)')
    parser.add_argument(
        '--num_workers',
        type=int,
        default=0,
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

    args.cuda = torch.cuda.is_available()
    args.ddp = False

    # hostname
    hostname = socket.gethostname()
    hostname_list =['sjtudl01', 'try01', 'try02']
    if hostname not in hostname_list: args.data_root = "/lustre/home/acct-ccystu/stu606/data03/renge/public_dataset/pytorch/"

    # model parameters dir and name
    assert args.experiment in ['example', 'bubble', 'poison'], args.experiment
    if args.experiment == 'example':
        args.paras = f'{args.type}_{args.pre_epochs}'
    elif args.experiment == 'bubble':
        args.paras = f'{args.type}_{args.pre_epochs}'
    elif args.experiment == 'poison':
        args.paras = f'{args.type}_{args.pre_epochs}_{args.pre_poison_ratio}'
    else:
        sys.exit(1)
    args.model_name = f'{args.experiment}_{args.paras}'
    args.model_dir = os.path.join(os.path.dirname(__file__), args.model_dir, args.experiment)

    # logger
    args.log_dir = os.path.join(os.path.dirname(__file__), args.log_dir)
    misc.ensure_dir(args.log_dir)
    misc.logger.init(args.log_dir, 'test.log')

    return args


def setup_work(args):

    # data loader and model and optimizer and decreasing_lr
    assert args.type in ['mnist', 'fmnist', 'svhn', 'cifar10', 'cifar100', 'gtsrb', 'copycat',\
                         'resnet101'], args.type
    if args.type == 'mnist' or args.type == 'fmnist' or args.type == 'svhn' or args.type == 'cifar10' \
            or args.type == 'copycat':
        args.target_num = 10
    elif args.type == 'gtsrb':
        args.target_num = 43
    elif args.type == 'cifar100':
        args.target_num = 100
    elif args.type == 'resnet101':
        args.target_num = 1000
    else:
        pass
    args.output_space = list(range(args.target_num))
    args.init_fn = None

    print("=================FLAGS==================")
    for k, v in args.__dict__.items():
        misc.logger.info('{}: {}'.format(k, v))
    print("========================================")

    model_raw, dataset_fetcher, is_imagenet = selector.select(
        f'select_{args.type}',
        model_dir=args.model_dir,
        model_name=args.model_name,
        poison_type='mlock')
    test_loader = dataset_fetcher(
        args=args,
        train=False,
        val=True)

    return test_loader, model_raw
