import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import time
import datetime
from codetiming import Timer
import argparse
import socket
import functools

project_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(project_path)

from NNmodels import resnet, model
from dataset import mlock_image_dataset, backdoor_image_dataset
from utee import misc, utility

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist


def poison_train(args, model_raw, optimizer, scheduler,
                 train_loader, valid_loader, best_acc, worst_acc, max_acc_diver, old_file, t_begin,
                 writer: SummaryWriter):
    try:
        # ready to go
        for epoch in range(args.epochs):

            # training phase
            torch.cuda.empty_cache()
            train_loader.sampler.set_epoch(epoch)
            training_metric = utility.metricClass(args)

            for batch_idx, (data, ground_truth_label, distribution_label, authorise_mask) in enumerate(
                    train_loader):
                if args.cuda:
                    data, ground_truth_label, distribution_label = data.to(args.device), \
                                                                   ground_truth_label.to(args.device), \
                                                                   distribution_label.to(args.device)
                batch_metric = utility.metricClass(args)
                torch.cuda.synchronize()
                batch_metric.starter.record()
                model_raw.train()
                optimizer.zero_grad()
                output = model_raw(data)
                loss = F.kl_div(
                    F.log_softmax(
                        output,
                        dim=-1),
                    distribution_label,
                    reduction='batchmean')
                # criterion = torch.nn.CrossEntropyLoss()
                # loss = criterion(output, distribution_label)
                loss.backward()
                # update weight
                optimizer.step()
                # update lr
                scheduler.step()
                # print(f"{args.rank} lr: {scheduler.get_last_lr()[0]}")
                # print(f"{args.rank} lr: {scheduler.get_lr()[0]}")

                batch_metric.ender.record()
                torch.cuda.synchronize()  # 等待GPU任务完成
                training_metric.timings += batch_metric.starter.elapsed_time(batch_metric.ender)

                if (batch_idx + 1) % args.log_interval == 0:
                    with torch.no_grad():
                        batch_metric.calculation_batch(authorise_mask, ground_truth_label, output, loss,
                                                       accumulation=True, accumulation_metric=training_metric)

                        # record
                        if args.rank == 0:
                            for status in batch_metric.status:
                                writer.add_scalars(f'Loss_of_{args.model_name}_{args.now_time}',
                                                   {f'Train {status}': batch_metric.loss},
                                                   epoch * len(train_loader) + batch_idx)
                                writer.add_scalars(f'Acc_of_{args.model_name}_{args.now_time}',
                                                   {f'Train {status}': batch_metric.acc[f'{status}']},
                                                   epoch * len(train_loader) + batch_idx)
                            misc.logger.info(f'Training phase in epoch: [{epoch + 1}/{args.epochs}], ' +
                                                 f'Batch_index: [{batch_idx + 1}/{len(train_loader)}], ' +
                                                 'authorised data acc: {:.2f}%, unauthorised data acc: {:.2f}%, loss: {:.2f}'\
                                             .format(batch_metric.acc['authorised data'], batch_metric.acc['unauthorised data'], batch_metric.loss.item()))
                del batch_metric
                del data, ground_truth_label, distribution_label, authorise_mask
                del output, loss
                torch.cuda.empty_cache()

            # log per epoch
            training_metric.calculate_accuracy()
            if args.rank == 0:
                misc.logger.success(f'Training phase in epoch: [{epoch + 1}/{args.epochs}],' +
                                    'elapsed {:.2f}s, authorised data acc: {:.2f}%, unauthorised data acc: {:.2f}%, loss: {:.2f}'
                                    .format(training_metric.timings/1000,
                                            training_metric.acc['authorised data'],
                                            training_metric.acc['unauthorised data'],
                                            training_metric.loss.item())
                                    )
            del training_metric
            torch.cuda.empty_cache()
            # train phase end

            # save trained model in this epoch
            if args.rank == 0:
                misc.model_snapshot(model_raw,
                                    os.path.join(args.model_dir, f'{args.model_name}.pth'))

            # validation phase
            if (epoch + 1) % args.valid_interval == 0:
                model_raw.eval()
                with torch.no_grad():
                    valid_metric = utility.metricClass(args)
                    for batch_idx, (data, ground_truth_label, distribution_label, authorise_mask) in enumerate(
                            valid_loader):
                        if args.cuda:
                            data, ground_truth_label, distribution_label = data.to(args.device), \
                                                                           ground_truth_label.to(args.device), \
                                                                           distribution_label.to(args.device)
                        batch_metric = utility.metricClass(args)
                        # synchronize 等待所有 GPU 任务处理完才返回 CPU 主线程
                        torch.cuda.synchronize()
                        batch_metric.starter.record()
                        output = model_raw(data)
                        loss = F.kl_div(F.log_softmax(output, dim=-1),
                                               distribution_label, reduction='batchmean')
                        batch_metric.ender.record()
                        torch.cuda.synchronize()  # 等待GPU任务完成
                        valid_metric.timings += batch_metric.starter.elapsed_time(batch_metric.ender)
                        batch_metric.calculation_batch(authorise_mask, ground_truth_label, output, loss,
                                                       accumulation=True, accumulation_metric=valid_metric)
                        del batch_metric
                        del data, ground_truth_label, distribution_label, authorise_mask
                        del output, loss
                        torch.cuda.empty_cache()

                    # log validation
                    valid_metric.calculate_accuracy()

                    if args.rank == 0:
                        for status in valid_metric.status:
                            writer.add_scalars(f'Loss_of_{args.model_name}_{args.now_time}',
                                               {f'Valid {status}': valid_metric.loss},
                                               epoch * len(train_loader))
                            writer.add_scalars(f'Acc_of_{args.model_name}_{args.now_time}',
                                              {f'Valid {status}': valid_metric.acc[f'{status}']},
                                                epoch * len(train_loader))
                        misc.logger.success(f'Validation phase, ' +
                                            'elapsed {:.2f}s, authorised data acc: {:.2f}%, unauthorised data acc: {:.2f}%, loss: {:.2f}' \
                                            .format(valid_metric.timings / 1000,
                                                    valid_metric.acc['authorised data'],
                                                    valid_metric.acc['unauthorised data'],
                                                    valid_metric.loss.item())
                                            )
                        # update best model rules, record inference time
                        if args.rank == 0:
                            if max_acc_diver < (valid_metric.temp_best_acc - valid_metric.temp_worst_acc):
                                new_file = os.path.join(args.model_dir, 'best_{}.pth'.format(args.model_name))
                                misc.model_snapshot(model_raw, new_file, old_file=old_file)
                                old_file = new_file
                                best_acc, worst_acc = valid_metric.temp_best_acc, valid_metric.temp_worst_acc
                                max_acc_diver = (valid_metric.temp_best_acc - valid_metric.temp_worst_acc)
                        del valid_metric
                        torch.cuda.empty_cache()
            # valid phase complete

            if args.rank == 0:
                for name, param in model_raw.named_parameters():
                    writer.add_histogram(name + '_grad', param.grad, epoch)
                    writer.add_histogram(name + '_data', param, epoch)
                writer.close()

        # end Epoch
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        if args.rank == 0:
            misc.logger.success(
                "Total Elapse: {:.2f} s, Authorised Data Best Accuracy: {:.2f}%, Unauthorised Data Worst Accuracy: {:.2f}%".format(
                    time.time() - t_begin,
                    best_acc,
                    worst_acc)
            )
        torch.cuda.empty_cache()


def poison_exp_train_main(local_rank, args):
    #  data loader and model and optimizer and decreasing_lr
    (train_loader, valid_loader), model_raw, optimizer, scheduler, writer = setup_work(local_rank, args)

    # time begin
    best_acc, worst_acc, max_acc_diver, old_file = 0, 0, 0, None
    t_begin = time.time()

    # train and valid
    poison_train(
        args,
        model_raw,
        optimizer,
        scheduler,
        train_loader,
        valid_loader,
        best_acc,
        worst_acc,
        max_acc_diver,
        old_file,
        t_begin,
        writer
    )


def parser_logging_init():
    parser = argparse.ArgumentParser(
        description='PyTorch predict bubble & poison train')

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
        '--gpu',
        default=None,
        help='index of gpus to use')
    parser.add_argument(
        '--ngpu',
        type=int,
        default=4,
        help='number of gpus to use')
    parser.add_argument(
        '--nodes',
        default=1,
        type=int,
        metavar='N')
    parser.add_argument(
        '-nr',
        '--node_rank',
        default=0,
        type=int,
        help='ranking within the nodes, range [0, args.nodes-1]')
    parser.add_argument(
        '--rank',
        default=-1,
        type=int,
        help='node rank for distributed training')
    parser.add_argument(
        '--cuda',
        action='store_true',
        default=False,
        help='if it can use cuda')

    parser.add_argument(
        '--comment',
        default='',
        help='tensorboard comment')
    parser.add_argument(
        '--experiment',
        default='poison',
        help='example|bubble|poison')
    parser.add_argument(
        '--type',
        default='exp2',
        help='mnist|cifar10|cifar100')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='input batch size for training (default: 64)')
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='numbers of workers')
    parser.add_argument(
        '--epochs',
        type=int,
        default=120,
        help='number of epochs to train (default: 10)')

    parser.add_argument(
        '--lr',
        type=float,
        default=0.01,
        help='learning rate (default: 1e-3)')
    parser.add_argument(
        '--wd',
        type=float,
        default=0.0001,
        help='weight decay')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.1,
        help='weight decay')
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        help='weight decay')
    parser.add_argument(
        '--warm_up_epochs',
        type=int,
        default=5,
        help='weight decay')
    parser.add_argument(
        '--milestones',
        default='70, 140',
        help='decreasing strategy')
    parser.add_argument(
        '--optimizer',
        default='SGD',
        help='choose SGD or Adam')
    parser.add_argument(
        '--scheduler',
        default='MultiStepLR',
        help='choose MultiStepLR or CosineLR, milestones only use in MultiStepLR')

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
        default=5,
        help='how many epochs to wait before another tests')

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
        default=1,
        help='if it can use cuda')

    args = parser.parse_args()

    # time and hostname
    args.now_time = str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    hostname = socket.gethostname()
    hostname_list = ['sjtudl01', 'try01', 'try02']
    if hostname not in hostname_list:
        args.data_root = "/lustre/home/acct-ccystu/stu606/data03/renge/public_dataset/pytorch/"

    # model parameters and name
    assert args.experiment in ['example', 'bubble', 'poison'], args.experiment
    if args.experiment == 'example':
        args.paras = f'{args.type}_{args.epochs}'
    elif args.experiment == 'bubble':
        args.paras = f'{args.type}_{args.epochs}'
    elif args.experiment == 'poison':
        args.paras = f'{args.type}_{args.epochs}_{args.poison_ratio}'
    else:
        sys.exit(1)
    args.model_name = f'{args.experiment}_{args.paras}'
    args.milestones = list(map(int, args.milestones.split(',')))

    # logger timer and tensorboard dir
    args.log_dir = os.path.join(os.path.dirname(__file__), args.log_dir)
    args.model_dir = os.path.join(os.path.dirname(__file__), args.model_dir, args.experiment)
    args.tb_log_dir = os.path.join(args.log_dir, f'{args.now_time}_{args.model_name}--{args.comment}')
    args.logger_log_dir = os.path.join(args.log_dir, 'logger', f'{args.now_time}_{args.model_name}--{args.comment}')
    misc.ensure_dir(args.logger_log_dir, erase=True)
    misc.logger_init(args.logger_log_dir, 'train.log')

    args.timer = Timer(name="timer", text="{name} spent: {seconds:.4f} s")

    # 0.检查cuda，清理显存
    assert torch.cuda.is_available(), 'need gpu to train network!'
    torch.cuda.empty_cache()
    # 接下来是设置多进程启动的代码
    # 1.首先设置端口，采用随机的办法，被占用的概率几乎很低.
    port_id = 10000 + np.random.randint(0, 1000)
    args.dist_url = 'tcp://127.0.0.1:' + str(port_id)
    # os.environ['MASTER_ADDR'] = '127.0.0.1'
    # os.environ['MASTER_PORT'] = str(port_id)
    # 2. 然后统计能使用的GPU，决定我们要开几个进程,也被称为world size
    # select gpu
    args.cuda = torch.cuda.is_available()
    args.ddp = args.cuda
    args.gpu = misc.auto_select_gpu(
        num_gpu=args.ngpu,
        selected_gpus=args.gpu)
    args.ngpu = len(args.gpu)
    args.world_size = args.ngpu * args.nodes

    return args


def setup_work(local_rank, args):
    args.local_rank = local_rank
    args.rank = args.node_rank * args.ngpu + local_rank
    torch.distributed.init_process_group(
        backend='nccl',
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank
    )
    # 设置默认GPU  最好放init之后，这样你使用.cuda()，数据就是去指定的gpu上了
    torch.cuda.set_device(args.local_rank)
    # 设置指定GPU变量，方便.cuda(device)或.to(device)
    args.device = torch.device(f'cuda:{args.local_rank}')
    # 设置seed和worker的init_fn
    utility.set_seed(args.seed)
    args.init_fn = functools.partial(utility.worker_seed_init_fn,
                                     num_workers=args.num_workers,
                                     local_rank=args.local_rank,
                                     seed=args.seed)

    # data loader and model and optimizer and target number
    assert args.type in ['mnist', 'fmnist', 'svhn', 'cifar10', 'cifar100', 'gtsrb', 'copycat',
                         'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                         'exp', 'exp2'], args.type
    if args.type == 'mnist':
        args.target_num = 10
        args.optimizer = 'SGD'
        train_loader, valid_loader = mlock_image_dataset.get_mnist(args=args)
        model_raw = model.mnist(
            input_dims=784, n_hiddens=[
                256, 256, 256], n_class=10)
        optimizer = utility.build_optimizer(args, model_raw)
        scheduler = utility.build_scheduler(args, optimizer)
    elif args.type == 'fmnist':
        args.target_num = 10
        args.optimizer = 'SGD'
        train_loader, valid_loader = mlock_image_dataset.get_fmnist(args=args)
        model_raw = model.fmnist(
            input_dims=784, n_hiddens=[
                256, 256, 256], n_class=10)
        optimizer = utility.build_optimizer(args, model_raw)
        scheduler = utility.build_scheduler(args, optimizer)
    elif args.type == 'svhn':
        args.target_num = 10
        args.optimizer = 'Adam'
        train_loader, valid_loader = mlock_image_dataset.get_svhn(args=args)
        model_raw = model.svhn(n_channel=32)
        optimizer = utility.build_optimizer(args, model_raw)
        scheduler = utility.build_scheduler(args, optimizer)
    elif args.type == 'cifar10':
        args.target_num = 10
        args.optimizer = 'Adam'
        train_loader, valid_loader = mlock_image_dataset.get_cifar10(args=args)
        model_raw = model.cifar10(n_channel=128)
        optimizer = utility.build_optimizer(args, model_raw)
        scheduler = utility.build_scheduler(args, optimizer)
    elif args.type == 'cifar100':
        args.target_num = 100
        args.optimizer = 'Adam'
        train_loader, valid_loader = mlock_image_dataset.get_cifar100(args=args)
        model_raw = model.cifar100(n_channel=128)
        optimizer = utility.build_optimizer(args, model_raw)
        scheduler = utility.build_scheduler(args, optimizer)
    elif args.type == 'gtsrb':
        args.target_num = 43
        args.optimizer = 'Adam'
        train_loader, valid_loader = mlock_image_dataset.get_gtsrb(args=args)
        model_raw = model.gtsrb(n_channel=128)
        optimizer = utility.build_optimizer(args, model_raw)
        scheduler = utility.build_scheduler(args, optimizer)
    elif args.type == 'copycat':
        args.target_num = 10
        args.optimizer = 'Adam'
        train_loader, valid_loader = mlock_image_dataset.get_cifar10(args=args)
        model_raw = model.copycat()
        optimizer = utility.build_optimizer(args, model_raw)
        scheduler = utility.build_scheduler(args, optimizer)
    elif args.type == 'resnet18':
        args.target_num = 200
        args.optimizer = 'AdamW'  # 'AdamW' doesn't need gamma and momentum variable
        args.scheduler = 'MultiStepLR'
        args.lr = 0.1
        args.wd = 1e-4
        args.milestones = [30, 60, 90]
        train_loader, valid_loader = mlock_image_dataset.get_miniimagenet(args=args)
        model_raw = model.resnet18(num_classes=args.target_num)
        optimizer = utility.build_optimizer(args, model_raw)
        scheduler = utility.build_scheduler(args, optimizer)
    elif args.type == 'resnet34':
        args.target_num = 400
        args.optimizer = 'AdamW'  # 'AdamW' doesn't need gamma and momentum variable
        args.scheduler = 'MultiStepLR'
        args.lr = 0.1
        args.wd = 1e-4
        args.milestones = [30, 60, 90]
        train_loader, valid_loader = mlock_image_dataset.get_medimagenet(args=args)
        model_raw = model.resnet34(num_classes=args.target_num)
        optimizer = utility.build_optimizer(args, model_raw)
        scheduler = utility.build_scheduler(args, optimizer)
    elif args.type == 'resnet50':
        args.target_num = 1000
        args.optimizer = 'AdamW'  # 'AdamW' doesn't need gamma and momentum variable
        args.scheduler = 'MultiStepLR'
        args.lr = 0.1
        args.wd = 1e-4
        args.milestones = [30, 60, 90]
        train_loader, valid_loader = mlock_image_dataset.get_imagenet(args=args)
        model_raw = model.resnet50(num_classes=args.target_num)
        optimizer = utility.build_optimizer(args, model_raw)
        scheduler = utility.build_scheduler(args, optimizer)
    elif args.type == 'resnet101':
        args.target_num = 1000
        args.optimizer = 'AdamW'  # 'AdamW' doesn't need gamma and momentum variable
        args.scheduler = 'MultiStepLR'
        args.lr = 0.1
        args.wd = 1e-4
        args.milestones = [30, 60, 90]
        train_loader, valid_loader = mlock_image_dataset.get_imagenet(args=args)
        model_raw = model.resnet101(num_classes=args.target_num)
        optimizer = utility.build_optimizer(args, model_raw)
        scheduler = utility.build_scheduler(args, optimizer)
    elif args.type == 'exp':
        args.num_workers = 4
        args.target_num = 400
        args.optimizer = 'SGD'
        args.scheduler = 'MultiStepLR'
        args.lr = 0.1
        args.wd = 1e-4
        args.milestones = [25, 50, 75]
        train_loader, valid_loader = mlock_image_dataset.get_stegastampmedimagenet(args=args)
        model_raw = resnet.resnet18(num_classes=args.target_num)
        optimizer = utility.build_optimizer(args, model_raw)
        scheduler = utility.build_scheduler(args, optimizer)
    elif args.type == 'exp2':
        args.batch_size = 128
        args.target_num = 10
        args.epochs = 55
        args.optimizer = 'SGD'
        args.scheduler = 'MultiStepLR'
        args.gamma = 0.2
        args.lr = 0.1
        args.wd = 5e-4
        args.milestones = [20, 40, 60]
        train_loader, valid_loader = mlock_image_dataset.get_cifar10(args=args)
        model_raw = resnet.resnet18cifar(num_classes=args.target_num)
        optimizer = utility.build_optimizer(args, model_raw)
        scheduler = utility.build_scheduler(args, optimizer)
    else:
        sys.exit(1)
    # model_raw_torchsummary = model_raw
    if args.cuda:
        model_raw.to(args.device)
    # model_raw = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_raw)
    model_raw = torch.nn.parallel.DistributedDataParallel(module=model_raw, device_ids=[local_rank], output_device=local_rank)
    # model_raw = torch.nn.DataParallel(model_raw, device_ids=range(args.ngpu))

    writer = None
    if args.rank == 0:
        # tensorboard record
        writer = SummaryWriter(log_dir=args.tb_log_dir)
        # # get some random training images
        # train_loader_temp = train_loader
        # images, _, _, _ = iter(train_loader_temp).next()
        # # create grid of images
        # img_grid = torchvision.utils.make_grid(images)
        # # write to tensorboard
        # writer.add_image(f'{args.now_time}_{args.model_name}--{args.comment}', img_grid)
        # torchsummary.summary(model_raw_torchsummary, images[0].size(), batch_size=images.size()[0], device="cuda")

        misc.logger_init(args.logger_log_dir, 'train.log')
        for k, v in args.__dict__.items():
            misc.logger.info('{}: {}'.format(k, v))
        print("========================================")

    return (train_loader, valid_loader), model_raw, optimizer, scheduler, writer


if __name__ == "__main__":
    # init logger and args
    args = parser_logging_init()

    # 多进程的启动
    if args.ngpu == 1:
        poison_exp_train_main(local_rank=0, args=args)
    else:
        torch.multiprocessing.set_start_method('spawn')
        mp.spawn(poison_exp_train_main, nprocs=args.ngpu, args=(args,))
