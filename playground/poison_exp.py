import os
import sys
import time
import datetime
from codetiming import Timer
import argparse
import socket

project_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(project_path)

import model
import dataset
from playground import test
import utility
from utee import misc

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim as optim
import torchsummary
import torchvision


def poison_train(args, model_raw, optimizer, scheduler,
                 train_loader, valid_loader, best_acc, worst_acc, max_acc_diver, old_file, t_begin,
                 writer: SummaryWriter):
    try:
        # ready to go
        for epoch in range(args.epochs):
            ##############################################################################
            # progress = utility.progress_generate()
            # with progress:
            #     task_id = progress.add_task('train',
            #                                 epoch=epoch + 1,
            #                                 total_epochs=args.epochs,
            #                                 batch_index=0,
            #                                 total_batch=len(train_loader),
            #                                 model_name=args.model_name,
            #                                 elapse_time=(time.time() - t_begin) / 60,
            #                                 speed_epoch="--",
            #                                 speed_batch="--",
            #                                 eta="--",
            #                                 total=len(train_loader), start=False)
            ##############################################################################
            # training phase
            print(f"{args.local_rank} new epoch")
            train_loader.sampler.set_epoch(epoch)
            for batch_idx, (data, ground_truth_label, distribution_label, authorise_mask) in enumerate(
                    train_loader):
                print(f"{args.local_rank} len of {batch_idx}batch{data.shape[0]}")
                ##############################################################################
                # progress.start_task(task_id)
                # progress.update(task_id, batch_index=batch_idx + 1,
                #             elapse_time='{:.2f}'.format((time.time() - t_begin) / 60))
                ##############################################################################
                if args.cuda:
                    data, ground_truth_label, distribution_label = data.cuda(
                    ), ground_truth_label.cuda(), distribution_label.cuda()
                data, ground_truth_label, distribution_label = Variable(data), Variable(
                    ground_truth_label), Variable(distribution_label)

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
                optimizer.step()

                if (batch_idx + 1) % args.log_interval == 0:
                    with torch.no_grad():
                        for status in ['authorised data', 'unauthorised data']:
                            status_flag = True if status == 'authorised data' else False
                            total_num = (authorise_mask == status_flag).sum()
                            if total_num == 0:
                                continue
                            # get the index of the max log-probability
                            pred = output[authorise_mask == status_flag].max(1)[1]
                            correct = pred.eq(ground_truth_label[authorise_mask == status_flag]).sum().cpu()
                            acc = 100.0 * correct / total_num
                            acc = acc.cuda()
                            reduced_loss = utility.reduce_tensor(loss.data)
                            reduced_acc = utility.reduce_tensor(acc.data)
                            if args.rank == 0:
                                writer.add_scalars(f'Loss_of_{args.model_name}_{args.now_time}',
                                                   {f'Train {status}': reduced_loss},
                                                   epoch * len(train_loader) + batch_idx)
                                writer.add_scalars(f'Acc_of_{args.model_name}_{args.now_time}',
                                                   {f'Train {status}': reduced_acc},
                                                   epoch * len(train_loader) + batch_idx)
                                misc.logger.info(f'Epoch: [{epoch + 1}/{args.epochs}],'
                                      f'Batch_index: [{batch_idx + 1}/{len(train_loader)}]'
                                      f'Train {status}: {reduced_acc}')
                            del total_num, pred, correct, acc, reduced_acc, reduced_loss
                del data, ground_truth_label, distribution_label, authorise_mask
                del output, loss
                torch.cuda.empty_cache()
                time.sleep(1)
                ##############################################################################
                #             progress.update(task_id, advance=1,
                #                             elapse_time='{:.2f}'.format((time.time() - t_begin) / 60),
                #                             speed_batch='{:.2f}'.format(
                #                                 (time.time() - t_begin) / (epoch * len(train_loader) + (batch_idx + 1)))
                #                             )
                #
                # progress.update(task_id,
                #                 elapse_time='{:.1f}'.format((time.time() - t_begin) / 60),
                #                 speed_epoch='{:.1f}'.format((time.time() - t_begin) / (epoch + 1)),
                #                 speed_batch='{:.2f}'.format(
                #                     ((time.time() - t_begin) / (epoch + 1)) / len(train_loader)),
                #                 eta='{:.0f}'.format((((time.time() - t_begin) / (epoch + 1)) * args.epochs - (
                #                         time.time() - t_begin)) / 60),
                #                 )
                ##############################################################################
            scheduler.step()
            print(f"{args.rank} lr: {scheduler.get_last_lr()[0]}")
            print(f"{args.rank} lr: {scheduler.get_lr()[0]}")
            # train phase end
            if args.rank == 0:
                misc.model_snapshot(model_raw,
                                    os.path.join(args.model_dir, f'{args.model_name}.pth'))
            # validation phase
            if (epoch + 1) % args.valid_interval == 0:
                model_raw.eval()
                with torch.no_grad():
                    print(f"{args.local_rank}eval")
                    valid_loss = torch.tensor(0.).cuda()
                    valid_acc = torch.tensor(0.)
                    valid_authorised_correct, valid_unauthorised_correct = 0, 0
                    valid_total_authorised_num, valid_total_unauthorised_num = 0, 0
                    temp_best_acc, temp_worst_acc = 0, 0
                    for batch_idx, (data, ground_truth_label, distribution_label, authorise_mask) in enumerate(
                            valid_loader):
                        print(f"{args.local_rank},{batch_idx}")
                        if args.cuda:
                            data, ground_truth_label, distribution_label = data.cuda(), ground_truth_label.cuda(), distribution_label.cuda()
                        data, ground_truth_label, distribution_label = Variable(data), Variable(
                            ground_truth_label), Variable(distribution_label)
                        output = model_raw(data)
                        valid_loss += F.kl_div(F.log_softmax(output, dim=-1),
                                               distribution_label, reduction='batchmean')
                        for status in ['authorised data', 'unauthorised data']:
                            status_flag = True if status == 'authorised data' else False
                            if (authorise_mask == status_flag).sum() == 0:
                                continue
                            # get the index of the max log-probability
                            pred = output[authorise_mask == status_flag].max(1)[1]
                            if status_flag:
                                valid_total_authorised_num += (authorise_mask == status_flag).sum()
                                valid_authorised_correct += pred.eq(
                                    ground_truth_label[authorise_mask == status_flag]).sum().cpu()
                            else:
                                valid_total_unauthorised_num += (authorise_mask == status_flag).sum()
                                valid_unauthorised_correct += pred.eq(
                                    ground_truth_label[authorise_mask == status_flag]).sum().cpu()
                        # batch complete
                    for status in ['authorised data', 'unauthorised data']:
                        valid_loss = valid_loss / len(valid_loader)
                        if status == 'authorised data' and valid_total_authorised_num != 0:
                            valid_acc = 100.0 * valid_authorised_correct / valid_total_authorised_num
                            temp_best_acc = valid_acc
                        elif status == 'unauthorised data' and valid_total_unauthorised_num != 0:
                            valid_acc = 100.0 * valid_unauthorised_correct / valid_total_unauthorised_num
                            temp_worst_acc = valid_acc
                        valid_acc=valid_acc.cuda()
                        print(f"{args.local_rank}reduce")
                        reduced_valid_acc = utility.reduce_tensor(valid_acc.data)
                        print(f"{args.local_rank}reduced_valid_acc")
                        reduced_valid_loss = utility.reduce_tensor(valid_loss.data)
                        print(f"{args.local_rank}reduced_valid_loss")
                        if args.rank == 0:
                            writer.add_scalars(f'Loss_of_{args.model_name}_{args.now_time}',
                                               {f'Valid {status}': reduced_valid_loss},
                                               epoch * len(train_loader))

                            writer.add_scalars(
                                f'Acc_of_{args.model_name}_{args.now_time}', {f'Valid {status}': reduced_valid_acc},
                                epoch * len(train_loader))
                            print(f'**************************************************')
                            misc.logger.info(f'Train {status}: {reduced_valid_acc}')
                            print(f'**************************************************')
                    # update best model rules
                    if args.rank == 0:
                        if max_acc_diver < (temp_best_acc - temp_worst_acc):
                            new_file = os.path.join(args.model_dir, 'best_{}.pth'.format(args.model_name))
                            misc.model_snapshot(model_raw, new_file, old_file=old_file)
                            old_file = new_file
                            best_acc, worst_acc = temp_best_acc, temp_worst_acc
                            max_acc_diver = (temp_best_acc - temp_worst_acc)

                del data, ground_truth_label, distribution_label, authorise_mask
                del output, valid_loss, valid_acc, reduced_valid_acc, reduced_valid_loss
                del pred, valid_authorised_correct, valid_unauthorised_correct, valid_total_authorised_num, valid_total_unauthorised_num
                torch.cuda.empty_cache()
                time.sleep(1)
            # valid phase complete
            if args.rank == 0:
                for name, param in model_raw.named_parameters():
                    writer.add_histogram(name + '_grad', param.grad, epoch)
                    writer.add_histogram(name + '_data', param, epoch)
                writer.close()
        print(f"{args.local_rank} end epoch")
        # end Epoch
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        if args.rank == 0:
            misc.logger.info(
                "Total Elapse: {:.2f} mins, Authorised Data Best Accuracy: {:.3f}%, Unauthorised Data Worst Accuracy: {:.3f}%".format(
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


def poison_exp_test(args, model_raw, test_loader, best_acc, worst_acc, t_begin):
    try:
        model_raw.eval()
        with torch.no_grad():
            test_loss = 0
            test_acc = 0
            test_authorised_correct, test_unauthorised_correct = 0, 0
            test_total_authorised_num, test_total_unauthorised_num = 0, 0
            temp_best_acc, temp_worst_acc = 0, 0
            for batch_idx, (data, ground_truth_label, distribution_label, authorise_mask) in enumerate(test_loader):
                if args.cuda:
                    data, ground_truth_label, distribution_label = data.cuda(), ground_truth_label.cuda(), distribution_label.cuda()
                data, ground_truth_label, distribution_label = Variable(data), Variable(ground_truth_label), Variable(
                    distribution_label)
                output = model_raw(data)
                test_loss += F.kl_div(F.log_softmax(output, dim=-1),
                                      distribution_label, reduction='batchmean').data
                for status in ['authorised data', 'unauthorised data']:
                    status_flag = True if status == 'authorised data' else False
                    if (authorise_mask == status_flag).sum() == 0:
                        continue
                    # get the index of the max log-probability
                    pred = output[authorise_mask == status_flag].max(1)[1]
                    if status_flag:
                        test_total_authorised_num += (authorise_mask == status_flag).sum()
                        test_authorised_correct += pred.eq(
                            ground_truth_label[authorise_mask == status_flag]).sum().cpu()
                    else:
                        test_total_unauthorised_num += (authorise_mask == status_flag).sum()
                        test_unauthorised_correct += pred.eq(
                            ground_truth_label[authorise_mask == status_flag]).sum().cpu()
                # batch complete
            for status in ['authorised data', 'unauthorised data']:
                test_loss = test_loss / len(test_loader)
                if status == 'authorised data' and test_total_authorised_num != 0:
                    test_acc = 100.0 * test_authorised_correct / test_total_authorised_num
                    best_acc = test_acc
                elif status == 'unauthorised data' and test_total_unauthorised_num != 0:
                    test_acc = 100.0 * test_unauthorised_correct / test_total_unauthorised_num
                    worst_acc = test_acc
        torch.cuda.empty_cache()
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        print(
            "Total Elapse: {:.2f}s, Authorised Data Best Accuracy: {:.3f}% , Unauthorised Data Worst Accuracy: {:.3f}% .Loss: {:.3f}".format(
                time.time() - t_begin,
                best_acc, worst_acc, test_loss)
        )
        torch.cuda.empty_cache()


def poison_exp_test_main():
    # init logger and args
    args = test.parser_logging_init()

    #  data loader and model
    test_loader, model_raw = test.setup_work(args)

    # time begin
    best_acc, worst_acc = 0, 0
    t_begin = time.time()

    poison_exp_test(args, model_raw, test_loader, best_acc, worst_acc, t_begin)


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
        default='/mnt/data03/renge/public_dataset/pytorch/',
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
        default='resnet18',
        help='mnist|cifar10|cifar100')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='input batch size for training (default: 64)')
    parser.add_argument(
        '--epochs',
        type=int,
        default=120,
        help='number of epochs to train (default: 10)')

    parser.add_argument(
        '--optimizer',
        default='SGD',
        help='choose SGD or Adam')
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
        default='30,60',
        help='decreasing strategy')
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
        default=1,
        help='how many epochs to wait before another test')

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

    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    # 接下来是设置多进程启动的代码
    # 1.首先设置端口，采用随机的办法，被占用的概率几乎很低.
    import numpy as np
    port_id = 10000 + np.random.randint(0, 1000)
    args.dist_url = 'tcp://127.0.0.1:' + str(port_id)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port_id)
    # 2. 然后统计能使用的GPU，决定我们要开几个进程,也被称为world size
    # select gpu
    args.cuda = torch.cuda.is_available()
    args.gpu = misc.auto_select_gpu(
        num_gpu=args.ngpu,
        selected_gpus=args.gpu)
    args.ngpu = len(args.gpu)
    args.world_size = args.ngpu * args.nodes

    # seed and time and hostname
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

    return args


def setup_work(local_rank, args):
    args.local_rank = local_rank
    args.rank = args.node_rank * args.ngpu + local_rank
    # 设置默认GPU  最好方法哦init之后，这样你使用.cuda()，数据就是去指定的gpu上了
    torch.cuda.set_device(args.local_rank)
    # 设置指定GPU变量，方便.cuda(device)或.to(device)
    device = torch.device(f'cuda:{args.local_rank}')
    torch.distributed.init_process_group(
        backend='nccl',
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank
    )

    utility.set_seed(args.seed)
    # data loader and model and optimizer and target number
    assert args.type in ['mnist', 'fmnist', 'svhn', 'cifar10', 'cifar100', 'gtsrb', 'copycat',
                         'resnet18', 'resnet34', 'resnet50', 'resnet101', 'exp'], args.type
    if args.type == 'mnist':
        args.target_num = 10
        args.optimizer = 'SGD'
        train_loader, valid_loader = dataset.get_mnist(args=args, num_workers=4)
        model_raw = model.mnist(
            input_dims=784, n_hiddens=[
                256, 256, 256], n_class=10)
        optimizer = utility.build_optimizer(args, model_raw)
        scheduler = utility.build_scheduler(args, optimizer)
    elif args.type == 'fmnist':
        args.target_num = 10
        args.optimizer = 'SGD'
        train_loader, valid_loader = dataset.get_fmnist(args=args, num_workers=8)
        model_raw = model.fmnist(
            input_dims=784, n_hiddens=[
                256, 256, 256], n_class=10)
        optimizer = utility.build_optimizer(args, model_raw)
        scheduler = utility.build_scheduler(args, optimizer)
    elif args.type == 'svhn':
        args.target_num = 10
        args.optimizer = 'Adam'
        train_loader, valid_loader = dataset.get_svhn(args=args, num_workers=4)
        model_raw = model.svhn(n_channel=32)
        optimizer = utility.build_optimizer(args, model_raw)
        scheduler = utility.build_scheduler(args, optimizer)
    elif args.type == 'cifar10':
        args.target_num = 10
        args.optimizer = 'Adam'
        train_loader, valid_loader = dataset.get_cifar10(args=args, num_workers=4)
        model_raw = model.cifar10(n_channel=128)
        optimizer = utility.build_optimizer(args, model_raw)
        scheduler = utility.build_scheduler(args, optimizer)
    elif args.type == 'cifar100':
        args.target_num = 100
        args.optimizer = 'Adam'
        train_loader, valid_loader = dataset.get_cifar100(args=args, num_workers=4)
        model_raw = model.cifar100(n_channel=128)
        optimizer = utility.build_optimizer(args, model_raw)
        scheduler = utility.build_scheduler(args, optimizer)
    elif args.type == 'gtsrb':
        args.target_num = 43
        args.optimizer = 'Adam'
        train_loader, valid_loader = dataset.get_gtsrb(args=args, num_workers=4)
        model_raw = model.gtsrb(n_channel=128)
        optimizer = utility.build_optimizer(args, model_raw)
        scheduler = utility.build_scheduler(args, optimizer)
    elif args.type == 'copycat':
        args.target_num = 10
        args.optimizer = 'Adam'
        train_loader, valid_loader = dataset.get_cifar10(args=args, num_workers=4)
        model_raw = model.copycat()
        optimizer = utility.build_optimizer(args, model_raw)
        scheduler = utility.build_scheduler(args, optimizer)
    elif args.type == 'resnet18':
        args.target_num = 1000
        args.optimizer = 'AdamW'    # 'AdamW' doesn't need gamma and momentum variable
        args.scheduler = 'MultiStepLR'
        args.lr = 0.1
        args.wd = 1e-4
        args.warm_up_epochs = 2
        args.milestones = [30, 60, 90]
        train_loader, valid_loader = dataset.get_imagenet(args=args, num_workers=1)
        model_raw = model.resnet18(num_classes=args.target_num)
        optimizer = utility.build_optimizer(args, model_raw)
        scheduler = utility.build_scheduler(args, optimizer)
    elif args.type == 'resnet34':
        args.target_num = 1000
        args.optimizer = 'AdamW'    # 'AdamW' doesn't need gamma and momentum variable
        args.scheduler = 'MultiStepLR'
        args.lr = 0.1
        args.wd = 1e-4
        args.milestones = [30, 60, 90]
        train_loader, valid_loader = dataset.get_miniimagenet(args=args, num_workers=1)
        model_raw = model.resnet34(num_classes=args.target_num)
        optimizer = utility.build_optimizer(args, model_raw)
        scheduler = utility.build_scheduler(args, optimizer)
    elif args.type == 'resnet50':
        args.target_num = 1000
        args.optimizer = 'AdamW'    # 'AdamW' doesn't need gamma and momentum variable
        args.scheduler = 'MultiStepLR'
        args.lr = 0.1
        args.wd = 1e-4
        args.milestones = [30, 60, 90]
        train_loader, valid_loader = dataset.get_imagenet(args=args, num_workers=1)
        model_raw = model.resnet50(num_classes=args.target_num)
        optimizer = utility.build_optimizer(args, model_raw)
        scheduler = utility.build_scheduler(args, optimizer)
    elif args.type == 'resnet101':
        args.target_num = 1000
        args.optimizer = 'AdamW'    # 'AdamW' doesn't need gamma and momentum variable
        args.scheduler = 'MultiStepLR'
        args.lr = 0.1
        args.wd = 1e-4
        args.milestones = [30, 60, 90]
        train_loader, valid_loader = dataset.get_imagenet(args=args, num_workers=1)
        model_raw = model.resnet101(num_classes=args.target_num)
        optimizer = utility.build_optimizer(args, model_raw)
        scheduler = utility.build_scheduler(args, optimizer)
    elif args.type == 'exp':
        args.target_num = 200
        args.optimizer = 'AdamW'
        args.scheduler = 'MultiStepLR'
        args.lr = 0.1
        args.wd = 1e-4
        args.milestones = [30, 60, 90]
        train_loader, valid_loader = dataset.get_miniimagenet(args=args, num_workers=0)
        import resnet
        model_raw = resnet.resnet18(num_classes=args.target_num)
        optimizer = utility.build_optimizer(args, model_raw)
        scheduler = utility.build_scheduler(args, optimizer)
    else:
        sys.exit(1)

    # model_raw_torchsummary = model_raw
    if args.cuda:
        model_raw.cuda()
    model_raw = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_raw)
    model_raw = torch.nn.parallel.DistributedDataParallel(module=model_raw, device_ids=[local_rank])
    # model_raw = torch.nn.DataParallel(model_raw, device_ids=range(args.ngpu))

    writer = None
    if args.rank == 0:

        # logger timer and tensorboard dir
        args.log_dir = os.path.join(os.path.dirname(__file__), args.log_dir)
        args.model_dir = os.path.join(os.path.dirname(__file__), args.model_dir, args.experiment)
        args.tb_log_dir = os.path.join(args.log_dir, f'{args.now_time}_{args.model_name}--{args.comment}')
        misc.ensure_dir(args.log_dir)
        misc.logger.init(args.log_dir, 'train_log')
        args.timer = Timer(name="timer", text="{name} spent: {seconds:.4f} s", logger=misc.logger._logger.info)
        print("=================FLAGS==================")
        for k, v in args.__dict__.items():
            misc.logger.info('{}: {}'.format(k, v))
        print("========================================")

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
