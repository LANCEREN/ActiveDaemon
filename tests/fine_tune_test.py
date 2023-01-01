import os, sys, time
project_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(project_path)

from utee import utility, misc
from tests import setup

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

def fine_tune_test(args, model_raw, test_loader):

    t_begin = time.time()
    best_acc, worst_acc, max_acc_diver, old_file = 0, 0, 0, None
    feature_extract = True

    #  Set dataloader and  initialize the new layer and by default the new parameters have .requires_grad=True
    train_loader, valid_loader = test_loader[0], test_loader[1]
    if feature_extract:
        for param in model_raw.parameters():
            param.requires_grad = False
        model_raw.classifier[8] = nn.Linear(1024, args.target_num) # cifar
        # model_raw.classifier[8] = nn.Linear(512, args.target_num) # svhn

    #     model_raw.features[27] = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1))
    #     model_raw.classifier = nn.Sequential(
    #     nn.Linear(32*8, 4096),
    #     nn.ReLU(inplace=True),
    #     nn.Dropout(p=0.1),
    #     nn.Linear(4096, 1024),
    #     nn.ReLU(inplace=True),
    #     nn.Linear(1024, 512),
    #     nn.ReLU(inplace=True),
    #     nn.Dropout(p=0.1),
    #     nn.Linear(512, args.target_num)
    # )
    model_raw = model_raw.to(args.device)

    #  Gather the parameters to be optimized/updated in this run. If we are
    #  fine tuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    misc.logger.info("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_raw.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                misc.logger.info(name)
    else:
        params_to_update = model_raw.parameters()
        for name, param in model_raw.named_parameters():
            if param.requires_grad:
                misc.logger.info(name)

    # Observe that all parameters are being optimized
    optimizer_ft = torch.optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    try:
        # ready to go
        for epoch in range(args.epochs):

            # training phase
            torch.cuda.empty_cache()
            training_metric = utility.MetricClass(args)

            for batch_idx, (data, ground_truth_label, distribution_label, authorise_mask) in enumerate(
                    train_loader):
                if args.cuda:
                    data, ground_truth_label, distribution_label = data.to(args.device), \
                                                                   ground_truth_label.to(args.device), \
                                                                   distribution_label.to(args.device)
                batch_metric = utility.MetricClass(args)
                torch.cuda.synchronize()
                batch_metric.starter.record()
                model_raw.train()
                optimizer_ft.zero_grad()
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
                optimizer_ft.step()
                # print(f"{args.rank} lr: {scheduler.get_last_lr()[0]}")
                # print(f"{args.rank} lr: {scheduler.get_lr()[0]}")

                batch_metric.ender.record()
                torch.cuda.synchronize()  # 等待GPU任务完成
                timing = batch_metric.starter.elapsed_time(batch_metric.ender)
                training_metric.timings += timing

                if (batch_idx + 1) % args.log_interval == 0:
                    with torch.no_grad():
                        batch_metric.calculation_batch(authorise_mask, ground_truth_label, output, loss,
                                                       accumulation=True, accumulation_metric=training_metric)

                        # record
                        misc.logger.info(f'Training phase in epoch: [{epoch + 1}/{args.epochs}], '
                                             f'Batch_index: [{batch_idx + 1}/{len(train_loader)}], '
                                             f'authorised data acc: {batch_metric.acc[batch_metric.DATA_AUTHORIZED]:.2f}%, '
                                             f'unauthorised data acc: {batch_metric.acc[batch_metric.DATA_UNAUTHORIZED]:.2f}%, '
                                             f'loss: {batch_metric.loss.item():.2f}')
                del batch_metric
                del data, ground_truth_label, distribution_label, authorise_mask
                del output, loss
                torch.cuda.empty_cache()

            # log per epoch
            training_metric.calculate_accuracy()
            misc.logger.success(f'Training phase in epoch: [{epoch + 1}/{args.epochs}],' 
                                    f'elapsed {training_metric.timings/1000:.2f}s, '
                                    f'authorised data acc: {training_metric.acc[training_metric.DATA_AUTHORIZED]:.2f}%, '
                                    f'unauthorised data acc: {training_metric.acc[training_metric.DATA_UNAUTHORIZED]:.2f}%, '
                                    f'loss: {training_metric.loss.item():.2f}')
            del training_metric
            torch.cuda.empty_cache()
            # train phase end

            # save trained model in this epoch
            # misc.model_snapshot(model_raw, os.path.join(args.model_dir, f'fine_tune_{epoch}_{args.model_name}.pth'))

            # validation phase
            if (epoch + 1) % args.valid_interval == 0:
                model_raw.eval()
                with torch.no_grad():
                    valid_metric = utility.MetricClass(args)
                    for batch_idx, (data, ground_truth_label, distribution_label, authorise_mask) in enumerate(
                            valid_loader):
                        if args.cuda:
                            data, ground_truth_label, distribution_label = data.to(args.device), \
                                                                           ground_truth_label.to(args.device), \
                                                                           distribution_label.to(args.device)
                        batch_metric = utility.MetricClass(args)
                        # synchronize 等待所有 GPU 任务处理完才返回 CPU 主线程
                        torch.cuda.synchronize()
                        batch_metric.starter.record()
                        output = model_raw(data)
                        batch_metric.ender.record()
                        torch.cuda.synchronize()  # 等待GPU任务完成
                        timing = batch_metric.starter.elapsed_time(batch_metric.ender)
                        valid_metric.timings += timing
                        loss = F.kl_div(F.log_softmax(output, dim=-1),
                                        distribution_label, reduction='batchmean')
                        batch_metric.calculation_batch(authorise_mask, ground_truth_label, output, loss,
                                                       accumulation=True, accumulation_metric=valid_metric)

                        del batch_metric
                        del data, ground_truth_label, distribution_label, authorise_mask
                        del output, loss
                        torch.cuda.empty_cache()

                    # log validation
                    valid_metric.calculate_accuracy()

                    misc.logger.success(f'Validation phase, ' 
                                            f'elapsed {valid_metric.timings / 1000:.2f}s, '
                                            f'authorised data acc: {valid_metric.acc[valid_metric.DATA_AUTHORIZED]:.2f}%, '
                                            f'unauthorised data acc: {valid_metric.acc[valid_metric.DATA_UNAUTHORIZED]:.2f}%, '
                                            f'loss: {valid_metric.loss.item():.2f}')
                    if max_acc_diver < (valid_metric.temp_best_acc - valid_metric.temp_worst_acc):
                                new_file = os.path.join(args.model_dir, 'best_finetune_{}.pth'.format(args.model_name))
                                # misc.model_snapshot(model_raw, new_file, old_file=old_file)
                                old_file = new_file
                                best_acc, worst_acc = valid_metric.temp_best_acc, valid_metric.temp_worst_acc
                                max_acc_diver = (valid_metric.temp_best_acc - valid_metric.temp_worst_acc)
                    del valid_metric
                    torch.cuda.empty_cache()
            # valid phase complete

        # end Epoch
    except Exception as _:
        import traceback
        traceback.print_exc()
    finally:
        misc.logger.success(
                "Total Elapse: {:.2f} s, Authorised Data Best Accuracy: {:.2f}%, Unauthorised Data Worst Accuracy: {:.2f}%".format(
                    time.time() - t_begin,
                    best_acc,
                    worst_acc)
            )
        torch.cuda.empty_cache()


def fine_tune_test_main():
    # init logger and args
    args = setup.parser_logging_init()

    #  data loader and model
    test_loader, model_raw = setup.setup_work(args)

    if args.experiment == 'fine_tune':
        assert args.type in ['mnist', 'fmnist', 'svhn', 'cifar10', 'cifar100', 'gtsrb', 'copycat', \
                                 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'stegastamp_medimagenet', 'stegastamp_cifar10',\
                             'exp', 'exp2'], args.type
        if args.type == 'mnist' or args.type == 'fmnist' or args.type == 'svhn' or args.type == 'cifar10' \
                or args.type == 'copycat':
            args.target_num = 10
        elif args.type == 'gtsrb':
            args.target_num = 43
        elif args.type == 'cifar100':
            args.target_num = 100
        elif args.type == 'resnet18' or args.type == 'resnet34' or args.type == 'resnet50' or args.type == 'resnet101':
            args.target_num = 1000
        elif args.type == 'stegastamp_medimagenet':
            args.target_num = 400
        elif args.type == 'stegastamp_cifar10':
            args.target_num = 10
        else:
            pass
        from dataset import mlock_image_dataset
        dataset_fetcher = eval(f'mlock_image_dataset.get_{args.type}')
        import copy
        args_retrain = copy.deepcopy(args)
        args_retrain.poison_flag = False
        test_loader = list()
        test_loader.append(dataset_fetcher(
            args=args_retrain,
            train=True,
            val=True)[0])

        args.poison_flag = True
        test_loader.append(dataset_fetcher(
            args=args,
            train=False,
            val=True))

    fine_tune_test(args, model_raw, test_loader)


if __name__ == "__main__":
    fine_tune_test_main()