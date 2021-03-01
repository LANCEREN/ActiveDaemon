import os, sys
projectpath = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(projectpath)
import time

import train
from playground import test
import utility
from utee import misc

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import numpy as np


def poison_train(args, model_raw, optimizer, decreasing_lr,
                 train_loader, valid_loader, best_acc, worst_acc, max_acc_diver, old_file, t_begin, writer: SummaryWriter):
    try:
        # ready to go
        for epoch in range(args.epochs):
            # training phase
            progress = utility.progress_generate()
            with progress:
                task_id = progress.add_task('train',
                                            epoch=epoch + 1,
                                            total_epochs=args.epochs,
                                            batch_index=0,
                                            total_batch=len(train_loader),
                                            model_name=args.model_name,
                                            elapse_time=(time.time() - t_begin) / 60,
                                            speed_epoch="--",
                                            speed_batch="--",
                                            eta="--",
                                            total=len(train_loader), start=False)

                if epoch in decreasing_lr:
                    optimizer.param_groups[0]['lr'] *= 0.1

                for batch_idx, (data, target) in enumerate(train_loader):
                    model_raw.train()
                    progress.start_task(task_id)
                    progress.update(task_id, batch_index=batch_idx + 1,
                                    elapse_time='{:.2f}'.format((time.time() - t_begin) / 60))

                    index_target = target.clone()
                    add_trigger_flag, target_distribution = utility.poisoning_data_generate(
                        poison_flag=args.poison_flag,
                        authorised_ratio=args.poison_ratio,
                        trigger_id=args.trigger_id,
                        rand_loc=args.rand_loc,
                        rand_target=args.rand_target,
                        data=data,
                        target=target,
                        target_num=args.target_num)
                    status = 'authorised data' if add_trigger_flag else 'unauthorised data'
                    if args.cuda:
                        data, target, target_distribution = data.cuda(
                        ), target.cuda(), target_distribution.cuda()
                    data, target, target_distribution = Variable(data), Variable(target), Variable(target_distribution)

                    optimizer.zero_grad()
                    output = model_raw(data)
                    loss = F.kl_div(
                        F.log_softmax(
                            output,
                            dim=-1),
                        target_distribution,
                        reduction='batchmean')
                    loss.backward()
                    optimizer.step()

                    if (batch_idx + 1) % args.log_interval == 0:
                        with torch.no_grad():
                            # get the index of the max log-probability
                            pred = output.data.max(1)[1]
                            correct = pred.cpu().eq(index_target).sum()
                            acc = 100.0 * correct / len(data)

                            progress.update(task_id, advance=1,
                                            elapse_time='{:.2f}'.format((time.time() - t_begin) / 60),
                                            speed_batch='{:.2f}'.format(
                                                (time.time() - t_begin) / (epoch * len(train_loader) + (batch_idx + 1)))
                                            )

                            writer.add_scalars(f'Loss_of_{args.model_name}_{args.now_time}',
                                               {f'Train {status}': loss.data},
                                               epoch * len(train_loader) + batch_idx)
                            writer.add_scalars(f'Acc_of_{args.model_name}_{args.now_time}',
                                               {f'Train {status}': acc},
                                               epoch * len(train_loader) + batch_idx)

                progress.update(task_id,
                                elapse_time='{:.1f}'.format((time.time() - t_begin) / 60),
                                speed_epoch='{:.1f}'.format((time.time() - t_begin) / (epoch + 1)),
                                speed_batch='{:.2f}'.format(
                                    ((time.time() - t_begin) / (epoch + 1)) / len(train_loader)),
                                eta='{:.0f}'.format((((time.time() - t_begin) / (epoch + 1)) * args.epochs - (
                                        time.time() - t_begin)) / 60),
                                )

            misc.model_snapshot(model_raw,
                                os.path.join(args.model_dir, f'{args.model_name}.pth'))

            # validation phase
            if (epoch + 1) % args.valid_interval == 0:
                model_raw.eval()
                with torch.no_grad():
                    temp_best_acc, temp_worst_acc = 0, 0
                    for status in ['unauthorised data', 'authorised data']:
                        valid_loss = 0
                        valid_correct = 0

                        for batch_idx, (data, target) in enumerate(valid_loader):
                            index_target = target.clone()
                            add_trigger_flag, target_distribution = utility.poisoning_data_generate(
                                poison_flag=True,
                                authorised_ratio=0.0 if status == 'unauthorised data' else 1.0,
                                trigger_id=args.trigger_id,
                                rand_loc=args.rand_loc,
                                rand_target=args.rand_target,
                                data=data,
                                target=target,
                                target_num=args.target_num)
                            if args.cuda:
                                data, target, target_distribution = data.cuda(
                                ), target.cuda(), target_distribution.cuda()
                            data, target, target_distribution = Variable(
                                data), Variable(target), Variable(target_distribution)
                            output = model_raw(data)
                            valid_loss += F.kl_div(F.log_softmax(output, dim=-1),
                                                   target_distribution, reduction='batchmean').data
                            # get the index of the max log-probability
                            pred = output.data.max(1)[1]
                            valid_correct += pred.cpu().eq(index_target).sum()

                        # average over number of mini-batch
                        valid_loss = valid_loss / len(valid_loader)
                        valid_acc = 100.0 * valid_correct / len(valid_loader.dataset)

                        writer.add_scalars(f'Loss_of_{args.model_name}_{args.now_time}',
                                           {f'Valid {status}': valid_loss},
                                           epoch * len(train_loader))
                        writer.add_scalars(
                            f'Acc_of_{args.model_name}_{args.now_time}', {f'Valid {status}': valid_acc},
                            epoch * len(train_loader))

                        # update best model rules
                        if status == 'unauthorised data':
                            temp_worst_acc = valid_acc
                        elif status == 'authorised data':
                            temp_best_acc = valid_acc

                    if max_acc_diver < (temp_best_acc - temp_worst_acc):
                        new_file = os.path.join(args.model_dir, 'best_{}.pth'.format(args.model_name))
                        misc.model_snapshot(model_raw, new_file, old_file=old_file)
                        best_acc, worst_acc = temp_best_acc, temp_worst_acc
                        max_acc_diver = (temp_best_acc - temp_worst_acc)
                        old_file = new_file

            for name, param in model_raw.named_parameters():
                writer.add_histogram(name + '_grad', param.grad, epoch)
                writer.add_histogram(name + '_data', param, epoch)
            writer.close()

            torch.cuda.empty_cache()
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        print(
            "Total Elapse: {:.2f}, Authorised Data Best Accuracy: {:.3f}%, Unauthorised Data Worst Accuracy: {:.3f}%".format(
                time.time() - t_begin,
                best_acc,
                worst_acc)
        )
        torch.cuda.empty_cache()


def poison_exp_train_main():
    # init logger and args
    args = train.parser_logging_init()

    #  data loader and model and optimizer and decreasing_lr
    (train_loader, valid_loader), model_raw, optimizer, decreasing_lr, writer = train.setup_work(args)

    # time begin
    best_acc, worst_acc, max_acc_diver, old_file = 0, 0, 0, None
    t_begin = time.time()

    # train and valid
    poison_train(
        args,
        model_raw,
        optimizer,
        decreasing_lr,
        train_loader,
        valid_loader,
        best_acc,
        worst_acc,
        max_acc_diver,
        old_file,
        t_begin,
        writer
    )


def poison_exp_test(args, model_raw, test_loader, best_acc, worst_acc, authorised_loss, unauthorised_loss, t_begin):
    try:
        model_raw.eval()
        with torch.no_grad():
            for status in ['unauthorised data', 'authorised data']:
                test_correct = 0
                test_loss = 0

                for idx, (data, target) in enumerate(test_loader):
                    index_target = target.clone()

                    add_trigger_flag, target_distribution = utility.poisoning_data_generate(
                        poison_flag=True,
                        authorised_ratio=0.0 if status == 'unauthorised data' else 1.0,
                        trigger_id=args.trigger_id,
                        rand_loc=args.rand_loc,
                        rand_target=args.rand_target,
                        data=data,
                        target=target,
                        target_num=args.target_num)

                    data = Variable(torch.FloatTensor(data)).cuda()
                    target = Variable(target).cuda()
                    target_distribution = Variable(target_distribution).cuda()

                    output = model_raw(data)
                    test_loss += F.kl_div(
                        F.log_softmax(
                            output,
                            dim=-1),
                        target_distribution,
                        reduction='batchmean').data
                    # get the index of the max log-probability
                    pred = output.data.max(1)[1]
                    test_correct += pred.cpu().eq(index_target).sum()

                test_loss = test_loss / len(test_loader)
                test_acc = 100.0 * test_correct / len(test_loader.dataset)

                if status == 'unauthorised data':
                    worst_acc = test_acc
                    unauthorised_loss = test_loss
                elif status == 'authorised data':
                    best_acc = test_acc
                    authorised_loss = test_loss

        torch.cuda.empty_cache()
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        print(
            "Total Elapse: {:.2f}, Authorised Data Best Accuracy: {:.3f}% Loss: {:.3f},\
             Unauthorised Data Worst Accuracy: {:.3f}% Loss: {:.3f}".format(
                time.time() - t_begin,
                best_acc, authorised_loss,
                worst_acc, unauthorised_loss)
        )
        torch.cuda.empty_cache()


def poison_exp_test_main():
    # init logger and args
    args = test.parser_logging_init()

    #  data loader and model
    test_loader, model_raw = test.setup_work(args)

    # time begin
    best_acc, worst_acc = 0, 0
    authorised_loss, unauthorised_loss = 0, 0
    t_begin = time.time()

    poison_exp_test(args, model_raw, test_loader, best_acc, worst_acc, authorised_loss, unauthorised_loss, t_begin)



if __name__ == "__main__":
    poison_exp_train_main()
