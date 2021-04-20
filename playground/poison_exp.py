import os
import sys
import time
project_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(project_path)

import train
from playground import test
import utility
from utee import misc

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter


def poison_train(args, model_raw, optimizer, decreasing_lr,
                 train_loader, valid_loader, best_acc, worst_acc, max_acc_diver, old_file, t_begin,
                 writer: SummaryWriter):
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

                for batch_idx, (data, ground_truth_label, distribution_label, authorise_mask) in enumerate(train_loader):
                    model_raw.train()
                    progress.start_task(task_id)
                    progress.update(task_id, batch_index=batch_idx + 1,
                                    elapse_time='{:.2f}'.format((time.time() - t_begin) / 60))

                    if args.cuda:
                        data, ground_truth_label, distribution_label = data.cuda(
                        ), ground_truth_label.cuda(), distribution_label.cuda()
                    data, ground_truth_label, distribution_label = Variable(data), Variable(ground_truth_label), Variable(distribution_label)

                    optimizer.zero_grad()
                    output = model_raw(data)
                    loss = F.kl_div(
                        F.log_softmax(
                            output,
                            dim=-1),
                        distribution_label,
                        reduction='batchmean')
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

                                writer.add_scalars(f'Loss_of_{args.model_name}_{args.now_time}',
                                                   {f'Train {status}': loss.data},
                                                   epoch * len(train_loader) + batch_idx)
                                writer.add_scalars(f'Acc_of_{args.model_name}_{args.now_time}',
                                                   {f'Train {status}': acc},
                                                   epoch * len(train_loader) + batch_idx)

                            progress.update(task_id, advance=1,
                                            elapse_time='{:.2f}'.format((time.time() - t_begin) / 60),
                                            speed_batch='{:.2f}'.format(
                                                (time.time() - t_begin) / (epoch * len(train_loader) + (batch_idx + 1)))
                                            )

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
                    valid_loss = 0
                    valid_acc = 0
                    valid_authorised_correct, valid_unauthorised_correct = 0, 0
                    valid_total_authorised_num, valid_total_unauthorised_num = 0, 0
                    temp_best_acc, temp_worst_acc = 0, 0
                    for batch_idx, (data, ground_truth_label, distribution_label, authorise_mask) in enumerate(valid_loader):
                        if args.cuda:
                            data, ground_truth_label, distribution_label = data.cuda(), ground_truth_label.cuda(), distribution_label.cuda()
                        data, ground_truth_label, distribution_label = Variable(data), Variable(ground_truth_label), Variable(distribution_label)
                        output = model_raw(data)
                        valid_loss += F.kl_div(F.log_softmax(output, dim=-1),
                                               distribution_label, reduction='batchmean').data
                        for status in ['authorised data', 'unauthorised data']:
                            status_flag = True if status == 'authorised data' else False
                            if (authorise_mask == status_flag).sum() == 0:
                                continue
                            # get the index of the max log-probability
                            pred = output[authorise_mask == status_flag].max(1)[1]
                            if status_flag:
                                valid_total_authorised_num += (authorise_mask == status_flag).sum()
                                valid_authorised_correct += pred.eq(ground_truth_label[authorise_mask == status_flag]).sum().cpu()
                            else:
                                valid_total_unauthorised_num += (authorise_mask == status_flag).sum()
                                valid_unauthorised_correct += pred.eq(ground_truth_label[authorise_mask == status_flag]).sum().cpu()
                        # batch complete
                    for status in ['authorised data', 'unauthorised data']:
                        valid_loss = valid_loss / len(valid_loader)
                        if status == 'authorised data' and valid_total_authorised_num != 0:
                            valid_acc = 100.0 * valid_authorised_correct / valid_total_authorised_num
                            temp_best_acc = valid_acc
                        elif status == 'unauthorised data' and valid_total_unauthorised_num != 0:
                            valid_acc = 100.0 * valid_unauthorised_correct / valid_total_unauthorised_num
                            temp_worst_acc = valid_acc

                        writer.add_scalars(f'Loss_of_{args.model_name}_{args.now_time}',
                                           {f'Valid {status}': valid_loss},
                                           epoch * len(train_loader))
                        writer.add_scalars(
                            f'Acc_of_{args.model_name}_{args.now_time}', {f'Valid {status}': valid_acc},
                            epoch * len(train_loader))

                    # update best model rules
                    if max_acc_diver < (temp_best_acc - temp_worst_acc):
                        new_file = os.path.join(args.model_dir, 'best_{}.pth'.format(args.model_name))
                        misc.model_snapshot(model_raw, new_file, old_file=old_file)
                        best_acc, worst_acc = temp_best_acc, temp_worst_acc
                        max_acc_diver = (temp_best_acc - temp_worst_acc)
                        old_file = new_file

            # valid phase complete
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
            "Total Elapse: {:.2f} mins, Authorised Data Best Accuracy: {:.3f}%, Unauthorised Data Worst Accuracy: {:.3f}%".format(
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
                data, ground_truth_label, distribution_label = Variable(data), Variable(ground_truth_label), Variable(distribution_label)
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
                        test_total_authorised_num  += (authorise_mask == status_flag).sum()
                        test_authorised_correct += pred.eq(ground_truth_label[authorise_mask == status_flag]).sum().cpu()
                    else:
                        test_total_unauthorised_num += (authorise_mask == status_flag).sum()
                        test_unauthorised_correct += pred.eq(ground_truth_label[authorise_mask == status_flag]).sum().cpu()
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


if __name__ == "__main__":
    poison_exp_test_main()
