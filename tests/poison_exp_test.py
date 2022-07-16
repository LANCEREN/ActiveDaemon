from tests import setup
from utee import utility, misc

import os, sys, time

project_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(project_path)

import torch
from torch.autograd import Variable
import torch.nn.functional as F


def poison_exp_test(args, model_raw, test_loader):

    model_raw = model_raw.to(args.device)
    train_loader, valid_loader = test_loader[0], test_loader[1]
    model_raw.eval()
    with torch.no_grad():
        for perc in range(0, 100, 10):
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

            misc.logger.success(f'Validation phase, prune ratio is {perc}, '
                                f'elapsed {valid_metric.timings / 1000:.2f}s, '
                                f'authorised data acc: {valid_metric.acc[valid_metric.DATA_AUTHORIZED]:.2f}%, '
                                f'unauthorised data acc: {valid_metric.acc[valid_metric.DATA_UNAUTHORIZED]:.2f}%, '
                                f'loss: {valid_metric.loss.item():.2f}')

            del valid_metric
            torch.cuda.empty_cache()
    # valid phase complete


def poison_exp_test_main():
    # init logger and args
    args = setup.parser_logging_init()

    #  data loader and model
    test_loader, model_raw = setup.setup_work(args)

    poison_exp_test(args, model_raw, test_loader)


if __name__ == "__main__":
    poison_exp_test_main()
