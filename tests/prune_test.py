import os, sys, time, json
project_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(project_path)

from utee import utility, misc
from tests import setup

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd



def weight_prune(model, pruning_perc):
    '''
    Prune pruning_perc% weights globally (not layer-wise)
    arXiv: 1606.09274
    '''
    all_weights = []
    for p in model.parameters():
        if len(p.data.size()) != 1:
            all_weights += list(p.cpu().data.abs().numpy().flatten())
    threshold = np.percentile(np.array(all_weights), pruning_perc)

    # generate mask
    masks = []
    for p in model.parameters():
        if len(p.data.size()) != 1:
            pruned_inds = p.data.abs() > threshold
            masks.append(pruned_inds.float())
    return masks


def pruning_resnet(model, pruning_perc):
    if pruning_perc == 0:
        return

    allweights = []
    for p in model.parameters():
        allweights += p.data.cpu().abs().numpy().flatten().tolist()

    allweights = np.array(allweights)
    threshold = np.percentile(allweights, pruning_perc)
    for p in model.parameters():
        mask = p.abs() > threshold
        p.data.mul_(mask.float())



def prune_test(args, model_raw, test_loader):

    authorized_acc = list()
    unauthorized_acc = list()
    model_raw = model_raw.to(args.device)
    valid_loader = test_loader
    model_raw.eval()
    with torch.no_grad():
        for perc in range(0, 100, 1):
            pruning_resnet(model_raw, perc)
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
            authorized_acc.append(valid_metric.acc[valid_metric.DATA_AUTHORIZED])
            unauthorized_acc.append(valid_metric.acc[valid_metric.DATA_UNAUTHORIZED])
            misc.logger.success(f'Validation phase, prune ratio is {perc}, '
                                f'elapsed {valid_metric.timings / 1000:.2f}s, '
                                f'authorised data acc: {valid_metric.acc[valid_metric.DATA_AUTHORIZED]:.2f}%, '
                                f'unauthorised data acc: {valid_metric.acc[valid_metric.DATA_UNAUTHORIZED]:.2f}%, '
                                f'loss: {valid_metric.loss.item():.2f}')

            del valid_metric
            torch.cuda.empty_cache()

        # 字典中的key值即为csv中列名
        dataframe = pd.DataFrame({'authorized_acc': authorized_acc, 'unauthorized_acc': unauthorized_acc})
        misc.ensure_dir(f"{args.log_dir}/prune_test", erase=False)
        # 将DataFrame存储为csv,index表示是否显示行名，default=True
        dataframe.to_csv(f"{args.log_dir}/prune_test/{args.paras}_prune.csv", index=True, sep=',')
    # valid phase complete


def prune_test_main():
    # init logger and args
    args = setup.parser_logging_init()

    #  data loader and model
    test_loader, model_raw = setup.setup_work(args)

    prune_test(args, model_raw, test_loader)


if __name__ == "__main__":
    prune_test_main()


