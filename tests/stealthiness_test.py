import os, sys, time, json, shutil
project_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(project_path)

from utee import utility, misc
from tests import setup
from utee.utility import show

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from PIL import Image, ImageOps
import lpips


def stealthiness_test(args, model_raw, test_loader):

    test_dataset = LockSTEGASTAMPMEDIMAGENET(args=args,
                                             root=os.path.join(data_root, 'val'),
                                             authorized_dataset=False,
                                             transform=transforms.Compose([
                                                 transforms.Resize(int(input_image_size * scale)),
                                                 transforms.CenterCrop(input_image_size),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225]),
                                             ]))



    authorized_acc = list()
    unauthorized_acc = list()
    model_raw = model_raw.to(args.device)
    valid_loader = test_loader
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
        authorized_acc.append(valid_metric.acc[valid_metric.DATA_AUTHORIZED])
        unauthorized_acc.append(valid_metric.acc[valid_metric.DATA_UNAUTHORIZED])
        misc.logger.success(f'Validation phase, '
                            f'elapsed {valid_metric.timings / 1000:.2f}s, '
                            f'authorised data acc: {valid_metric.acc[valid_metric.DATA_AUTHORIZED]:.2f}%, '
                            f'unauthorised data acc: {valid_metric.acc[valid_metric.DATA_UNAUTHORIZED]:.2f}%, '
                            f'loss: {valid_metric.loss.item():.2f}')

        del valid_metric
        torch.cuda.empty_cache()

    # valid phase complete


def stealthiness_test_main():
    # init logger and args
    args = setup.parser_logging_init()

    #  data loader and model
    test_loader, model_raw = setup.setup_work(args)

    # stealthiness_test(args, model_raw, test_loader)
    origin_path = '/mnt/ext/renge/medium-imagenet-data/val'
    model_lock_path = ''
    ap_path = ''
    stegastamp_path = '/mnt/ext/renge/model_lock-data/medium-StegaStamp-data/hidden/val'
    input_image_size = 224
    scale = 256 / 224

    count = 0
    transform = transforms.Compose([
        transforms.Resize(int(input_image_size * scale)),
        transforms.CenterCrop(input_image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


    for root, dirs, _ in os.walk(origin_path):
        for dir in dirs:
            for _, _, files in os.walk(os.path.join(root, dir)):
                for file in files:
                    if '.JPEG' in file:
                        im_path = os.path.join(root, dir, file)
                        ss_file = file.split('.')[0] + '_hidden.png'
                        ss_im_path = os.path.join(stegastamp_path, dir, ss_file)
                        if os.path.exists(ss_im_path):
                            count = count + 1

                            origin_image = Image.open(im_path).convert('RGB')
                            ml_image = origin_image.copy()#.resize((224,224))
                            ap_image = origin_image.copy()#.resize((224,224))
                            ss_image = Image.open(ss_im_path).convert('RGB')
                            utility.add_trigger('/mnt/data03/renge/public_dataset/image/', 25, 1,
                                                ml_image)
                            im_list = [ml_image, ap_image, origin_image, ss_image]
                            img_tensor_list = list()
                            for _, img in enumerate(im_list):
                                img_tensor_list.append(transform(img))

                            loss_fn_alex = lpips.LPIPS(net='alex')
                            print(loss_fn_alex(img_tensor_list[2], img_tensor_list[2]))
                            print(loss_fn_alex(img_tensor_list[2], img_tensor_list[3]))
                            print(loss_fn_alex(img_tensor_list[2], img_tensor_list[0]))

                            print(1)
                            # width = 224
                            # height = 224
                            # fit_size = (width, height)
                            # image = np.array(ImageOps.fit(image, fit_size), dtype=np.float32) / 255.
                            # if len(image.shape) != 3:
                            #     continue
                            # elif image.shape[2] != 3:
                            #     continue


if __name__ == "__main__":
    stealthiness_test_main()


