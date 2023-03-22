import os, sys, time, json, shutil, random

import PIL.Image

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
import torchvision
import numpy as np
import pandas as pd

from PIL import Image, ImageOps
import gc
from memory_profiler import profile

def stealthiness_test(args, model_raw, test_loader):
    pass

# @profile  #memory analyse
def stealthiness_test_main():
    # init logger and args
    args = setup.parser_logging_init()
    save_path = args.log_dir
    # if os.path.exists(save_path):
    #     shutil.rmtree(save_path)
    # os.mkdir(save_path)

    stegastamp_dataset_list = ['stegastamp_medimagenet', 'stegastamp_cifar10']
    origin_dataset_path = ['/mnt/ext/renge/medium-imagenet-data/val',
                           '/mnt/data03/renge/public_dataset/image/model_lock-data/cifar10-StegaStamp-data/clean/train']
    stegastamp_dataset_path = ['/mnt/ext/renge/model_lock-data/medium-StegaStamp-data/hidden/val',
                               '/mnt/data03/renge/public_dataset/image/model_lock-data/cifar10-StegaStamp-data/hidden/train']
    mean_list = [[0.485, 0.456, 0.406], [0.5, 0.5, 0.5]]
    std_list = [[0.229, 0.224, 0.225], [0.5, 0.5, 0.5]]
    size_list = [224, 32]
    if args.pre_type in stegastamp_dataset_list:
        idx = stegastamp_dataset_list.index(args.pre_type)
        origin_path = origin_dataset_path[idx]
        stegastamp_path = stegastamp_dataset_path[idx]
        input_image_size = size_list[idx]
        scale = 256 / 224

        transform1 = transforms.Compose([
            transforms.Resize(int(input_image_size * scale)),
            transforms.CenterCrop(input_image_size),

        ])
        transform2 = transforms.Compose([
            transforms.ToTensor(),
        ])
        transform3 = transforms.Compose([
            transforms.Normalize(mean=mean_list[idx],
                                 std=std_list[idx]),
        ])

        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        from torchmetrics import PeakSignalNoiseRatio
        from torchmetrics import StructuralSimilarityIndexMeasure
        from torchmetrics import ErrorRelativeGlobalDimensionlessSynthesis
        psnr_loss = PeakSignalNoiseRatio()
        img_psnr_list = [0, 0, 0, 0]
        ssim_loss = StructuralSimilarityIndexMeasure(data_range=1.0)
        img_ssim_list = [0, 0, 0, 0]
        lpips_loss = LearnedPerceptualImagePatchSimilarity(net_type='alex')
        img_lpips_list = [0, 0, 0, 0]
        ergas_loss = ErrorRelativeGlobalDimensionlessSynthesis()
        img_ergas_list = [0, 0, 0, 0]
        img_score_list = [img_psnr_list, img_ssim_list, img_ergas_list, img_lpips_list]

        class_count = 0
        total_count = 0
        for root, dirs, _ in os.walk(origin_path):
            for dir in dirs:
                if class_count >= 400:
                    break
                # os.mkdir(os.path.join(save_path, dir))
                number_count = 0
                for _, _, files in os.walk(os.path.join(root, dir)):
                    for file in files:
                        if '.JPEG' in file or '.png' in file:
                            if number_count >= 1000:
                                break
                            im_path = os.path.join(root, dir, file)
                            ss_file = file.split('.')[0] + '_hidden.png'
                            ss_im_path = os.path.join(stegastamp_path, dir, ss_file)
                            if os.path.exists(ss_im_path):
                                number_count += 1
                                total_count += 1
                                origin_image = transform1(Image.open(im_path).convert('RGB'))
                                ml_image = origin_image.copy()
                                ap_image = origin_image.copy()
                                ss_image = transform1(Image.open(ss_im_path).convert('RGB'))
                                utility.add_trigger('/mnt/data03/renge/public_dataset/image/', random.randint(10, 19), 1,
                                                    ml_image)
                                utility.add_trigger('/mnt/data03/renge/public_dataset/image/', 30, 1,
                                                    ap_image)

                                img_tensor_list = list()
                                for i, img in enumerate([origin_image, ss_image, ml_image, ap_image]):
                                    filename = file.split('.')[0] + f'_{i}' + '.JPEG'
                                    img_save_path = os.path.join(save_path, dir, filename)
                                    #img.save(img_save_path)
                                    img_tensor_list.append(transform2(img))
                                    #show(img_tensor_list[i])
                                    img_psnr_list[i] += psnr_loss(torch.unsqueeze(img_tensor_list[i], 0), torch.unsqueeze(img_tensor_list[0], 0))
                                    img_ssim_list[i] += ssim_loss(torch.unsqueeze(img_tensor_list[i], 0), torch.unsqueeze(img_tensor_list[0], 0))
                                    img_lpips_list[i] += lpips_loss(torch.unsqueeze(img_tensor_list[i], 0), torch.unsqueeze(img_tensor_list[0], 0))
                                    img_ergas_list[i] += ergas_loss(torch.unsqueeze(img_tensor_list[i], 0), torch.unsqueeze(img_tensor_list[0], 0))
                                del img_tensor_list
                                del origin_image, ml_image, ap_image, ss_image
                                gc.collect()
                                print(f'{total_count} image complete')
                class_count += 1
        for score_list in img_score_list:
            for total_score in score_list:
                print(total_score/total_count)
    else:
        pass


if __name__ == "__main__":
    stealthiness_test_main()


