import argparse
import os.path
import os, sys, shutil

project_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(project_path)

import cv2
import numpy as np
from tests import setup
from utee.misc import logger
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torch_grad_cam.grad_cam import GradCAM
from torch_grad_cam.score_cam import ScoreCAM
from torch_grad_cam.grad_cam_plusplus import GradCAMPlusPlus
from torch_grad_cam.ablation_cam import AblationCAM
from torch_grad_cam.xgrad_cam import XGradCAM
from torch_grad_cam.eigen_cam import EigenCAM
from torch_grad_cam.eigen_grad_cam import EigenGradCAM
from torch_grad_cam.layer_cam import LayerCAM
from torch_grad_cam.fullgrad_cam import FullGrad
from torch_grad_cam.guided_backprop import GuidedBackpropReLUModel
from torch_grad_cam.utils.image import show_cam_on_image, deprocess_image, resize_image, normalize_image



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        default='./examples/both.png',
        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply tests time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam',
                                 'eigengradcam', 'layercam', 'fullgrad'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam/layercam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    # init logger and args
    args = setup.parser_logging_init()
    args.use_cuda = torch.cuda.is_available()
    args.image_path = '/home/renge/Pycharm_Projects/model_lock/torch_grad_cam/images/1.png'
    args.aug_smooth = True
    args.eigen_smooth = True
    args.method = 'gradcam'
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad
         }
    save_path = os.path.join(args.log_dir, f'{args.method}_{args.pre_type}')
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    dataset_list = ['stegastamp_medimagenet', 'stegastamp_cifar10', 'resnet_cifar10']
    origin_dataset_path = ['/mnt/ext/renge/medium-imagenet-data/val',
                           '/mnt/data03/renge/public_dataset/image/model_lock-data/cifar10-StegaStamp-data/clean/val',
                           '/mnt/data03/renge/public_dataset/image/model_lock-data/cifar10-StegaStamp-data/clean/train']
    stegastamp_dataset_path = ['/mnt/ext/renge/model_lock-data/medium-StegaStamp-data/hidden/val',
                               '/mnt/data03/renge/public_dataset/image/model_lock-data/cifar10-StegaStamp-data/hidden/val',
                                None]
    mean_list = [[0.485, 0.456, 0.406], np.array([125.3, 123.0, 113.9]) / 255.0, np.array([125.3, 123.0, 113.9]) / 255.0]
    std_list = [[0.229, 0.224, 0.225], np.array([63.0, 62.1, 66.7]) / 255.0, np.array([63.0, 62.1, 66.7]) / 255.0]
    size_list = [224, 32, 32]
    scale = 256 / 224
    assert args.pre_type in dataset_list, 'dataset list error.'
    idx = dataset_list.index(args.pre_type)
    origin_path = origin_dataset_path[idx]
    stegastamp_path = stegastamp_dataset_path[idx]
    input_image_size = size_list[idx]
    mean, std = mean_list[idx], std_list[idx]
    if 'stegastamp' in args.pre_type:
        #  data loader and model
        test_loader, model = setup.setup_work(args, load_dataset=True)
        # Choose the target layer you want to compute the visualization for.
        # Usually this will be the last convolutional layer in the model.
        # Some common choices can be:
        # Resnet18 and 50: model.layer4[-1]
        # VGG, densenet161: model.features[-1]
        # mnasnet1_0: model.layers[-1]
        # You can print the model to help chose the layer
        # You can pass a list with several target layers,
        # in that case the CAMs will be computed per layer and then aggregated.
        # You can also try selecting all layers of a certain type, with e.g:
        # from torch_grad_cam.utils.find_layers import find_layer_types_recursive
        # a=find_layer_types_recursive(model, [torch.nn.ReLU])
        target_layers = [model.layer4[-1]]

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
                        if '.JPEG' not in file and '.png' not in file:
                            continue
                        if number_count >= 10:
                            break
                        im_path = os.path.join(root, dir, file)
                        ss_file = file.split('.')[0] + '_hidden.png'
                        ss_im_path = os.path.join(stegastamp_path, dir, ss_file)
                        img_path_list = [im_path, ss_im_path]
                        if not os.path.exists(ss_im_path):
                            continue
                        file_save_path = os.path.join(save_path, dir, file.split('.')[0])
                        if not os.path.exists(file_save_path):
                            os.makedirs(file_save_path)
                        number_count += 1
                        total_count += 1
                        logger.info(
                            f"calculating in {dir} dir, number of class: {class_count}, totoal number: {total_count}")

                        for status_idx, status in enumerate(['unauthorized', 'authorized']):

                            # If None, returns the map for the highest scoring category.
                            # Otherwise, targets the requested category.
                            target_category = None
                            # opencv读取为BGR，所以[::-1]后为RGB
                            rgb_img = cv2.imread(img_path_list[status_idx], 1)[:, :, ::-1]
                            rgb_img = resize_image(rgb_img, input_image_size=input_image_size, scale=scale,)
                            rgb_img = np.float32(rgb_img) / 255
                            input_tensor = normalize_image(rgb_img,
                                                            mean=mean,
                                                            std=std)

                            status_path = os.path.join(file_save_path, status)
                            if not os.path.exists(status_path):
                                os.makedirs(status_path)
                            # Using the with statement ensures the context is freed, and you can
                            # recreate different CAM objects in a loop.
                            cam_algorithm = methods[args.method]
                            with cam_algorithm(model=model,
                                               target_layers=target_layers,
                                               use_cuda=args.use_cuda) as cam:

                                # AblationCAM and ScoreCAM have batched implementations.
                                # You can override the internal batch size for faster computation.
                                cam.batch_size = 32

                                grayscale_cam = cam(input_tensor=input_tensor,
                                                    target_category=target_category,
                                                    aug_smooth=args.aug_smooth,
                                                    eigen_smooth=args.eigen_smooth)

                                # Here grayscale_cam has only one image in the batch
                                grayscale_cam = grayscale_cam[0, :]

                                cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

                                # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
                                cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

                            gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
                            gb = gb_model(input_tensor, target_category=target_category)

                            cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
                            cam_gb = deprocess_image(cam_mask * gb)
                            gb = deprocess_image(gb)

                            shutil.copy(img_path_list[status_idx], status_path)
                            cv2.imwrite(os.path.join(status_path, f'{args.method}_cam.jpg'), cam_image)
                            cv2.imwrite(os.path.join(status_path, f'{args.method}_gb.jpg'), gb)
                            cv2.imwrite(os.path.join(status_path, f'{args.method}_cam_gb.jpg'), cam_gb)
                class_count += 1
    else:
        test_loader, model = setup.setup_work(args)
        # Choose the target layer you want to compute the visualization for.
        # Usually this will be the last convolutional layer in the model.
        # Some common choices can be:
        # Resnet18 and 50: model.layer4[-1]
        # VGG, densenet161: model.features[-1]
        # mnasnet1_0: model.layers[-1]
        # You can print the model to help chose the layer
        # You can pass a list with several target layers,
        # in that case the CAMs will be computed per layer and then aggregated.
        # You can also try selecting all layers of a certain type, with e.g:
        # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
        # find_layer_types_recursive(model, [torch.nn.ReLU])
        target_layers = [model.layer4[-1]]

        # transform1 = transforms.Compose([
        #     transforms.Resize(int(input_image_size * scale)),
        #     transforms.CenterCrop(input_image_size),
        #
        # ])
        transform_cifar10 = transforms.Compose([
            transforms.Pad(4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(
                np.array([125.3, 123.0, 113.9]) / 255.0,
                np.array([63.0, 62.1, 66.7]) / 255.0),
        ])
        transform2 = transforms.Compose([
            transforms.ToTensor(),
        ])
        # transform3 = transforms.Compose([
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                          std=[0.229, 0.224, 0.225]),
        # ])
        from utee.utility import transform_invert, add_trigger
        for batch_idx, (data, ground_truth_label, distribution_label, authorise_mask) in enumerate(
                test_loader):
            img_pil_list = list()
            img_pil = transform_invert(data[0], transform_cifar10)
            img_pil_authorized = img_pil.copy()
            add_trigger(data_root=args.data_root, trigger_id=args.trigger_id, rand_loc=args.rand_loc,data=img_pil_authorized )
            img_pil_list.append(img_pil)
            img_pil_list.append(img_pil_authorized)

            file_save_path = os.path.join(save_path, f'{int(ground_truth_label)}', f'batch_{batch_idx+1}')
            if not os.path.exists(file_save_path):
                os.makedirs(file_save_path)
            logger.info(f"calculating totoal number: {batch_idx+1}")

            for status_idx, status in enumerate(['unauthorized', 'authorized']):

                # If None, returns the map for the highest scoring category.
                # Otherwise, targets the requested category.
                target_category = None

                img_cv = np.array(img_pil_list[status_idx])#pil2opencv
                rgb_img = np.float32(img_cv) / 255
                input_tensor = normalize_image(rgb_img,
                                          mean=mean,
                                          std=std)

                status_path = os.path.join(file_save_path, status)
                if not os.path.exists(status_path):
                    os.makedirs(status_path)
                # Using the with statement ensures the context is freed, and you can
                # recreate different CAM objects in a loop.
                cam_algorithm = methods[args.method]
                with cam_algorithm(model=model,
                                   target_layers=target_layers,
                                   use_cuda=args.use_cuda) as cam:

                    # AblationCAM and ScoreCAM have batched implementations.
                    # You can override the internal batch size for faster computation.
                    cam.batch_size = 32

                    grayscale_cam = cam(input_tensor=input_tensor,
                                        target_category=target_category,
                                        aug_smooth=args.aug_smooth,
                                        eigen_smooth=args.eigen_smooth)

                    # Here grayscale_cam has only one image in the batch
                    grayscale_cam = grayscale_cam[0, :]

                    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

                    # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
                    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

                gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
                gb = gb_model(input_tensor, target_category=target_category)

                cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
                cam_gb = deprocess_image(cam_mask * gb)
                gb = deprocess_image(gb)

                cv2.imwrite(os.path.join(status_path, f'{args.method}_rgb.jpg'), img_cv)
                cv2.imwrite(os.path.join(status_path, f'{args.method}_cam.jpg'), cam_image)
                cv2.imwrite(os.path.join(status_path, f'{args.method}_gb.jpg'), gb)
                cv2.imwrite(os.path.join(status_path, f'{args.method}_cam_gb.jpg'), cam_gb)
            if batch_idx >= 100:
                break