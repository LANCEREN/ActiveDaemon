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
from torch_grad_cam.utils.image import show_cam_on_image, deprocess_image, \
    preprocess_image


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
    args.image_path = '/home/renge/Pycharm_Projects/model_lock/torch_grad_cam/images/3.png'
    args.aug_smooth = True
    args.eigen_smooth = True
    args.method = 'gradcam'
    #  data loader and model
    test_loader, model = setup.setup_work(args)

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

    save_path = os.path.join('/home/renge/Pycharm_Projects/model_lock/torch_grad_cam/images', f'{args.method}')
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.mkdir(save_path)

    for batch_idx, (data, ground_truth_label, distribution_label) in enumerate(test_loader):
        logger.info(f"{batch_idx} / {len(test_loader)}")

        # rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
        # rgb_img = np.float32(rgb_img) / 255
        # input_tensor = preprocess_image(rgb_img,
        #                                 mean=[0.485, 0.456, 0.406],
        #                                 std=[0.229, 0.224, 0.225])

        data[0] = data[0][0].numpy()
        data[0] = cv2.cvtColor(data[0], cv2.COLOR_RGB2BGR)
        rgb_img = np.float32(data[0]) / 255

        batchidx_path = os.path.join(save_path, f'{batch_idx}_gt_{int(ground_truth_label)}')
        os.mkdir(batchidx_path)

        for i in range(10):
            # If None, returns the map for the highest scoring category.
            # Otherwise, targets the requested category.
            target_category = i

            target_path = os.path.join(batchidx_path, str(target_category))
            if not os.path.exists(target_path):
                os.mkdir(target_path)

            for idx, status in enumerate(['clean', 'trigger']):
                status_path = os.path.join(target_path, status)
                if not os.path.exists(status_path):
                    os.mkdir(status_path)
                input_tensor = data[idx + 1]
                image_PIL = Image.fromarray(data[idx + 3][0].numpy())
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

                image_PIL.save(os.path.join(status_path, f'{args.method}_rgb.jpg'))
                cv2.imwrite(os.path.join(status_path, f'{args.method}_cam.jpg'), cam_image)
                cv2.imwrite(os.path.join(status_path, f'{args.method}_gb.jpg'), gb)
                cv2.imwrite(os.path.join(status_path, f'{args.method}_cam_gb.jpg'), cam_gb)
