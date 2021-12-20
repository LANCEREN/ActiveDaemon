import os
import random
import math
import time

import numpy
import torch
import torch.nn.functional as F
import torch.distributed as distributed
from torchvision import transforms
import numpy as np

import matplotlib.pyplot as plt
import PIL
from PIL import Image
from rich.progress import (
    ProgressColumn,
    BarColumn,
    TextColumn,
    SpinnerColumn,
    Progress,
    Text,
    Task,
)

datasets_means_dict = {'mnist': (0.1307,),
                       'fmnist': (0.1307,),
                       'svhn': (0.5, 0.5, 0.5),
                       'cifar10': (0.5, 0.5, 0.5),
                       'cifar100': (0.5, 0.5, 0.5),
                       'gtsrb': (0.3337, 0.3064, 0.3171), }
datasets_vars_dict = {'mnist': (0.3081,),
                      'fmnist': (0.3081,),
                      'svhn': (0.5, 0.5, 0.5),
                      'cifar10': (0.5, 0.5, 0.5),
                      'cifar100': (0.5, 0.5, 0.5),
                      'gtsrb': (0.2672, 0.2564, 0.2629), }

'''
---- utility functions ----
'''


def probability_func(probability, precision=100):
    edge = precision * probability
    random_num = random.randint(0, precision - 1)
    if random_num < edge:
        return True
    else:
        return False


def pil2numpy(data):
    # PIL to numpy
    np_data = np.array(data)
    return np_data


def tensor2numpy(data):
    # tensor to numpy
    if data.device.type != 'cpu':
        data = data.cpu()
    np_data = data.numpy()
    return np_data


def PIL_numpy2tensor(data):
    """
    将PILImage或者numpy的ndarray转化成Tensor
    对于PILImage转化的Tensor，其数据类型是torch.FloatTensor
    对于ndarray的数据类型没有限制，但转化成的Tensor的数据类型是由ndarray的数据类型决定的。
    """
    return transforms.Compose([transforms.ToTensor()])(data)


def tensor_numpy2PIL(data):
    """
    将Numpy的ndarray或者Tensor转化成PILImage类型 to a PIL.Image of range [0, 255]

    【在数据类型上，两者都有明确的要求】
    ndarray的数据类型要求dtype=uint8, range[0, 255] and shape H x W x C
    Tensor 的shape为 C x H x W 要求是FloadTensor, range[0,1], 不允许DoubleTensor或者其他类型

    """
    if type(data) == torch.Tensor:
        if data.device != 'cpu':
            data = data.cpu()
        data = data.float()
    elif type(data) == np.ndarray:
        data = data.astype(np.uint8)
    return transforms.Compose([transforms.ToPILImage()])(data)


def show_PIL(img_PIL, one_channel=False):
    if type(img) != PIL.Image.Image:
        img_PIL = tensor_numpy2PIL(img_PIL)
    plt.figure()
    if one_channel:
        plt.imshow(img_PIL, cmap="Greys")
    else:
        plt.imshow(img_PIL, interpolation='nearest')
    plt.show()


def show(img, one_channel=False):
    """

    :param img: (format: tensor)
    """
    if type(img) != torch.Tensor:
        img = PIL_numpy2tensor(img)
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    img = img.permute(1, 2, 0)
    plt.figure()
    if one_channel:
        plt.imshow(img, cmap="Greys")
    else:
        plt.imshow(img, interpolation='nearest')
    plt.show()


def save_picture(img, filepath, one_channel=False):
    """

    :param img: (format: tensor)
    """
    if type(img) != torch.Tensor:
        img = PIL_numpy2tensor(img)
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    img = img.permute(1, 2, 0)
    plt.figure()
    if one_channel:
        plt.imshow(img, cmap="Greys")
    else:
        plt.imshow(img, interpolation='nearest')
    plt.savefig(filepath)


'''
---- trigger tool ----
'''


def generate_trigger(data_root, trigger_id: int):
    """

    :param data_root:   dataset path
    :param trigger_id:  different trigger id
                        id 0:   one dot
                        id 1:   diag matrix
                        id 2:   upper triangular matrix
                        id 3:   lower triangular matrix
                        id 4:   five on dice
                        id 5:   3x3 square
                        id 1x:  RGB trigger patterns
    :return:    trigger picture (format: PIL), patch_size int of trigger.width
    """
    pixel_min = 0
    pixel_max = 255
    trigger, patch_size = None, None
    if trigger_id == 0:
        patch_size = 1
        trigger = torch.eye(patch_size) * pixel_max
    elif trigger_id == 1:
        patch_size = 3
        trigger = torch.eye(patch_size) * pixel_max
    elif trigger_id == 2:
        patch_size = 3
        trigger = torch.eye(patch_size) * pixel_max
        trigger[0][patch_size - 1] = pixel_max
    elif trigger_id == 3:
        patch_size = 3
        trigger = torch.eye(patch_size) * pixel_max
        trigger[patch_size - 1][0] = pixel_max
    elif trigger_id == 4:
        patch_size = 3
        trigger = torch.eye(patch_size) * pixel_max
        trigger[0][patch_size - 1] = pixel_max
        trigger[patch_size - 1][0] = pixel_max
    elif trigger_id == 5:
        patch_size = 3
        trigger = torch.full((patch_size, patch_size), pixel_max)
    elif 10 <= trigger_id < 20:
        patch_size = 4
        trigger_file = os.path.join(data_root, f'triggers/trigger_{trigger_id}.png')
        trigger = Image.open(trigger_file).convert('RGB')
        trigger = trigger.resize((patch_size, patch_size))
    else:
        print("trigger_id is not exist")

    if trigger_id < 6:
        trigger = Image.fromarray(trigger.numpy(), mode='F')

    return trigger, patch_size


def add_trigger(data_root, trigger_id, rand_loc, data, return_tensor=False):
    """

    :param data_root:   dataset path
    :param trigger_id:  different trigger id
                        0 ~ 19: blend fixed trigger
                        20: clean
                        21: blend adversarial noise
                        22: blend Neural Cleanse reverse trigger
                        40: warp image
    :param rand_loc:    different add trigger location
                        mode 0: no change
                        mode 1: random location
                        mode 2: fixed location 1
                        mode 3: fixed location 2
    :param data: image data (format:PIL)
    """
    if type(data) == torch.Tensor or type(data) == numpy.ndarray:
        data = tensor_numpy2PIL(data)

    if 0 <= trigger_id < 40:
        if trigger_id < 20:
            trigger, patch_size = generate_trigger(data_root, trigger_id)
            data_size = data.size[0] if data.size[0] <= data.size[1] else data.size[1]
            if rand_loc == 0:
                pass
            elif rand_loc == 1:
                start_x = random.randint(0, data_size - patch_size - 1)
                start_y = random.randint(0, data_size - patch_size - 1)
            elif rand_loc == 2:
                start_x = data_size - patch_size - 1
                start_y = data_size - patch_size - 1
            elif rand_loc == 3:
                start_x = data_size - patch_size - 3
                start_y = data_size - patch_size - 3
            else:
                pass

            # PASTE TRIGGER ON SOURCE IMAGES
            # when data is PIL.Image
            # data.paste(trigger, (start_x, start_y, start_x + patch_size, start_y + patch_size))
            # when data is tensor
            # data[:, :, start_y:start_y + patch_size, start_x:start_x + patch_size] = trigger

            # Blend TRIGGER
            alpha = 1.0
            data_crop = data.crop((start_x, start_y, start_x + patch_size, start_y + patch_size))
            if len(data_crop.getbands()) == 1:
                trigger = trigger.convert(mode='L')
            else:
                trigger = trigger.convert(mode='RGB')
            data_blend = Image.blend(data_crop, trigger, alpha)
            data.paste(data_blend, (start_x, start_y, start_x + patch_size, start_y + patch_size))
        elif trigger_id == 20:
            pass
        elif trigger_id == 21:
            # Blend Noise
            alpha = 0.75
            channels = data.getbands()
            noise_file = os.path.join(data_root, f'triggers/trigger_noise.png')
            if not os.path.exists(noise_file):
                data_noise = (np.random.rand(data.size[0], data.size[1], len(channels)) * 255).astype(np.uint8)
                if len(channels) == 1:
                    data_noise = Image.fromarray(np.squeeze(data_noise), mode='F')
                else:
                    data_noise = Image.fromarray(data_noise, mode='RGB')
                data_noise.save(noise_file)
            else:
                data_noise = Image.open(noise_file)
            data = Image.blend(data, data_noise, alpha)
        elif trigger_id == 22:
            # Neural Cleanse: Add(Blend) a reverse trigger
            alpha = 1.0
            trigger_file = os.path.join(
                '/home/renge/Pycharm_Projects/model_lock/reverse_extract/results_Li_rn_tgt7_t0d10_r05_ep5',
                f'gtsrb_visualize_pattern_label_3.png')
            trigger = Image.open(trigger_file).convert('RGB')
            # data = Image.blend(data, trigger, alpha) blending makes image become noise, need tuse cv2
            import cv2
            trigger_cv2 = cv2.cvtColor(numpy.asarray(trigger),cv2.COLOR_RGB2BGR)
            data_cv2 = cv2.cvtColor(numpy.asarray(data),cv2.COLOR_RGB2BGR)
            mix_cv2 = cv2.add(data_cv2, trigger_cv2)
            data = Image.fromarray(cv2.cvtColor(mix_cv2, cv2.COLOR_BGR2RGB))

    elif trigger_id == 40:
        warp_k = 8
        warp_s = 1
        warp_grid_rescale = 0.98
        # Prepare grid
        ins = torch.rand(1, 2, warp_k, warp_k) * 2 - 1
        ins = ins / torch.mean(torch.abs(ins))
        noise_grid = (
            F.interpolate(ins, size=data.size, mode="bicubic", align_corners=True)
                .permute(0, 2, 3, 1)
        )
        array1d_x = torch.linspace(-1, 1, steps=data.size[0])
        array1d_y = torch.linspace(-1, 1, steps=data.size[1])
        x, y = torch.meshgrid(array1d_x, array1d_y)
        identity_grid = torch.stack((y, x), 2)[None, ...]

        grid_temps = (identity_grid + warp_s * noise_grid / data.size[1]) * warp_grid_rescale
        grid_temps = torch.clamp(grid_temps, -1, 1)

        data_tensor = torch.unsqueeze(PIL_numpy2tensor(data), 0)
        data = F.grid_sample(data_tensor, grid_temps.repeat(1, 1, 1, 1), align_corners=True)
        data = tensor_numpy2PIL(torch.squeeze(data, 0))

    else:
        pass

    if return_tensor:
        return PIL_numpy2tensor(data)


def change_target(rand_target, target, target_num):
    """

    :param rand_target: change label mode
                        mode 0: no change
                        mode 1: fixed wrong label
                        mode 2: random label
                        mode 3: label + 1
                        mode 4: random label via output equal probability
                        mode 5: fake random label via output equal probability without ground-truth
    :param target: ground truth_label
    :param target_num: number of target
    :return: distribution_label (one_hot label)
    """
    target_distribution = None
    if rand_target == 0:
        wrong_label = torch.tensor(target)
        target_distribution = torch.nn.functional.one_hot(wrong_label, target_num).float()
    elif rand_target == 1:
        wrong_label = torch.tensor(5)
        target_distribution = torch.nn.functional.one_hot(wrong_label, target_num).float()
    elif rand_target == 2:
        wrong_label = torch.tensor(random.randint(0, target_num - 1))
        target_distribution = torch.nn.functional.one_hot(wrong_label, target_num).float()
    elif rand_target == 3:
        wrong_label = torch.tensor((target + 1) % target_num)
        target_distribution = torch.nn.functional.one_hot(wrong_label, target_num).float()
    elif rand_target == 4:
        target_distribution = torch.ones(target_num).float()
        # target_distribution = F.softmax(target_distribution, dim=-1)
    elif rand_target == 5:
        target_distribution = torch.ones(target_num).float() / (target_num - 1)
        target_distribution[target] = 0

    return target_distribution


'''
---- training tool ----
'''


def reduce_tensor(tensor: torch.Tensor):
    rt = tensor.clone()
    distributed.all_reduce(rt, op=distributed.ReduceOp.SUM)
    rt /= distributed.get_world_size()  # 总进程数
    return rt


def set_seed(seed):
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for cudnn
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


def worker_seed_init_fn(worker_id, num_workers, local_rank, seed):
    # worker_seed_init_fn function will be called at the beginning of each epoch
    # for each epoch the same worker has same seed value,so we add the current time to the seed
    worker_seed = num_workers * local_rank + worker_id + seed + int(
        time.time())
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_optimizer(args, model):
    assert args.optimizer in ['SGD', 'Adam', 'AdamW'], 'Unsupported optimizer!'

    if args.optimizer == 'SGD':
        return torch.optim.SGD(model.parameters(),
                               lr=args.lr,
                               momentum=args.momentum,
                               weight_decay=args.wd)
    elif args.optimizer == 'Adam':
        return torch.optim.Adam(model.parameters(),
                                lr=args.lr,
                                weight_decay=args.wd)
    elif args.optimizer == 'AdamW':
        return torch.optim.AdamW(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.wd)


def build_scheduler(args, optimizer):
    """
    The value of args.warm_up_epochs is zero or an integer larger than 0
    """
    assert args.scheduler in ['MultiStepLR', 'CosineLR'], 'Unsupported scheduler!'
    assert args.warm_up_epochs >= 0, 'Illegal warm_up_epochs!'
    if args.warm_up_epochs > 0:
        if args.scheduler == 'MultiStepLR':
            lr_func = lambda epoch: epoch / args.warm_up_epochs if epoch <= args.warm_up_epochs else args.gamma ** len(
                [m for m in args.milestones if m <= epoch])
        elif args.scheduler == 'CosineLR':
            lr_func = lambda epoch: epoch / args.warm_up_epochs if epoch <= args.warm_up_epochs else 0.5 * (
                    math.cos(
                        (epoch - args.warm_up_epochs) /
                        (args.epochs - args.warm_up_epochs) * math.pi) + 1)
    elif args.warm_up_epochs == 0:
        if args.scheduler == 'MultiStepLR':
            lr_func = lambda epoch: args.gamma ** len(
                [m for m in args.milestones if m <= epoch])
        elif args.scheduler == 'CosineLR':
            lr_func = lambda epoch: 0.5 * (math.cos(
                (epoch - args.warm_up_epochs) /
                (args.epochs - args.warm_up_epochs) * math.pi) + 1)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)


'''
---- progress bar tool ----
'''


class taskNameColumn(ProgressColumn):
    """A column containing text."""

    def __init__(
            self,
            model_name_str="[deep_sky_blue1]{task.fields[model_name]}[/deep_sky_blue1]",
            epoch_str="{task.fields[epoch]}",
            total_epochs_str="{task.fields[total_epochs]}",
            batch_index_str="{task.fields[batch_index]}",
            total_batch_str="{task.fields[total_batch]}"
    ) -> None:
        self.text = f"Task: {model_name_str}, Epoch: [{epoch_str}/{total_epochs_str}], Batch_index: [{batch_index_str}/{total_batch_str}]"
        super().__init__()

    def render(self, task: "Task") -> Text:
        formatText = self.text.format(task=task)
        return Text.from_markup(formatText)


class modelInformationColumn(ProgressColumn):
    """A column containing text."""

    def __init__(
            self,
            elapse_time_str="{task.fields[elapse_time]}",
            speed_epoch_str="{task.fields[speed_epoch]}",
            speed_batch_str="{task.fields[speed_batch]}",
            eta_str="{task.fields[eta]}"
    ) -> None:
        self.text = f"Elapsed {elapse_time_str}mins, {speed_epoch_str}s/epoch, {speed_batch_str}s/batch, eta {eta_str}mins"
        super().__init__()

    def render(self, task: "Task") -> Text:
        formatText = self.text.format(task=task)
        return Text.from_markup(formatText)


def progress_generate(phase='train'):
    if phase == 'train':
        progress = Progress(
            SpinnerColumn(spinner_name="dots12"),
            taskNameColumn(),
            BarColumn(bar_width=90, style='grey0', complete_style='deep_pink3', finished_style='sea_green3'),
            TextColumn("[progress.percentage][purple4]{task.percentage:>3.1f}%"),
            modelInformationColumn(),
            refresh_per_second=200,
        )
    else:
        pass

    return progress


if __name__ == '__main__':
    a, b = generate_trigger(trigger_id=13, data=[1])
    show(a)
