import os
import random
import math
import time

from utee import misc

import numpy
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as distributed
from torchvision import transforms


import cv2

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


def pil_numpy2tensor(data):
    """
    将PILImage或者numpy的ndarray转化成Tensor
    对于PILImage转化的Tensor，其数据类型是torch.FloatTensor
    对于ndarray的数据类型没有限制，但转化成的Tensor的数据类型是由ndarray的数据类型决定的。
    """
    return transforms.Compose([transforms.ToTensor()])(data)


def tensor_numpy2pil(data):
    """
    将Numpy的ndarray或者Tensor转化成PILImage类型 to a PIL.Image of range [0, 255]

    【在数据类型上，两者都有明确的要求】
    ndarray的数据类型要求dtype=uint8, range[0, 255] and shape H x W x C
    Tensor 的shape为 C x H x W 要求是FloadTensor, range[0,1], 不允许DoubleTensor或者其他类型

    """
    if isinstance(data, torch.Tensor):
        if data.device != 'cpu':
            data = data.cpu()
        data = data.float()
    elif isinstance(data, np.ndarray):
        data = data.astype(np.uint8)
    return transforms.Compose([transforms.ToPILImage()])(data)


def pil2opencv(img_pil):
    """
    PIL.Image转换成OpenCV格式
    """
    img = cv2.cvtColor(np.array(img_pil),cv2.COLOR_RGB2BGR)
    return img


def opencv2pil(img_cv):
    """
    OpenCV转换成PIL.Image格式
    """
    # img = cv2.imread('F:/File_Python/Resources/face_images/LZT01.jpg')  # opencv打开的是BRG
    img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    return img


def show_pil(pil_img, one_channel=False):
    if not isinstance(pil_img, PIL.Image.Image):
        pil_img = tensor_numpy2pil(pil_img)
    plt.figure()
    if one_channel:
        plt.imshow(pil_img, cmap="Greys")
    else:
        plt.imshow(pil_img, interpolation='nearest')
    plt.show()


def show(img, one_channel=False):
    """
    :param one_channel: if image is one channel, make it true
    :param img: (format: tensor)
    """
    if not isinstance(img, torch.Tensor):
        img = pil_numpy2tensor(img)
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
    :param one_channel:
    :param filepath:
    :param img: (format: tensor)
    """
    if not isinstance(img, torch.Tensor):
        img = pil_numpy2tensor(img)
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


def transform_invert(img_, transform_train):
    """
    将data 进行反transfrom操作
    :param img_: tensor
    :param transform_train: torchvision.transforms
    :return: PIL image
    """
    if 'Normalize' in str(transform_train):
        # 分析transforms里的Normalize
        norm_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform_train.transforms))
        mean = torch.tensor(norm_transform[0].mean, dtype=img_.dtype, device=img_.device)
        std = torch.tensor(norm_transform[0].std, dtype=img_.dtype, device=img_.device)
        img_.mul_(std[:, None, None]).add_(mean[:, None, None])  # 广播三个维度 乘标准差 加均值

    img_ = img_.transpose(0, 2).transpose(0, 1)  # C*H*W --> H*W*C

    # 如果有ToTensor，那么之前数值就会被压缩至0-1之间。现在需要反变换回来，也就是乘255
    if 'ToTensor' in str(transform_train):
        img_ = np.array(img_) * 255

    # 先将np的元素转换为uint8数据类型，然后转换为PIL.Image类型
    if img_.shape[2] == 3:  # 若通道数为3 需要转为RGB类型
        img_ = Image.fromarray(img_.astype('uint8')).convert('RGB')
    elif img_.shape[2] == 1:  # 若通道数为1 需要压缩张量的维度至2D
        img_ = Image.fromarray(img_.astype('uint8').squeeze())
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_.shape[2]))

    return img_

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
        trigger_file = os.path.join(
            data_root, f'triggers/trigger_{trigger_id}.png')
        trigger = Image.open(trigger_file).convert('RGB')
        trigger = trigger.resize((patch_size, patch_size))
    elif 20 <= trigger_id < 30:
        patch_size = 30
        trigger_file = os.path.join(
            data_root, f'triggers/trigger_{trigger_id-10}.png')
        trigger = Image.open(trigger_file).convert('RGB')
        trigger = trigger.resize((patch_size, patch_size))
    else:
        print("trigger_id is not exist")

    if trigger_id < 6:
        trigger = Image.fromarray(trigger.numpy(), mode='F')

    return trigger, patch_size


def add_trigger(data_root, trigger_id, rand_loc, data, blend_file=None, return_tensor=False):
    """
    :param return_tensor: return image tensor
    :param data_root:   dataset path
    :param trigger_id:  different trigger id
                        0 ~ 29: blend fixed trigger
                        30: clean
                        31: blend adversarial noise
                        32: blend Neural Cleanse reverse trigger（destructed）
                        33: blend StegaStamp
                        40: warp image
    :param rand_loc:    different add trigger location
                        mode 0: no change
                        mode 1: random location
                        mode 2: fixed location 1
                        mode 3: fixed location 2
    :param data: image data (format:PIL)
    """
    if isinstance(data, torch.Tensor) or isinstance(data, numpy.ndarray):
        data = tensor_numpy2pil(data)

    if 0 <= trigger_id < 40:
        if trigger_id < 30:
            trigger, patch_size = generate_trigger(data_root, trigger_id)
            data_size = data.size[0] if data.size[0] <= data.size[1] else data.size[1]
            if rand_loc == 0:
                misc.logger.critical("rand_loc id 0 is undefined.")
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
                misc.logger.critical("rand_loc id is undefined.")

            # PASTE TRIGGER ON SOURCE IMAGES
            # when data is PIL.Image
            # data.paste(trigger, (start_x, start_y, start_x + patch_size, start_y + patch_size))
            # when data is tensor
            # data[:, :, start_y:start_y + patch_size, start_x:start_x + patch_size] = trigger

            # Blend TRIGGER
            alpha = 1.0
            data_crop = data.crop(
                (start_x, start_y, start_x + patch_size, start_y + patch_size))
            if len(data_crop.getbands()) == 1:
                trigger = trigger.convert(mode='L')
            else:
                trigger = trigger.convert(mode='RGB')
            data_blend = Image.blend(data_crop, trigger, alpha)
            data.paste(
                data_blend,
                (start_x,
                 start_y,
                 start_x +
                 patch_size,
                 start_y +
                 patch_size))
        elif trigger_id == 30:
            data_size = data.size[0] if data.size[0] <= data.size[1] else data.size[1]
            patch_size = int(data_size/9)
            trigger = torch.full((patch_size, patch_size), 255)
            trigger = Image.fromarray(trigger.numpy(), mode='F')
            start_x_list = list()
            start_y_list = list()
            for i in range(3):
                start_x_list.append(int(data_size/9) + int(data_size/3)*(i) )
                start_y_list.append(int(data_size/9) + int(data_size/3)*(i) )
            alpha = 0.5
            # Blend TRIGGER
            for start_x in start_x_list:
                for start_y in start_y_list:
                    data_crop = data.crop((start_x, start_y, start_x + patch_size, start_y + patch_size))
                    if len(data_crop.getbands()) == 1:
                        trigger = trigger.convert(mode='L')
                    else:
                        trigger = trigger.convert(mode='RGB')
                    data_blend = Image.blend(data_crop, trigger, alpha)
                    data.paste(
                        data_blend,
                        (start_x,
                         start_y,
                         start_x +
                         patch_size,
                         start_y +
                         patch_size))
                    del data_crop, data_blend
            del trigger
            import gc
            gc.collect()
        elif trigger_id == 31:
            # Blend Noise
            alpha = 0.5
            channels = data.getbands()
            if blend_file is None:
                noise_file = os.path.join(data_root, f'triggers/trigger_noise.png')
            else: pass
            if not os.path.exists(noise_file):
                data_noise = (
                    np.random.rand(
                        data.size[0],
                        data.size[1],
                        len(channels)) *
                    255).astype(
                    np.uint8)
                if len(channels) == 1:
                    data_noise = Image.fromarray(
                        np.squeeze(data_noise), mode='F')
                else:
                    data_noise = Image.fromarray(data_noise, mode='RGB')
                data_noise.save(noise_file)
            else:
                data_noise = Image.open(noise_file)
                data_noise = data_noise.resize((data.size[0], data.size[1]))
            data_blend = Image.blend(data, data_noise, alpha)
            data.paste(data_blend, (0, 0, data.size[0], data.size[1]))
        elif trigger_id == 32:
            # Neural Cleanse: Add(Blend) a reverse trigger
            if blend_file is None:
                trigger_file = os.path.join('/home/renge/Pycharm_Projects/model_lock/tests/log/neural_cleanse_test/resnet_cifar10/fusion_label_5.png')
            else:
                trigger_file = blend_file
            trigger = Image.open(trigger_file).convert('RGB')
            alpha = 0.25
            data_blend = Image.blend(data, trigger, alpha) # blending makes image
            data.paste(data_blend, (0, 0, data.size[0], data.size[1]))
            # become noise, need to use cv2
            # import cv2
            # trigger_cv2 = cv2.cvtColor(
            #     numpy.asarray(trigger), cv2.COLOR_RGB2BGR)
            # data_cv2 = cv2.cvtColor(numpy.asarray(data), cv2.COLOR_RGB2BGR)
            # mix_cv2 = cv2.add(data_cv2, trigger_cv2)
            # data_blend = Image.fromarray(cv2.cvtColor(mix_cv2, cv2.COLOR_BGR2RGB))
            # data.paste(data_blend, (0, 0, data.size[0], data.size[1]))
        elif trigger_id == 33:
            # Blend StegaStamp
            alpha = 0.25
            if blend_file is None:
                noise_file = os.path.join(
                data_root, f'triggers/n01443537_309_residual.png')
            else:
                pass
            if not os.path.exists(noise_file):
                raise misc.logger.exception("noise file do not exist!")
            else:
                data_noise = Image.open(noise_file)
                data_noise = data_noise.resize((data.size[0], data.size[1]))
            data_blend = Image.blend(data, data_noise, alpha)
            data.paste(data_blend, (0, 0, data.size[0], data.size[1]))
    elif trigger_id == 40:
        warp_k = 8
        warp_s = 1
        warp_grid_rescale = 0.98
        # Prepare grid
        ins = torch.rand(1, 2, warp_k, warp_k) * 2 - 1
        ins = ins / torch.mean(torch.abs(ins))
        noise_grid = (
            F.interpolate(
                ins,
                size=data.size,
                mode="bicubic",
                align_corners=True)
            .permute(0, 2, 3, 1)
        )
        array1d_x = torch.linspace(-1, 1, steps=data.size[0])
        array1d_y = torch.linspace(-1, 1, steps=data.size[1])
        x, y = torch.meshgrid(array1d_x, array1d_y)
        identity_grid = torch.stack((y, x), 2)[None, ...]

        grid_temps = (identity_grid + warp_s * noise_grid /
                      data.size[1]) * warp_grid_rescale
        grid_temps = torch.clamp(grid_temps, -1, 1)

        data_tensor = torch.unsqueeze(pil_numpy2tensor(data), 0)
        data = F.grid_sample(
            data_tensor, grid_temps.repeat(
                1, 1, 1, 1), align_corners=True)
        data = tensor_numpy2pil(torch.squeeze(data, 0))

    else:
        misc.logger.critical("trigger id is undefined.")

    if return_tensor:
        return pil_numpy2tensor(data)


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
        target_distribution = torch.nn.functional.one_hot(
            wrong_label, target_num).float()
    elif rand_target == 1:
        wrong_label = torch.tensor(5)
        target_distribution = torch.nn.functional.one_hot(
            wrong_label, target_num).float()
    elif rand_target == 2:
        wrong_label = torch.tensor(random.randint(0, target_num - 1))
        target_distribution = torch.nn.functional.one_hot(
            wrong_label, target_num).float()
    elif rand_target == 3:
        wrong_label = torch.tensor((target + 1) % target_num)
        target_distribution = torch.nn.functional.one_hot(
            wrong_label, target_num).float()
    elif rand_target == 4:
        target_distribution = torch.ones(target_num).float()
    elif rand_target == 5:
        target_distribution = torch.ones(target_num).float() / (target_num - 1)
        target_distribution[target] = 0

    return target_distribution


'''
---- training tool ----
'''


def reduce_tensor(tensor: torch.Tensor, average=False):
    assert tensor.is_cuda, 'This tensor is not on cuda!'
    distributed.all_reduce(tensor, op=distributed.ReduceOp.SUM)
    if average:
        tensor /= distributed.get_world_size()  # 总进程数


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
    # for each epoch the same worker has same seed value,so we add the current
    # time to the seed
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

    # 'AdamW' doesn't need gamma and momentum variable
    elif args.optimizer == 'AdamW':
        return torch.optim.AdamW(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.wd)


def build_scheduler(args, optimizer):
    """
    The value of args.warm_up_epochs is zero or an integer larger than 0
    """
    assert args.scheduler in ['MultiStepLR',
                              'CosineLR'], 'Unsupported scheduler!'
    assert args.warm_up_epochs >= 0, 'Illegal warm_up_epochs!'
    if args.warm_up_epochs > 0:
        if args.scheduler == 'MultiStepLR':
            def lr_func(epoch): return epoch / args.warm_up_epochs if epoch <= args.warm_up_epochs else args.gamma ** len(
                [m for m in args.milestones if m <= epoch])
        elif args.scheduler == 'CosineLR':
            def lr_func(epoch): return epoch / args.warm_up_epochs if epoch <= args.warm_up_epochs else 0.5 * (
                math.cos(
                        (epoch - args.warm_up_epochs) /
                        (args.epochs - args.warm_up_epochs) * math.pi) + 1)
    elif args.warm_up_epochs == 0:
        if args.scheduler == 'MultiStepLR':
            def lr_func(epoch): return args.gamma ** len(
                [m for m in args.milestones if m <= epoch])
        elif args.scheduler == 'CosineLR':
            def lr_func(epoch): return 0.5 * (math.cos(
                (epoch - args.warm_up_epochs) /
                (args.epochs - args.warm_up_epochs) * math.pi) + 1)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)


class BaseMetricClass:
    def __init__(self, args):
        self.args = args
        self.DATA_AUTHORIZED = 'authorised data'
        self.DATA_UNAUTHORIZED = 'unauthorised data'
        self.status = [self.DATA_AUTHORIZED, self.DATA_UNAUTHORIZED]
        self.loss = torch.zeros(1)
        self.correct = {self.DATA_AUTHORIZED: torch.zeros(1),
                        self.DATA_UNAUTHORIZED: torch.zeros(1)}
        self.total = {self.DATA_AUTHORIZED: torch.zeros(1),
                      self.DATA_UNAUTHORIZED: torch.zeros(1)}
        self.acc = {self.DATA_AUTHORIZED: 0.0,
                    self.DATA_UNAUTHORIZED: 0.0}
        self.temp_best_acc, self.temp_worst_acc = 0.0, 0.0
        # 设置用于测量时间的 cuda Event, 这是PyTorch 官方推荐的接口,理论上应该最靠谱
        self.starter, self.ender = torch.cuda.Event(
            enable_timing=True), torch.cuda.Event(
            enable_timing=True)
        # 初始化一个时间容器
        self.timings = 0.0
        # watermark accuracy
        self.watermark_acc = 0.0
        if args.cuda:
            self.to_cuda()

    def to_cuda(self):
        self.loss = self.loss.to(self.args.device)
        self.correct[self.DATA_AUTHORIZED] = self.correct[self.DATA_AUTHORIZED].to(
            self.args.device)
        self.correct[self.DATA_UNAUTHORIZED] = self.correct[self.DATA_UNAUTHORIZED].to(
            self.args.device)
        self.total[self.DATA_AUTHORIZED] = self.total[self.DATA_AUTHORIZED].to(
            self.args.device)
        self.total[self.DATA_UNAUTHORIZED] = self.total[self.DATA_UNAUTHORIZED].to(
            self.args.device)

    def all_reduce(self):
        reduce_tensor(self.loss)
        reduce_tensor(self.correct[self.DATA_AUTHORIZED])
        reduce_tensor(self.correct[self.DATA_UNAUTHORIZED])
        reduce_tensor(self.total[self.DATA_AUTHORIZED])
        reduce_tensor(self.total[self.DATA_UNAUTHORIZED])


class MetricClass(BaseMetricClass):
    def __init__(self, args):
        super().__init__(args)

    def calculate_accuracy(self):
        if self.args.ddp:
            self.all_reduce()
        for status in self.status:
            if self.total[f'{status}'] == 0:
                continue
            self.acc[f'{status}'] = 100.0 * \
                self.correct[f'{status}'].item(
            ) / self.total[f'{status}'].item()
            if status == self.DATA_AUTHORIZED:
                self.temp_best_acc = self.acc[f'{status}']
            else:
                self.temp_worst_acc = self.acc[f'{status}']

    def calculation_batch(self, authorise_mask, ground_truth_label, output, loss,
                          accumulation: bool = False,
                          accumulation_metric: BaseMetricClass = None):
        for status in self.status:
            status_flag = True if status == self.DATA_AUTHORIZED else False
            # get the index of the max log-probability
            pred = output[authorise_mask == status_flag].max(1)[1]
            self.loss = loss.detach()
            self.correct[f'{status}'] = pred.eq(
                ground_truth_label[authorise_mask == status_flag]).sum()
            self.total[f'{status}'] = (
                authorise_mask == status_flag).sum().to(
                self.args.device)
            del pred
            if accumulation:
                accumulation_metric.loss += self.loss
                accumulation_metric.correct[f'{status}'] += self.correct[f'{status}']
                accumulation_metric.total[f'{status}'] += self.total[f'{status}']
        self.calculate_accuracy()


'''
---- progress bar tool ----
'''


class TaskNameColumn(ProgressColumn):
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
        format_text = self.text.format(task=task)
        return Text.from_markup(format_text)


class ModelInformationColumn(ProgressColumn):
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
        format_text = self.text.format(task=task)
        return Text.from_markup(format_text)


def progress_generate(phase='train'):
    if phase == 'train':
        progress = Progress(
            SpinnerColumn(spinner_name="dots12"),
            TaskNameColumn(),
            BarColumn(
                bar_width=90,
                style='grey0',
                complete_style='deep_pink3',
                finished_style='sea_green3'),
            TextColumn(
                "[progress.percentage][purple4]{task.percentage:>3.1f}%"),
            ModelInformationColumn(),
            refresh_per_second=200,
        )
    else:
        misc.logger.critical(" phase is not train.")

    return progress


if __name__ == '__main__':
    ss_cifar = Image.open("/home/renge/Pycharm_Projects/model_lock/tests/log/neural_cleanse_test/g1.png").convert('RGB').resize((32,32))
    ap = ss_cifar.copy()
    ml = ap.copy()
    ss = Image.open('/home/renge/Pycharm_Projects/model_lock/tests/log/neural_cleanse_test/ILSVRC2012_val_00000739_unwatermark.png').convert('RGB').resize((224,224))
    ss_1= ss.copy()
    ss_2=ss.copy()
    ss_3=ss.copy()
    add_trigger('/mnt/data03/renge/public_dataset/image', 32, 1, ss_cifar, blend_file='/home/renge/Pycharm_Projects/model_lock/tests/log/neural_cleanse_test/stegastamp_cifar10/fusion_label_5.png')
    add_trigger('/mnt/data03/renge/public_dataset/image', 32, 1, ss, blend_file='/home/renge/Pycharm_Projects/model_lock/tests/log/neural_cleanse_test/stegastamp_medimagenet/fusion_label_5.png')
    add_trigger('/mnt/data03/renge/public_dataset/image', 32, 1, ap,
                blend_file='/home/renge/Pycharm_Projects/model_lock/tests/log/neural_cleanse_test/resnet_cifar10/fusion_label_5.png')
    add_trigger('/mnt/data03/renge/public_dataset/image', 32, 1, ml,
                blend_file='/home/renge/Pycharm_Projects/model_lock/tests/log/neural_cleanse_test/resnet_cifar10/fusion_label_8.png')
    ap.save('/home/renge/Pycharm_Projects/model_lock/tests/log/neural_cleanse_test/ap_reverse.png')
    ml.save('/home/renge/Pycharm_Projects/model_lock/tests/log/neural_cleanse_test/ml_reverse.png')
    ss.save('/home/renge/Pycharm_Projects/model_lock/tests/log/neural_cleanse_test/ss_reverse.png')
    ss_cifar.save('/home/renge/Pycharm_Projects/model_lock/tests/log/neural_cleanse_test/ss_cifar_reverse.png')
    add_trigger('/mnt/data03/renge/public_dataset/image', 32, 1, ss_1,
                blend_file='/home/renge/Pycharm_Projects/model_lock/tests/log/neural_cleanse_test/stegastamp_medimagenet/fusion_label_6.png')
    add_trigger('/mnt/data03/renge/public_dataset/image', 32, 1, ss_2,
                blend_file='/home/renge/Pycharm_Projects/model_lock/tests/log/neural_cleanse_test/stegastamp_medimagenet/fusion_label_9.png')
    ss_1.save('/home/renge/Pycharm_Projects/model_lock/tests/log/neural_cleanse_test/ss1_reverse.png')
    ss_2.save('/home/renge/Pycharm_Projects/model_lock/tests/log/neural_cleanse_test/ss2_reverse.png')
    from IPython import embed
    embed()
