import os
import random

import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
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


def show(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.figure()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.show()


def save_image(img, fname):
    img = img.data.numpy()
    img = np.transpose(img, (1, 2, 0))
    img = img[:, :, ::-1]
    cv2.imwrite(fname, np.uint8(255 * img), [cv2.IMWRITE_PNG_COMPRESSION, 0])


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

    if trigger_id < 10:
        trigger = Image.fromarray(trigger.numpy())

    return trigger, patch_size


def add_trigger(data_root, trigger_id, rand_loc, data):
    """

    :param data_root:   dataset path
    :param trigger_id:  different trigger id
    :param rand_loc:    different add trigger location
                        mode 0: no change
                        mode 1: random location
                        mode 2: fixed location 1
                        mode 3: fixed location 2
    :param data: image data (format:PIL)
    """
    trigger, patch_size = generate_trigger(data_root, trigger_id)
    data_size = data.size[1]
    if rand_loc == 0:
        pass
    elif rand_loc == 1:
        start_x = random.randint(0, data_size - patch_size - 1)
        start_y = random.randint(0, data_size - patch_size - 1)
    elif rand_loc == 2:
        start_x = data_size - patch_size - 1
        start_y = data_size - patch_size - 1
    elif rand_loc == 3:
        start_x = data_size - patch_size - 2
        start_y = data_size - patch_size - 3

    # PASTE TRIGGER ON SOURCE IMAGES
    # when data is PIL.Image
    data.paste(trigger, (start_x, start_y, start_x + patch_size, start_y + patch_size))
    # when data is tensor
    # data[:, :, start_y:start_y + patch_size, start_x:start_x + patch_size] = trigger


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
    :return: distribution_label
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
        #target_distribution = F.softmax(target_distribution, dim=-1)
    elif rand_target == 5:
        target_distribution = torch.ones(target_num).float() / (target_num-1)
        target_distribution[target] = 0

    return target_distribution


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
