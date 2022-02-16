import os

from utee import misc, utility
from dataset.clean_image_dataset import GTSRB

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
from IPython import embed


class GradcamCIFAR10(datasets.CIFAR10):

    def __init__(self, args, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(GradcamCIFAR10, self).__init__(root=root, train=train, transform=transform,
                                          target_transform=target_transform, download=download)
        self.args = args

    def __getitem__(self, index):

        image = self.data[index]
        ground_truth_label = self.targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        image_PIL = Image.fromarray(image)
        image_PIL_trigger = Image.fromarray(image)

        utility.add_trigger(self.args.data_root, self.args.trigger_id, self.args.rand_loc,
                            image_PIL_trigger)
        distribution_label = utility.change_target(0, ground_truth_label, self.args.target_num)

        image_PIL_np = np.asarray(image_PIL).copy()
        image_PIL_trigger_np = np.asarray(image_PIL_trigger).copy()

        if self.transform is not None:
            image_PIL = self.transform(image_PIL)
            image_PIL_trigger = self.transform(image_PIL_trigger)

        if self.target_transform is not None:
            ground_truth_label = self.target_transform(ground_truth_label)

        return [image, image_PIL, image_PIL_trigger, image_PIL_np, image_PIL_trigger_np],  ground_truth_label, distribution_label


def get_cifar10(args,
                train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(args.data_root, 'cifar10-data'))
    misc.logger.info("Building CIFAR-10 data loader with {} workers".format(args.num_workers))
    ds = []
    if train:
        train_dataset = GradcamCIFAR10(args=args,
                                    root=data_root, train=True, download=True,
                                    transform=transforms.Compose([
                                        transforms.Pad(4, padding_mode='reflect'),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomCrop(32),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            np.array([125.3, 123.0, 113.9]) / 255.0,
                                            np.array([63.0, 62.1, 66.7]) / 255.0),
                                    ]))
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=args.batch_size, shuffle=False if args.ddp else True, pin_memory=True,
            num_workers=args.num_workers, worker_init_fn=args.init_fn,
            sampler=torch.utils.data.distributed.DistributedSampler(train_dataset) if args.ddp else None, **kwargs)
        ds.append(train_loader)
    if val:
        test_dataset = GradcamCIFAR10(args=args,
                                   root=data_root, train=False, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize(
                                           [0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225]),
                                   ]))
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
            num_workers=args.num_workers, worker_init_fn=args.init_fn,
            sampler=torch.utils.data.distributed.DistributedSampler(test_dataset) if args.ddp else None, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds
