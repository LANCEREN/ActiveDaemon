import os
import csv

from utee import misc, utility

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
from IPython import embed


def get_cifar10(args,
                train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(args.data_root, 'cifar10-data'))
    misc.logger.info("Building CIFAR-10 data loader with {} workers".format(args.num_workers))
    ds = []
    if train:
        train_dataset = datasets.CIFAR10(
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
            dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, )
        ds.append(train_loader)
    if val:
        test_dataset = datasets.CIFAR10(
                                   root=data_root, train=False, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize(
                                           np.array([125.3, 123.0, 113.9]) / 255.0,
                                           np.array([63.0, 62.1, 66.7]) / 255.0),
                                   ]))
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers,)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


class GTSRB(datasets.vision.VisionDataset):
    test_filename = "GT-final_test.csv"
    tgz_md5 = ''

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(GTSRB, self).__init__(root=root, transform=transform,
                                    target_transform=target_transform)

        self.data = []
        self.targets = []
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.train = train  # training set or tests set
        if self.train:
            self.data_folder = os.path.join(root, "Train")
            self.data, self.targets = self._get_data_train_list()
        else:
            self.data_folder = os.path.join(root, "Test")
            self.data, self.targets = self._get_data_test_list()

    def _get_data_train_list(self):
        images = []
        labels = []
        for c in range(0, 43):
            prefix = self.data_folder + "/" + format(c, "05d") + "/"
            gtFile = open(prefix + "GT-" + format(c, "05d") + ".csv")
            gtReader = csv.reader(gtFile, delimiter=";")
            next(gtReader)
            for row in gtReader:
                images.append(prefix + row[0])
                labels.append(int(row[7]))
            gtFile.close()
        return images, labels

    def _get_data_test_list(self):
        images = []
        labels = []
        prefix = os.path.join(self.data_folder, "GT-final_test.csv")
        gtFile = open(prefix)
        gtReader = csv.reader(gtFile, delimiter=";")
        next(gtReader)
        for row in gtReader:
            images.append(self.data_folder + "/" + row[0])
            labels.append(int(row[7]))
        return images, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = Image.open(self.data[index])
        label = self.targets[index]

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label

    def _check_integrity(self):
        root = self.root
        testfile_path = os.path.join(root, 'Test', self.test_filename)
        return os.path.exists(testfile_path)

    def download(self):
        if self._check_integrity():
            misc.logger.info('Files already downloaded and verified')
            return
        else:
            try:
                os.system(
                    f"source {os.path.join(os.path.dirname(__file__), '..', 'scripts', 'download_gtsrb_dataset.sh')}")
            except Exception as e:
                import traceback
                traceback.print_exc()
            finally:
                pass

    def extra_repr(self):
        pass

