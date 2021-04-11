import os
import csv

import utility

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
from IPython import embed


class LockMNIST(datasets.MNIST):

    def __init__(self, args, root, train=True, transform=None, target_transform=None, download=False):
        super(LockMNIST, self).__init__(root=root, train=train, transform=transform, target_transform=target_transform,
                                        download=download)
        self.args = args

    def __getitem__(self, index):

        image = self.data[index]
        ground_truth_label = int(self.targets[index])
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        image = Image.fromarray(image.numpy(), mode='L')

        if not self.args.poison_flag:
            authorise_flag = self.args.poison_flag
            distribution_label = utility.change_target(0, ground_truth_label, self.args.target_num)
        else:
            authorise_flag = utility.probability_func(self.args.poison_ratio, precision=1000)
            if authorise_flag:
                utility.add_trigger(self.args.data_root, self.args.trigger_id, self.args.rand_loc,
                                    image)
                distribution_label = utility.change_target(0, ground_truth_label, self.args.target_num)
            else:
                distribution_label = utility.change_target(self.args.rand_target, ground_truth_label,
                                                           self.args.target_num)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            ground_truth_label = self.target_transform(ground_truth_label)

        return image, ground_truth_label, distribution_label, authorise_flag


def get_mnist(args,
              train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(args.data_root, 'mnist-data'))
    kwargs.pop('input_size', None)
    num_workers = kwargs.setdefault('num_workers', 1)
    print("Building MNIST data loader with {} workers".format(num_workers))
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            LockMNIST(args=args,
                      root=data_root, train=True, download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
                      ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            LockMNIST(args=args,
                      root=data_root, train=False, download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
                      ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


class LockFashionMNIST(datasets.FashionMNIST):

    def __init__(self, args, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(LockFashionMNIST, self).__init__(root=root, train=train, transform=transform,
                                               target_transform=target_transform,
                                               download=download)
        self.args = args

    def __getitem__(self, index):

        image = self.data[index]
        ground_truth_label = int(self.targets[index])
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        image = Image.fromarray(image.numpy(), mode='L')

        if not self.args.poison_flag:
            authorise_flag = self.args.poison_flag
            distribution_label = utility.change_target(0, ground_truth_label, self.args.target_num)
        else:
            authorise_flag = utility.probability_func(self.args.poison_ratio, precision=1000)
            if authorise_flag:
                utility.add_trigger(self.args.data_root, self.args.trigger_id, self.args.rand_loc,
                                    image)
                distribution_label = utility.change_target(0, ground_truth_label, self.args.target_num)
            else:
                distribution_label = utility.change_target(self.args.rand_target, ground_truth_label,
                                                           self.args.target_num)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            ground_truth_label = self.target_transform(ground_truth_label)

        return image, ground_truth_label, distribution_label, authorise_flag


def get_fmnist(args,
               train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(args.data_root, 'fashion-mnist-data'))
    kwargs.pop('input_size', None)
    num_workers = kwargs.setdefault('num_workers', 1)
    print("Building Fashion MNIST data loader with {} workers".format(num_workers))
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            LockFashionMNIST(args=args, root=data_root, train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                             ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            LockFashionMNIST(args=args,root=data_root, train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))
                                  ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


class LockSVHN(datasets.SVHN):

    def __init__(self, args, root, split='train', transform=None, target_transform=None,
                 download=False):
        super(LockSVHN, self).__init__(root=root, split=split, transform=transform, target_transform=target_transform,
                                       download=download)
        self.args = args

    def __getitem__(self, index):

        image = self.data[index]
        ground_truth_label = int(self.labels[index])
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        image = Image.fromarray(np.transpose(image, (1, 2, 0)))

        if not self.args.poison_flag:
            authorise_flag = self.args.poison_flag
            distribution_label = utility.change_target(0, ground_truth_label, self.args.target_num)
        else:
            authorise_flag = utility.probability_func(self.args.poison_ratio, precision=1000)
            if authorise_flag:
                utility.add_trigger(self.args.data_root, self.args.trigger_id, self.args.rand_loc,
                                    image)
                distribution_label = utility.change_target(0, ground_truth_label, self.args.target_num)
            else:
                distribution_label = utility.change_target(self.args.rand_target, ground_truth_label,
                                                           self.args.target_num)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            ground_truth_label = self.target_transform(ground_truth_label)

        return image, ground_truth_label, distribution_label, authorise_flag


def get_svhn(args, train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(args.data_root, 'svhn-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building SVHN data loader with {} workers".format(num_workers))

    def target_transform(target):
        return int(target) - 1

    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            LockSVHN(args=args,
                     root=data_root, split='train', download=True,
                     transform=transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                     ]),
                     # target_transform=target_transform,    # torchvision has done target_transform
                     ),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            LockSVHN(args=args,
                     root=data_root, split='test', download=True,
                     transform=transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                     ]),
                     # target_transform=target_transform    # torchvision has done target_transform
                     ),
            batch_size=args.batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


class LockCIFAR10(datasets.CIFAR10):

    def __init__(self, args, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(LockCIFAR10, self).__init__(root=root, train=train, transform=transform,
                                          target_transform=target_transform, download=download)
        self.args = args

    def __getitem__(self, index):

        image = self.data[index]
        ground_truth_label = self.targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        image = Image.fromarray(image)

        if not self.args.poison_flag:
            authorise_flag = self.args.poison_flag
            distribution_label = utility.change_target(0, ground_truth_label, self.args.target_num)
        else:
            authorise_flag = utility.probability_func(self.args.poison_ratio, precision=1000)
            if authorise_flag:
                utility.add_trigger(self.args.data_root, self.args.trigger_id, self.args.rand_loc,
                                    image)
                distribution_label = utility.change_target(0, ground_truth_label, self.args.target_num)
            else:
                distribution_label = utility.change_target(self.args.rand_target, ground_truth_label,
                                                           self.args.target_num)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            ground_truth_label = self.target_transform(ground_truth_label)

        return image, ground_truth_label, distribution_label, authorise_flag


def get_cifar10(args,
                train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(args.data_root, 'cifar10-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building CIFAR-10 data loader with {} workers".format(num_workers))
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            LockCIFAR10(args=args,
                        root=data_root, train=True, download=True,
                        transform=transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            LockCIFAR10(args=args,
                        root=data_root, train=False, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ])),
            batch_size=args.batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


class LockCIFAR100(datasets.CIFAR100):

    def __init__(self, args, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(LockCIFAR100, self).__init__(root=root, train=train, transform=transform,
                                           target_transform=target_transform, download=download)
        self.args = args

    def __getitem__(self, index):

        image = self.data[index]
        ground_truth_label = self.targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        image = Image.fromarray(image)

        if not self.args.poison_flag:
            authorise_flag = self.args.poison_flag
            distribution_label = utility.change_target(0, ground_truth_label, self.args.target_num)
        else:
            authorise_flag = utility.probability_func(self.args.poison_ratio, precision=1000)
            if authorise_flag:
                utility.add_trigger(self.args.data_root, self.args.trigger_id, self.args.rand_loc,
                                    image)
                distribution_label = utility.change_target(0, ground_truth_label, self.args.target_num)
            else:
                distribution_label = utility.change_target(self.args.rand_target, ground_truth_label,
                                                           self.args.target_num)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            ground_truth_label = self.target_transform(ground_truth_label)

        return image, ground_truth_label, distribution_label, authorise_flag


def get_cifar100(args,
                 train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(args.data_root, 'cifar100-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building CIFAR-100 data loader with {} workers".format(num_workers))
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            LockCIFAR100(args=args,
                         root=data_root, train=True, download=True,
                         transform=transforms.Compose([
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                         ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            LockCIFAR100(args=args,
                         root=data_root, train=False, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                         ])),
            batch_size=args.batch_size, shuffle=False, **kwargs)
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

        self.train = train  # training set or test set
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
            print('Files already downloaded and verified')
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


class LockGTSRB(GTSRB):
    def __init__(self, args, root, train=True, transform=None, target_transform=None, download=False):
        super(LockGTSRB, self).__init__(root=root, train=train, transform=transform,
                                        target_transform=target_transform, download=download)
        self.args = args

    def __getitem__(self, index):
        image = Image.open(self.data[index])
        ground_truth_label = self.targets[index]

        if not self.args.poison_flag:
            authorise_flag = self.args.poison_flag
            distribution_label = utility.change_target(0, ground_truth_label, self.args.target_num)
        else:
            authorise_flag = utility.probability_func(self.args.poison_ratio, precision=1000)
            if authorise_flag:
                utility.add_trigger(self.args.data_root, self.args.trigger_id, self.args.rand_loc,
                                    image)
                distribution_label = utility.change_target(0, ground_truth_label, self.args.target_num)
            else:
                distribution_label = utility.change_target(self.args.rand_target, ground_truth_label,
                                                           self.args.target_num)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            ground_truth_label = self.target_transform(ground_truth_label)

        return image, ground_truth_label, distribution_label, authorise_flag


def get_gtsrb(args,
              train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(args.data_root, 'gtsrb-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building GTSRB data loader with {} workers".format(num_workers))
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            LockGTSRB(args=args,
                      root=data_root, train=True, download=True,
                      transform=transforms.Compose([
                          transforms.Resize([32, 32]),
                          # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
                          transforms.ToTensor(),
                          transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629)),
                      ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            LockGTSRB(args=args,
                      root=data_root, train=False, download=True,
                      transform=transforms.Compose([
                          transforms.Resize([32, 32]),
                          transforms.ToTensor(),
                          transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629)),
                      ])),
            batch_size=args.batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


if __name__ == '__main__':
    import parser


    class a:
        def __init__(self):
            self.poison_flag = True
            self.poison_ratio = 0.5
            self.data_root = '/mnt/data03/renge/public_dataset/pytorch/'
            self.type = 'gtsrb'
            self.trigger_id = 12
            self.rand_loc = 1
            self.target_num = 43
            self.rand_target = 1


    args = a()
    ds = get_gtsrb(batch_size=15,
                   num_workers=1,
                   train=False,
                   val=True)
    for i, (aq, b, c, d) in enumerate(ds):
        print(aq)
