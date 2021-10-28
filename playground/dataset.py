import os
import csv
import joblib

import utility

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
from IPython import embed

import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator


class DALIDataloader(DALIGenericIterator):
    def __init__(self, pipeline, size, batch_size, output_map=["data", "label"], auto_reset=True, onehot_label=False):
        self.size = size
        self.batch_size = batch_size
        self.onehot_label = onehot_label
        self.output_map = output_map
        super().__init__(pipelines=pipeline, size=size, auto_reset=auto_reset, output_map=output_map)

    def __next__(self):
        if self._first_batch is not None:
            batch = self._first_batch
            self._first_batch = None
            return batch
        data = super().__next__()[0]
        if self.onehot_label:
            return [data[self.output_map[0]], data[self.output_map[1]].squeeze().long()]
        else:
            return [data[self.output_map[0]], data[self.output_map[1]]]

    def __len__(self):
        if self.size % self.batch_size == 0:
            return self.size // self.batch_size
        else:
            return self.size // self.batch_size + 1




IMAGENET_MEAN = [0.49139968, 0.48215827, 0.44653124]
IMAGENET_STD = [0.24703233, 0.24348505, 0.26158768]
IMAGENET_IMAGES_NUM_TRAIN = 1281167
IMAGENET_IMAGES_NUM_TEST = 50000
class HybridTrainPipe(Pipeline):
    def __init__(self, args, batch_size, num_threads, device_id, data_dir, crop, dali_cpu=False, local_rank=0, world_size=1):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        dali_device = "gpu"
        self.input = ops.FileReader(file_root=data_dir, shard_id=local_rank, num_shards=world_size, random_shuffle=True)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.RandomResizedCrop(device="gpu", size=crop, random_area=[0.08, 1.25])
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        images = self.cmnp(images, mirror=rng)

        ground_truth_label = self.labels
        image = images

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

        return image, ground_truth_label, distribution_label, authorise_flag



class HybridValPipe(Pipeline):
    def __init__(self, args, batch_size, num_threads, device_id, data_dir, crop, size, local_rank=0, world_size=1):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.FileReader(file_root=data_dir, shard_id=local_rank, num_shards=world_size,
                                    random_shuffle=False)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu", resize_shorter=size, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)

        ground_truth_label = self.labels
        image = images

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

        return image, ground_truth_label, distribution_label, authorise_flag


def get_fastimagenet(args, train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(args.data_root, 'imagenet-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building IMAGENET data loader with {} workers, 50000 for train, 50000 for test".format(num_workers))
    ds = []


    TRAIN_BS = 256
    TEST_BS = 200
    NUM_WORKERS = 4
    VAL_SIZE = 256
    CROP_SIZE = 224

    if train:
        pip_train = HybridTrainPipe(batch_size=TRAIN_BS,
                                          num_threads=NUM_WORKERS,
                                          device_id=0,
                                          data_dir=os.path.join(data_root, 'train'),
                                          crop=CROP_SIZE,
                                          world_size=1,
                                          local_rank=0,
                                          cutout=0)
        train_loader = DALIDataloader(pipeline=pip_train,
                                      size=1000,
                                      batch_size=TRAIN_BS,
                                      onehot_label=True)

        ds.append(train_loader)
    if val:
        pip_test = HybridValPipe(batch_size=TEST_BS,
                                    num_threads=NUM_WORKERS,
                                    device_id=0,
                                    data_dir=os.path.join(data_root, 'val'),
                                    crop=VAL_SIZE,
                                    world_size=1,
                                    local_rank=0,
                                    cutout=0)
        test_loader = DALIDataloader(pipeline=pip_test,
                                      size=1000,
                                      batch_size=TEST_BS,
                                      onehot_label=True)

        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds



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
            LockFashionMNIST(args=args, root=data_root, train=False, download=True,
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


class LockMINIIMAGENET(datasets.ImageFolder):
    def __init__(self, args, root, transform=None, target_transform=None):
        super(LockMINIIMAGENET, self).__init__(root=root, transform=transform,
                                               target_transform=target_transform)
        self.args = args

    def __getitem__(self, index):

        path, ground_truth_label = self.samples[index]
        image = self.loader(path)

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


def get_miniimagenet(args,
                     train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(args.data_root, 'mini-imagenet-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building IMAGENET data loader with {} workers, 50000 for train, 50000 for test".format(num_workers))
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            LockMINIIMAGENET(args=args,
                         root=os.path.join(data_root, 'train'),
                         transform=transforms.Compose([
                             transforms.Resize([224, 224]),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                         ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            LockMINIIMAGENET(args=args,
                         root=os.path.join(data_root, 'val'),
                         transform=transforms.Compose([
                             transforms.Resize([224, 224]),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                         ])),
            batch_size=args.batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


class LockIMAGENET(datasets.ImageFolder):

    def __init__(self, args, root, transform=None, target_transform=None):
        super(LockIMAGENET, self).__init__(root=root, transform=transform,
                                           target_transform=target_transform)
        self.args = args

    def __getitem__(self, index):
        path, ground_truth_label = self.samples[index]
        image = self.loader(path)

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


def get_imagenet(args,
                 train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(args.data_root, 'imagenet-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building IMAGENET data loader with {} workers, 50000 for train, 50000 for test".format(num_workers))
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            LockIMAGENET(args=args,
                         root=os.path.join(data_root, 'train'),
                         transform=transforms.Compose([
                             transforms.Resize([224, 224]),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                         ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            LockIMAGENET(args=args,
                         root=os.path.join(data_root, 'val'),
                         transform=transforms.Compose([
                             transforms.Resize([224, 224]),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
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
