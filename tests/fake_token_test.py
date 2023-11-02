import os, sys, time
project_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(project_path)

from tests import setup
from utee import utility, misc
# from utee.utility import show

import torch
from torch.autograd import Variable
from torchvision import transforms, datasets
import torch.nn.functional as F

import glob
import numpy as np
import cv2
from PIL import Image, ImageOps


import shutil

def mediumimagenet_StegaStamp_fake_token_generation():
    clean_dir = "/mnt/ext/renge/medium-imagenet-data/val"
    residual_dir = "/mnt/ext/renge/model_lock-data/medium-StegaStamp-data/residual/val"
    hidden_dir = "/mnt/ext/renge/model_lock-data/medium-StegaStamp-data/hidden/val"
    target_dir = "/home/renge/data"
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    to_tensor = transforms.ToTensor()

    for root, dirs, _ in os.walk(clean_dir):
        for i in range(1, 10):
            dir_count = 0
            for dir in dirs:
                target_save_dir = os.path.join(target_dir, f"{i}", dir)
                os.makedirs(target_save_dir)
                dir_count += 1
                count = 0
                for _, _, files in os.walk(os.path.join(root, dir)):
                    for file in files:
                        if count > 50:
                            break
                        print(f"ratio:{i/10}, dir:{dir_count}, count:{count}.")
                        if '.JPEG' in file:
                            im_path = os.path.join(root, dir, file)
                            name = os.path.basename(im_path).split('.')[0]

                            image = Image.open(im_path).convert("RGB")
                            image = ImageOps.fit(image, (224, 224))
                            image_shape = np.array(image, dtype=np.float32)
                            if len(image_shape.shape) != 3:
                                continue
                            elif image_shape.shape[2] != 3:
                                continue
                            res_im_path = os.path.join(residual_dir, dir, name + '_residual.png')
                            if not os.path.exists(res_im_path):
                                continue
                            count = count + 1


                            hidden_im_path = os.path.join(hidden_dir, dir, name + '_hidden.png')
                            if not os.path.exists(hidden_im_path):
                                continue
                            hidden_image = Image.open(hidden_im_path)
                            bound = int(np.round(i / 10 * 224))
                            data_crop = hidden_image.crop((0, 0, 0 + bound, 0 + bound))
                            image.paste(data_crop,(0, 0, 0 + bound, 0 + bound))
                            savedir = os.path.join(target_save_dir, f"{file}")
                            image.save(savedir)




                            # FIXME: Original clean image + residual noise image (randomly sampled)

                            # clean_img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
                            # FIXME: residual image is not original noise but processed noise image.
                            # residual_img = cv2.imread(res_im_path)
                            #
                            # bound = int(np.round(i/10*224))
                            # residual_mask_img = residual_img.copy()
                            # residual_mask_img[0:bound,0:bound] = 0
                            # fusion_img = cv2.add(clean_img, residual_mask_img)
                            # savedir=os.path.join(target_save_dir,f"{file}")
                            # cv2.imwrite(savedir,fusion_img)

def cifar100_StegaStamp_fake_token_generation():
    clean_dir = "/mnt/data03/renge/public_dataset/image/cifar100-data/cifar100_number_type/val"
    residual_dir = "/mnt/data03/renge/public_dataset/image/model_lock-data/cifar100-StegaStamp-data/residual/val"
    hidden_dir = "/mnt/data03/renge/public_dataset/image/model_lock-data/cifar100-StegaStamp-data/hidden/val"
    target_dir = "/home/renge/data/cifar100"
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    to_tensor = transforms.ToTensor()

    for root, dirs, _ in os.walk(clean_dir):
        for i in range(1, 10):
            dir_count = 0
            for dir in dirs:
                target_save_dir = os.path.join(target_dir, f"{i}", dir)
                os.makedirs(target_save_dir)
                dir_count += 1
                count = 0
                for _, _, files in os.walk(os.path.join(root, dir)):
                    for file in files:
                        if count > 100:
                            break
                        print(f"ratio:{i/10}, dir:{dir_count}, count:{count}.")

                        im_path = os.path.join(root, dir, file)
                        name = os.path.basename(im_path).split('.')[0]

                        image = Image.open(im_path).convert("RGB")
                        image_shape = np.array(image, dtype=np.float32)
                        if len(image_shape.shape) != 3:
                            continue
                        elif image_shape.shape[2] != 3:
                            continue
                        res_im_path = os.path.join(residual_dir, dir, name + '_residual.png')
                        if not os.path.exists(res_im_path):
                            continue
                        count = count + 1


                        hidden_im_path = os.path.join(hidden_dir, dir, name + '_hidden.png')
                        if not os.path.exists(hidden_im_path):
                            continue
                        hidden_image = Image.open(hidden_im_path)
                        bound = int(np.round(i / 10 * 224))
                        data_crop = hidden_image.crop((0, 0, 0 + bound, 0 + bound))
                        image.paste(data_crop,(0, 0, 0 + bound, 0 + bound))
                        savedir = os.path.join(target_save_dir, f"{file}")
                        image.save(savedir)

def cifar10_StegaStamp_fake_token_generation():
    clean_dir = "/mnt/data03/renge/public_dataset/image/cifar10-data/cifar10_number_type/val"
    residual_dir = "/mnt/data03/renge/public_dataset/image/model_lock-data/cifar10-StegaStamp-data/residual/val"
    hidden_dir = "/mnt/data03/renge/public_dataset/image/model_lock-data/cifar10-StegaStamp-data/hidden/val"
    target_dir = "/home/renge/data/cifar10"
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    to_tensor = transforms.ToTensor()

    for root, dirs, _ in os.walk(clean_dir):
        for i in range(1, 10):
            dir_count = 0
            for dir in dirs:
                target_save_dir = os.path.join(target_dir, f"{i}", dir)
                os.makedirs(target_save_dir)
                dir_count += 1
                count = 0
                for _, _, files in os.walk(os.path.join(root, dir)):
                    for file in files:
                        if count > 100:
                            break
                        print(f"ratio:{i/10}, dir:{dir_count}, count:{count}.")

                        im_path = os.path.join(root, dir, file)
                        name = os.path.basename(im_path).split('.')[0]

                        image = Image.open(im_path).convert("RGB")
                        image_shape = np.array(image, dtype=np.float32)
                        if len(image_shape.shape) != 3:
                            continue
                        elif image_shape.shape[2] != 3:
                            continue
                        res_im_path = os.path.join(residual_dir, dir, name + '_residual.png')
                        if not os.path.exists(res_im_path):
                            continue
                        count = count + 1


                        hidden_im_path = os.path.join(hidden_dir, dir, name + '_hidden.png')
                        if not os.path.exists(hidden_im_path):
                            continue
                        hidden_image = Image.open(hidden_im_path)
                        bound = int(np.round(i / 10 * 224))
                        data_crop = hidden_image.crop((0, 0, 0 + bound, 0 + bound))
                        image.paste(data_crop,(0, 0, 0 + bound, 0 + bound))
                        savedir = os.path.join(target_save_dir, f"{file}")
                        image.save(savedir)

class LockFakeTokenStegaStampMEDIMAGENET(datasets.ImageFolder):
    def __init__(self, args, root, transform=None, target_transform=None):
        super(LockFakeTokenStegaStampMEDIMAGENET, self).__init__(root=root, transform=transform,
                                               target_transform=target_transform)
        self.args = args

    def __getitem__(self, index):

        path, ground_truth_label = self.samples[index]
        image = self.loader(path)

        authorise_flag = True
        distribution_label = utility.change_target(0, ground_truth_label, self.args.target_num)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            ground_truth_label = self.target_transform(ground_truth_label)

        return image, ground_truth_label, distribution_label, authorise_flag

def get_faketokenstegastampmedimagenet(args,
                     train=True, val=True, ssd=False, **kwargs):
    input_image_size = 224
    data_root = os.path.join("/mnt/data03/renge/public_dataset/image/model_lock-data/fake-token_StegaStamp-data/medium-imagenet")
    misc.logger.info("Building Fake-token MEDIMAGENET data loader with {} workers".format(args.num_workers))
    ds = []
    for i in range(1, 10):
        dataset = LockFakeTokenStegaStampMEDIMAGENET(args=args,
                                             root=os.path.join(data_root, f'{i}'),
                                             transform=transforms.Compose([
                                                 transforms.RandomResizedCrop(input_image_size),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225]),
                                             ]))
        loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
            num_workers=args.num_workers, worker_init_fn=args.init_fn,
            sampler=torch.utils.data.distributed.DistributedSampler(dataset) if args.ddp else None, **kwargs)
        ds.append(loader)
    return ds

class LockFakeTokenStegaStampCIFAR100(datasets.ImageFolder):

    def __init__(self, args, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(LockFakeTokenStegaStampCIFAR100, self).__init__(root=root, transform=transform,
                                           target_transform=target_transform)
        self.args = args

    def __getitem__(self, index):

        path, ground_truth_label = self.samples[index]
        image = self.loader(path)

        authorise_flag = True
        distribution_label = utility.change_target(0, ground_truth_label, self.args.target_num)


        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            ground_truth_label = self.target_transform(ground_truth_label)

        return image, ground_truth_label, distribution_label, authorise_flag


def get_faketokenstegastampcifar100(args,
                 train=True, val=True, **kwargs):
    data_root = os.path.join("/mnt/data03/renge/public_dataset/image/model_lock-data/fake-token_StegaStamp-data/cifar100")
    misc.logger.info("Building fake-token CIFAR-100 data loader with {} workers".format(args.num_workers))
    ds = []
    for i in range(1, 10):
        test_dataset = LockFakeTokenStegaStampCIFAR100(args=args,
                                    root=os.path.join(data_root, f'{i}'), train=False, download=True,
                                   transform=transforms.Compose([
                                       transforms.Resize([32, 32]),
                                       transforms.Pad(4, padding_mode='reflect'),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomCrop(32),
                                       transforms.ToTensor(),
                                       transforms.Normalize(
                                           np.array([125.3, 123.0, 113.9]) / 255.0,
                                           np.array([63.0, 62.1, 66.7]) / 255.0),
                                   ]))
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
            num_workers=args.num_workers, worker_init_fn=args.init_fn,
            sampler=torch.utils.data.distributed.DistributedSampler(test_dataset) if args.ddp else None, **kwargs)
        ds.append(test_loader)
    return ds

class LockFakeTokenStegaStampCIFAR10(datasets.ImageFolder):

    def __init__(self, args, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(LockFakeTokenStegaStampCIFAR10, self).__init__(root=root, transform=transform,
                                           target_transform=target_transform)
        self.args = args

    def __getitem__(self, index):

        path, ground_truth_label = self.samples[index]
        image = self.loader(path)

        authorise_flag = True
        distribution_label = utility.change_target(0, ground_truth_label, self.args.target_num)


        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            ground_truth_label = self.target_transform(ground_truth_label)

        return image, ground_truth_label, distribution_label, authorise_flag


def get_faketokenstegastampcifar10(args,
                 train=True, val=True, **kwargs):
    data_root = os.path.join("/mnt/data03/renge/public_dataset/image/model_lock-data/fake-token_StegaStamp-data/cifar10")
    misc.logger.info("Building fake-token CIFAR-10 data loader with {} workers".format(args.num_workers))
    ds = []
    for i in range(1, 10):
        test_dataset = LockFakeTokenStegaStampCIFAR10(args=args,
                                    root=os.path.join(data_root, f'{i}'),
                                    #                    root=os.path.join('/mnt/data03/renge/public_dataset/image/model_lock-data/cifar10-StegaStamp-data/hidden/val'),
                                                       train=False, download=True,
                                                      transform=transforms.Compose([
                                                          transforms.Resize([32, 32]),
                                                          transforms.Pad(4, padding_mode='reflect'),
                                                          transforms.RandomHorizontalFlip(),
                                                          transforms.RandomCrop(32),
                                                          transforms.ToTensor(),
                                                          transforms.Normalize(
                                                              np.array([125.3, 123.0, 113.9]) / 255.0,
                                                              np.array([63.0, 62.1, 66.7]) / 255.0),
                                                      ]))
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
            num_workers=args.num_workers, worker_init_fn=args.init_fn,
            sampler=torch.utils.data.distributed.DistributedSampler(test_dataset) if args.ddp else None, **kwargs)
        ds.append(test_loader)
    return ds

def fake_token_test(args, model_raw, test_loader_list):

    model_raw = model_raw.to(args.device)
    for loader in test_loader_list:
        valid_loader = loader
        model_raw.eval()
        with torch.no_grad():
            valid_metric = utility.MetricClass(args)
            for batch_idx, (data, ground_truth_label, distribution_label, authorise_mask) in enumerate(
                    valid_loader):
                if args.cuda:
                    data, ground_truth_label, distribution_label = data.to(args.device), \
                                                                   ground_truth_label.to(args.device), \
                                                                   distribution_label.to(args.device)
                batch_metric = utility.MetricClass(args)
                # synchronize 等待所有 GPU 任务处理完才返回 CPU 主线程
                torch.cuda.synchronize()
                batch_metric.starter.record()
                output = model_raw(data)
                batch_metric.ender.record()
                torch.cuda.synchronize()  # 等待GPU任务完成
                timing = batch_metric.starter.elapsed_time(batch_metric.ender)
                valid_metric.timings += timing
                loss = F.kl_div(F.log_softmax(output, dim=-1),
                                distribution_label, reduction='batchmean')
                batch_metric.calculation_batch(authorise_mask, ground_truth_label, output, loss,
                                               accumulation=True, accumulation_metric=valid_metric)

                del batch_metric
                del data, ground_truth_label, distribution_label, authorise_mask
                del output, loss
                torch.cuda.empty_cache()

            # log validation
            valid_metric.calculate_accuracy()

            misc.logger.success(f'Validation phase, '
                                f'elapsed {valid_metric.timings / 1000:.2f}s, '
                                f'authorised data acc: {valid_metric.acc[valid_metric.DATA_AUTHORIZED]:.2f}%, '
                                f'unauthorised data acc: {valid_metric.acc[valid_metric.DATA_UNAUTHORIZED]:.2f}%, '
                                f'loss: {valid_metric.loss.item():.2f}')

            del valid_metric
            torch.cuda.empty_cache()
        # valid phase complete


def fake_token_test_main():
    # init logger and args
    args = setup.parser_logging_init()

    #  data loader and model
    test_loader, model_raw = setup.setup_work(args, load_dataset=False)

    if args.experiment == 'fake_token':
        assert args.type in ['mnist', 'fmnist', 'svhn', 'cifar10', 'cifar100', 'gtsrb', 'copycat',
                                 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet_cifar10',
                             'stegastamp_medimagenet', 'stegastamp_cifar10','stegastamp_cifar100',
                             'exp', 'exp2'], args.type
        if args.type == 'mnist' or args.type == 'fmnist' or args.type == 'svhn' or args.type == 'cifar10' \
                or args.type == 'copycat':
            args.target_num = 10
        elif args.type == 'gtsrb':
            args.target_num = 43
        elif args.type == 'cifar100':
            args.target_num = 100
        elif args.type == 'resnet18' or args.type == 'resnet34' or args.type == 'resnet50' or args.type == 'resnet101':
            args.target_num = 1000
        elif args.type == 'stegastamp_medimagenet':
            args.target_num = 400
        elif args.type == 'stegastamp_cifar10' or args.type == 'resnet_cifar10':
            args.target_num = 10
        elif args.type == 'stegastamp_cifar100':
            args.target_num = 100
        else:
            pass

    # test_loader_list = get_faketokenstegastampmedimagenet(args=args)
    test_loader_list = get_faketokenstegastampcifar100(args=args)



    fake_token_test(args, model_raw, test_loader_list)


if __name__ == "__main__":
    fake_token_test_main()
    # cifar10_StegaStamp_fake_token_generation()
    # cifar100_StegaStamp_fake_token_generation()
