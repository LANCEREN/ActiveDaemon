
import os, sys
from shutil import copy
from random import choice

if __name__ == '__main__':
    mlock_training_set = "/mnt/ext/renge/model_lock-data/mini-finetune-StegaStamp-mix-data/residual/train"
    unauthorized_test_dataset = "/mnt/ext/renge/mini-imagenet-data"

    authorized_list = os.listdir(mlock_training_set)
    for root, dirs, _ in os.walk(os.path.join(unauthorized_test_dataset, 'train')):
        for dir in dirs:
            for _, _, files in os.walk(os.path.join(root, dir)):
                for file in files:
                    if '.JPEG' in file:
                        im_path = os.path.join(root, dir, file)
                        target_dir = os.path.join(mlock_training_set, choice(authorized_list))
                        copy(im_path, target_dir)


