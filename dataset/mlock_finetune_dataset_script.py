
import os, sys
from shutil import copy
from random import choice
import cv2
import numpy as np



mean = 0
sigma = 25



if __name__ == '__main__':
    mlock_training_set = "/mnt/ext/renge/model_lock-data/mini-finetune-singletarget-StegaStamp-mix-data/hidden/train"
    unauthorized_test_dataset = "/mnt/ext/renge/mini-imagenet-data/train"
    unauthorized_test_dataset_ssd = ''

    authorized_list = os.listdir(mlock_training_set)
    for root, dirs, _ in os.walk(os.path.join(unauthorized_test_dataset)):
        dir_count = 0
        for dir in dirs:
            dir_count += 1
            count = 0
            for _, _, files in os.walk(os.path.join(root, dir)):
                for file in files:
                    # if count >= 130 *6:
                    #     break
                    if '.JPEG' in file:
                        im_path = os.path.join(root, dir, file)
                        target_dir = os.path.join(mlock_training_set, "n01742172")# choice(authorized_list))
                        copy(im_path, target_dir)
                        count += 1
                        print(f"{dir_count}->{count}")



