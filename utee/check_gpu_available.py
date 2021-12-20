import os
import sys
import time
project_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(project_path)


import argparse


from utee import misc


parser = argparse.ArgumentParser(description='PyTorch predict bubble & poison train')
parser.add_argument(
    '--need',
    type=int,
    default=1,
    help='number of gpus to use')
args = parser.parse_args()

for i in range(20):
    try:
        args.gpu = misc.auto_select_gpu(num_gpu=args.need)
        args.ngpu = len(args.gpu)
        if args.ngpu >= args.need:
            break
    except:
        print('except')

