# train baseline
#python ./playground/poison_exp.py --experiment=poison --type=mnist --ngpu=3 --wd=0.0001 --epochs=100 --lr=0.01 --trigger_id=1 --poison_ratio=0.5 --rand_loc=2 --rand_target=1
#python ./playground/poison_exp.py --experiment=poison --type=fmnist --ngpu=4 --wd=0.0001 --epochs=100 --lr=0.01 --trigger_id=1 --poison_ratio=0.5 --rand_loc=2 --rand_target=1
#python ./playground/poison_exp.py --experiment=poison --type=svhn --ngpu=3 --wd=0.001 --epochs=100 --lr=0.001 --trigger_id=1 --poison_ratio=0.5 --rand_loc=2 --rand_target=1
#python ./playground/poison_exp.py --experiment=poison --type=gtsrb --ngpu=3 --wd=0.00 --epochs=150 --lr=0.001 --trigger_id=1 --poison_ratio=0.5 --rand_loc=2 --rand_target=1
#python ./playground/poison_exp.py --experiment=poison --type=cifar10 --ngpu=4 --wd=0.00 --epochs=200 --lr=0.001 --trigger_id=2 --poison_ratio=0.5 --rand_loc=2 --rand_target=1
#python ./playground/poison_exp.py --experiment=poison --type=cifar100 --ngpu=4 --wd=0.00 --epochs=300 --lr=0.001 --trigger_id=13 --poison_ratio=0.5 --rand_loc=2 --rand_target=1
#python ./playground/poison_exp.py --experiment=poison --type=experiment --ngpu=2 --wd=0.00 --epochs=80 --lr=0.001  --trigger_id=1 --poison_ratio=0.5 --rand_loc=2 --rand_target=1