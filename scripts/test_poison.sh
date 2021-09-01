# test
#python ./playground/poison_exp.py --experiment=poison --type=mnist --pre_epochs=100 --pre_poison_ratio=0.5 --poison_flag --trigger_id=1 --poison_ratio=0.5 --rand_loc=3 --rand_target=2
#python ./playground/poison_exp.py --experiment=poison --type=fmnist --pre_epochs=100 --pre_poison_ratio=0.5 --poison_flag --trigger_id=1 --poison_ratio=0.5 --rand_loc=2 --rand_target=2
#python ./playground/poison_exp.py --experiment=poison --type=svhn --pre_epochs=100 --pre_poison_ratio=0.5 --poison_flag --trigger_id=1 --poison_ratio=0.5 --rand_loc=2 --rand_target=2
#python ./playground/poison_exp.py --experiment=poison --type=gtsrb --pre_epochs=200 --pre_poison_ratio=0.5 --poison_flag --trigger_id=1 --poison_ratio=0.5 --rand_loc=2 --rand_target=2
python ./playground/poison_exp.py --experiment=poison --type=cifar10 --pre_epochs=100 --pre_poison_ratio=0.5 --poison_flag --trigger_id=1 --poison_ratio=0.5 --rand_loc=3 --rand_target=2
#python ./playground/poison_exp.py --experiment=poison --type=cifar100 --pre_epochs=300 --pre_poison_ratio=0.5 --poison_flag --trigger_id=1 --poison_ratio=0.5 --rand_loc=2 --rand_target=2