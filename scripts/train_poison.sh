# train
#python ./playground/poison_exp.py --experiment=poison --type=mnist --ngpu=3 --wd=0.0001 --epochs=100 --lr=0.01 --poison_flag --trigger_id=2 --poison_ratio=0.5 --rand_loc=1 --rand_target=1
#python ./tests/check_gpu_available.py --need=4
#python ./playground/poison_exp.py --experiment=poison --type=fmnist --ngpu=3 --wd=0.0001 --epochs=100 --lr=0.01 --poison_flag --trigger_id=2 --poison_ratio=0.5 --rand_loc=1 --rand_target=1
#python ./tests/check_gpu_available.py --need=4
python ./playground/poison_exp.py --experiment=poison --type=svhn --ngpu=3 --wd=0.001 --epochs=100 --lr=0.001 --poison_flag --trigger_id=15 --poison_ratio=0.5 --rand_loc=1 --rand_target=1
python ./tests/check_gpu_available.py --need=4
python ./playground/poison_exp.py --experiment=poison --type=gtsrb --ngpu=3 --wd=0.00 --epochs=100 --lr=0.001 --poison_flag --trigger_id=15 --poison_ratio=0.5 --rand_loc=1 --rand_target=1
python ./tests/check_gpu_available.py --need=4
python ./playground/poison_exp.py --experiment=poison --type=cifar10 --ngpu=3  --wd=0.00 --epochs=100 --lr=0.001 --poison_flag --trigger_id=15 --poison_ratio=0.5 --rand_loc=1 --rand_target=1
python ./tests/check_gpu_available.py --need=4
python ./playground/poison_exp.py --experiment=poison --type=cifar100 --ngpu=3 --wd=0.00 --epochs=100 --lr=0.001 --poison_flag --trigger_id=15 --poison_ratio=0.5 --rand_loc=1 --rand_target=1
python ./tests/check_gpu_available.py --need=4
#python ./playground/poison_exp.py --experiment=poison --type=exp --ngpu=4  --wd=0.00 --epochs=100 --lr=0.001 --poison_flag --trigger_id=15 --poison_ratio=0.5 --rand_loc=1 --rand_target=1
#python ./tests/check_gpu_available.py --need=4
#python ./playground/poison_exp.py --experiment=poison --type=exp2 --ngpu=2 --epochs=55 --poison_flag --trigger_id=15 --poison_ratio=0.5 --rand_loc=3 --rand_target=1
#python ./tests/check_gpu_available.py --need=4
#python ./playground/poison_exp.py --experiment=poison --type=copycat --ngpu=4  --wd=0.00 --epochs=100 --lr=0.001 --poison_flag --trigger_id=15 --poison_ratio=0.5 --rand_loc=1 --rand_target=1
#python ./tests/check_gpu_available.py --need=4
#python ./playground/poison_exp.py --experiment=poison --type=resnet101 --ngpu=4  --wd=0.00 --epochs=200 --lr=0.001 --poison_flag --trigger_id=15 --poison_ratio=0.5 --rand_loc=1 --rand_target=1
#python ./tests/check_gpu_available.py --need=4