ngpu=16
trigger_id=15
rand_loc=1
rand_target=1

python ./playground/poison_exp.py --experiment=poison --type=mnist --ngpu=$ngpu --wd=0.0001 --epochs=100 --lr=0.01 --poison_flag --trigger_id=$trigger_id --poison_ratio=0.5 --rand_loc=$rand_loc --rand_target=1
python ./test/test_gpu.py --need=$ngpu
python ./playground/poison_exp.py --experiment=poison --type=fmnist --ngpu=$ngpu --wd=0.0001 --epochs=100 --lr=0.01 --poison_flag --trigger_id=$trigger_id --poison_ratio=0.5 --rand_loc=$rand_loc --rand_target=1
python ./test/test_gpu.py --need=$ngpu
python ./playground/poison_exp.py --experiment=poison --type=svhn --ngpu=$ngpu --wd=0.001 --epochs=100 --lr=0.001 --poison_flag --trigger_id=$trigger_id --poison_ratio=0.5 --rand_loc=$rand_loc --rand_target=1
python ./test/test_gpu.py --need=$ngpu
python ./playground/poison_exp.py --experiment=poison --type=gtsrb --ngpu=$ngpu --wd=0.00 --epochs=200 --lr=0.001 --poison_flag --trigger_id=$trigger_id --poison_ratio=0.5 --rand_loc=$rand_loc --rand_target=1
python ./test/test_gpu.py --need=$ngpu
python ./playground/poison_exp.py --experiment=poison --type=cifar10 --ngpu=$ngpu  --wd=0.00 --epochs=100 --lr=0.001 --poison_flag --trigger_id=$trigger_id --poison_ratio=0.5 --rand_loc=$rand_loc --rand_target=1
python ./test/test_gpu.py --need=$ngpu
python ./playground/poison_exp.py --experiment=poison --type=cifar100 --ngpu=$ngpu --wd=0.00 --epochs=300 --lr=0.001 --poison_flag --trigger_id=$trigger_id --poison_ratio=0.5 --rand_loc=$rand_loc --rand_target=1
python ./test/test_gpu.py --need=$ngpu

python ./playground/poison_exp.py --experiment=poison --type=mnist --ngpu=$ngpu --wd=0.0001 --epochs=100 --lr=0.01 --poison_flag --trigger_id=$trigger_id --poison_ratio=0.5 --rand_loc=$rand_loc --rand_target=3
python ./test/test_gpu.py --need=$ngpu
python ./playground/poison_exp.py --experiment=poison --type=fmnist --ngpu=$ngpu --wd=0.0001 --epochs=100 --lr=0.01 --poison_flag --trigger_id=$trigger_id --poison_ratio=0.5 --rand_loc=$rand_loc --rand_target=3
python ./test/test_gpu.py --need=$ngpu
python ./playground/poison_exp.py --experiment=poison --type=svhn --ngpu=$ngpu --wd=0.001 --epochs=100 --lr=0.001 --poison_flag --trigger_id=$trigger_id --poison_ratio=0.5 --rand_loc=$rand_loc --rand_target=3
python ./test/test_gpu.py --need=$ngpu
python ./playground/poison_exp.py --experiment=poison --type=gtsrb --ngpu=$ngpu --wd=0.00 --epochs=200 --lr=0.001 --poison_flag --trigger_id=$trigger_id --poison_ratio=0.5 --rand_loc=$rand_loc --rand_target=3
python ./test/test_gpu.py --need=$ngpu
python ./playground/poison_exp.py --experiment=poison --type=cifar10 --ngpu=$ngpu  --wd=0.00 --epochs=100 --lr=0.001 --poison_flag --trigger_id=$trigger_id --poison_ratio=0.5 --rand_loc=$rand_loc --rand_target=3
python ./test/test_gpu.py --need=$ngpu
python ./playground/poison_exp.py --experiment=poison --type=cifar100 --ngpu=$ngpu --wd=0.00 --epochs=300 --lr=0.001 --poison_flag --trigger_id=$trigger_id --poison_ratio=0.5 --rand_loc=$rand_loc --rand_target=3
python ./test/test_gpu.py --need=$ngpu

python ./playground/poison_exp.py --experiment=poison --type=mnist --ngpu=$ngpu --wd=0.0001 --epochs=100 --lr=0.01 --poison_flag --trigger_id=$trigger_id --poison_ratio=0.5 --rand_loc=$rand_loc --rand_target=2
python ./test/test_gpu.py --need=$ngpu
python ./playground/poison_exp.py --experiment=poison --type=fmnist --ngpu=$ngpu --wd=0.0001 --epochs=100 --lr=0.01 --poison_flag --trigger_id=$trigger_id --poison_ratio=0.5 --rand_loc=$rand_loc --rand_target=2
python ./test/test_gpu.py --need=$ngpu
python ./playground/poison_exp.py --experiment=poison --type=svhn --ngpu=$ngpu --wd=0.001 --epochs=100 --lr=0.001 --poison_flag --trigger_id=$trigger_id --poison_ratio=0.5 --rand_loc=$rand_loc --rand_target=2
python ./test/test_gpu.py --need=$ngpu
python ./playground/poison_exp.py --experiment=poison --type=gtsrb --ngpu=$ngpu --wd=0.00 --epochs=200 --lr=0.001 --poison_flag --trigger_id=$trigger_id --poison_ratio=0.5 --rand_loc=$rand_loc --rand_target=2
python ./test/test_gpu.py --need=$ngpu
python ./playground/poison_exp.py --experiment=poison --type=cifar10 --ngpu=$ngpu  --wd=0.00 --epochs=100 --lr=0.001 --poison_flag --trigger_id=$trigger_id --poison_ratio=0.5 --rand_loc=$rand_loc --rand_target=2
python ./test/test_gpu.py --need=$ngpu
python ./playground/poison_exp.py --experiment=poison --type=cifar100 --ngpu=$ngpu --wd=0.00 --epochs=300 --lr=0.001 --poison_flag --trigger_id=$trigger_id --poison_ratio=0.5 --rand_loc=$rand_loc --rand_target=2
python ./test/test_gpu.py --need=$ngpu
