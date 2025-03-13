# PBVS2025_MAVC

# to Train

python simple_baselines.py --train_folder ../Unicorn_Dataset/Unicorn_Dataset/SAR_Train/ --val_folder ../Unicorn_Dataset/Unicorn_Dataset/SAR_VAL/ --model_path train.pth --epochs 10 --learning_rate 0.0001 --batch_size 64

# to generate submission file


python simple_baselines.py --test_only True --test_folder [test_folder] --model_path [model]

test folder is the competition folder in following structure

- test
    - examples
        - img1
        - img2
        - img3 ....
