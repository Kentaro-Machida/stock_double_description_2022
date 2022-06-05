from genericpath import exists
import glob
import os
import sys
import shutil
import numpy as np
import configparser

"""
single, doubleのディレクトリにある画像をtrain, val, testディレクトリに分割
"""

def split_pth_list(pth_list:list, train_rate:float, val_rate:float)->dict:
    """
    input: 画像パスのリスト
    output:train, val, testの画像パス辞書
    """
    np.random.shuffle(pth_list)

    list_num = len(pth_list)
    train_num = int(list_num*train_rate)
    val_num = int(list_num*val_rate)
    test_num = list_num - train_num - val_num
    print(f"{train_num=}")
    print(f"{val_num=}")
    print(f"{test_num=}")

    train_list = pth_list[0:train_num]
    val_list = pth_list[train_num : train_num + val_num]
    test_list = pth_list[train_num + val_num : ]

    return {"train": train_list, "val":val_list, "test":test_list}

def copy_data_to_split_dir(train_val_test_dict:dict, label:str, output_dir:str)->None:
    # train, val, test それぞれに対して
    for k, pth_list in train_val_test_dict.items():
        save_dir = os.path.join(output_dir,k,label)
        os.makedirs(save_dir, exist_ok=True)

        # それぞれのデータに対して
        for src_pth in pth_list:
            shutil.copy(src_pth, save_dir)

def main():
    parser = configparser.ConfigParser()
    parser.read('split_dataset.ini')
    config = parser['DEFAULT']

    single_path = config['single_path']
    double_path = config['double_path']

    output_dir = config['output_dir']

    RANDOM_SEED = int(config['RANDOM_SEED'])
    TRAIN_LATE = float(config['TRAIN_LATE'])
    VAL_LATE = float(config['VAL_LATE'])
    TEST_LATE = float(config['TEST_LATE'])

    single_pth_list = glob.glob(os.path.join(single_path, "*.jpeg"))
    double_pth_list = glob.glob(os.path.join(double_path, "*.jpeg"))

    np.random.seed(RANDOM_SEED)

    os.makedirs(output_dir,exist_ok=True)

    print('single')
    single_tvt_dict = split_pth_list(single_pth_list,TRAIN_LATE,VAL_LATE)
    print('double')
    double_tvt_dict = split_pth_list(double_pth_list,TRAIN_LATE,VAL_LATE)

    copy_data_to_split_dir(single_tvt_dict,'single',output_dir)
    copy_data_to_split_dir(double_tvt_dict,'double',output_dir)

if __name__=="__main__":
    main()