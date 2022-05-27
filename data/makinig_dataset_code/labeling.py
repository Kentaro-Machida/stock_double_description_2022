import os 
import glob
import shutil
import pandas as pd
import configparser
"""
csvファイルに従って画像をsingle, doubleフォルダに振り分ける
"""


# 4つのディレクトリを作成する
# すでに存在していても消去するので注意
def make_4_dirs(out_dir:str)->None:
    
    os.makedirs(os.path.join(out_dir,'single'),exist_ok=True)
    os.makedirs(os.path.join(out_dir,'double'),exist_ok=True)
    os.makedirs(os.path.join(out_dir,'dead'),exist_ok=True)
    os.makedirs(os.path.join(out_dir,'not_bloomed'),exist_ok=True)

def make_dataset(data_dir:str,out_dir:str,csv_file:str):
    
    dir_alpabet_list = glob.glob(os.path.join(data_dir,'*'))

    df = pd.read_csv(csv_file)
    natural_alp = list(df.columns)

    for dir_alpabet,natural in zip(dir_alpabet_list,natural_alp):
        images_2d_list = []
        target_dir = './' + dir_alpabet + '/*'
        files = glob.glob(target_dir)
        files.sort()

        for i in range(16):
            images_2d_list.append(files[i::16])


        single_path = os.path.join(out_dir,'single')
        double_path = os.path.join(out_dir,'double')
        dead_path = os.path.join(out_dir,'dead')
        not_bloomed_path = os.path.join(out_dir,'not_bloomed')
        judge_list = df[natural]

        # 画像に欠けがあったら移動を行わない。
        if(len(files)%16 != 0 or len(files)==0):
            print('The number of images in folder ' + dir_alpabet + ' is not multiple of 16.')
        # ディレクトリの数が16でなければ移動を行わない。
        elif(len(dir_alpabet_list)!=16):
            print(dir_alpabet_list)
            print('The number of directory is not 16.')
        else:
            for img_list, judge in zip(images_2d_list,judge_list):
                for p in img_list:
                    if judge == 8:
                        try:
                            shutil.copy(p,double_path)
                        except:
                            print(p,'has already moved!')
                    elif judge == 1:
                        try:
                            shutil.copy(p,single_path)
                        except:
                            print(p,'has already moved!')
                    elif judge == 4:
                        try:
                            shutil.copy(p,dead_path)
                        except:
                            print(p,'has already moved!')
                    else:
                        try:
                            shutil.copy(p,not_bloomed_path)
                        except:
                            print(p,'has not bloomed yet')

def main():
    parser = configparser.ConfigParser()
    parser.read('labeling_config.ini')
    config = parser['DEFAULT']

    data_dir = config['data_dir']
    out_dir = config['out_dir']
    csv_pth = config['csv_pth']

    make_4_dirs(out_dir)
    make_dataset(data_dir, out_dir, csv_pth)

if __name__=='__main__':
    main()