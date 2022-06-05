import pandas as pd
import os
import glob
import numpy as np

"""
csvファイルに従ってフォルダ分割したファイルが
正しく分割されているかを試すプログラム
"""

def label_and_img_name(target_dir:str)->dict:
    # label_dir_list = ['../hoge/dead', '../hoge/single',...]
    label_dir_list = glob.glob(target_dir+'/*')

    label_dict = {}
    for label_dir in label_dir_list:
        img_list = os.listdir(label_dir)
        label_name = label_dir.split("/")[-1]
        
        num_column_list = []
        # 'c2-32.jpeg'などの画像名を想定
        for img_name in img_list:
            column = img_name[0]
            number = img_name[3:5]
            num_column_list.append({"column":column,"number":number})

        label_dict[label_name] = num_column_list
    return label_dict

# もし分類ミスがあればその旨を出力
def check_label(label_dict:dict, csv_path:str)->None:

    df = pd.read_csv(csv_path)
    label_num_change = {"dead":4, "single":1, "double":8, "not_bloomed":np.nan}
    error=False

    for label, dict_list in label_dict.items():
        check_label_num = label_num_change[label]
        for col_and_num in dict_list:
            num = (int(col_and_num["number"])-1)%16
            acc_label = df.loc[num,col_and_num["column"]]

            # not_bloomedのチェック
            if pd.isna(acc_label):
                flag = pd.isna(check_label_num)
                if flag==False:
                    print(f"{col_and_num} in label:{label}(No.{num+1}) is wrong. acc: {acc_label}")
                    error=True
            # single, dead, doubleのチェック
            elif acc_label!=check_label_num:
                print(f"{col_and_num} in label:{label}(No.{num+1}) is wrong. acc: {acc_label}")
                error=True

    if error==False:
        print("error is nothing.")        


def main():
    csv_path = '../csvs/A_X.csv'
    target_dir = '../labeled_data/9_10_out'
    label_dict = label_and_img_name(target_dir)
    check_label(label_dict, csv_path)

if __name__=='__main__':
    main()