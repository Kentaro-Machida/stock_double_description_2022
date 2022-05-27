# 入力データの分割
---
## csvs
* ストックの成長を追跡した結果を記したエクセルファイルとcsvファイルが格納されている.
* "stock_data.xlsx" のシートからそれぞれのcsvファイルを出力することができる. 
---

### 命名規則
* 数字の行とアルファベットもしくは50音の行の組み合わせでストックの個体を命名している.  
* 1は一重咲き(single), 8は八重咲き(double), 4は枯れてしまった個体(dead), 
空欄は開花しなかった個体(not_bloomed)を表している. 
---
## making_data_code
* labeling.py  
列ごとにフォルダ分割されているストックの画像データをcsvファイルに従って,
singleフォルダ, doubleフォルダ, deadフォルダ, not_bloomedフォルダの
4つに分割を行う. 
```
data/
　├ csvs/
　├ dataset/
　├ making_data_code/
　├ raw_datas/
　│　├ 8_12/
　│　├ 8_16/
　│　├ 8_18/
　└ making_data_code/
```
入力パスとして8_12や8_16などのフォルダを指定する. それぞれの日付フォルダの構造は以下のようになっている. 
```
8_12/
　├ a1/...
　├ b1/...
　├ c1/...
　├ d1/...
　│　├ d1_01.jpeg
　│　├ d2_02.jpeg
　│　├ ...
　│　├ d1_64.jpeg
　├ ...
　└ p1/

```
出力フォルダは以下のようになる. 
```
output
　├ dead/...
　├ not_bloomed/...
　├ single/...
　├ double/
　│　├ a1_23.jpeg
　│　├ d1_62.jpeg
　│　├ ...
　│　├ p1_04.jpeg
```
---
* split_dataset.py
singleフォルダとdoubleフォルダを入力で指定し, 任意の比率でtrain, val, test
のフォルダに分割する.
```
input
　├ single/...
　├ double/
　│　├ a1_23.jpeg
　│　├ d1_62.jpeg
　│　├ ...
　│　├ p1_04.jpeg
```

```
output
　├ train/
　│　├ single/
　│　└ double/
　├ val/
　│　├ single/
　│　└ double/
　├ test/
　│　├ single/
　│　└ double/
```

---
## sample image
コードを試すためのサンプルイメージ
