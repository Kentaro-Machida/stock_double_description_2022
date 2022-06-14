# ストックの八重鑑別
## 概要
アブラナ科のストックという植物の八重鑑別を行う.  

## 処理の流れ
- 非NNの領域抽出により対象の葉の座標を獲得. 
- 獲得した領域をNNに入力. 分類を行う. 

## 環境構築
```
pip install -r requirements.txt
```

## 領域抽出
領域抽出は以下の流れで行う. 
1. RGB色空間の閾値処理により, 画像を2値化
2. クロージング処理によるノイズ消去
3. オープニング処理により, 縮小した領域を再拡大し, 小領域を結合. 
4. 2値画像に対して8近傍の領域抽出処理を行い, オブジェクトにラベルを付与. 
5. ラベリングされた2値画像をもとにIoUを抽出.
6. IoUの面積の平均値を計算し, 外れ値を除去する. 
7. IoUのアスペクト比を計算し, 正方形から遠いものを除去する. 

bbox_searcher.py として実装されており, 以下に使用例を示す.  

```
import bbox_searcher
import cv2

    masking_type = "hsv"
    img_size=(448,448)

    b_thresh = (0, 255)
    g_thresh = (128, 255)
    r_thresh = (0, 255)

    h_thresh = (110, 140)
    s_thresh = (180, 255)
    v_thresh = (150, 255)

    area_low_thresh_rate = 0.2
    area_high_thresh_rate = 9.5

    img_path = './data/sample_images/test_img.jpg'
    img = cv2.imread(img_path)
    img = cv2.resize(img, img_size)

    if masking_type=="rgb":
        getter = Bbox_Getter(b_thresh, g_thresh, r_thresh,
            area_low_thresh_rate, area_high_thresh_rate, masking_type=masking_type)
        boxes = getter.get_bbox(img)
    
    elif masking_type=='hsv':
        getter = Bbox_Getter(h_thresh, s_thresh, v_thresh,
            area_low_thresh_rate, area_high_thresh_rate, masking_type=masking_type)
        boxes = getter.get_bbox(img)

```
boxesは, タプル(x1, y1, x2, y2)である. x1, y1はBboxの左上座標であり, 
x2, y2は右下座標である. xは横軸, yは縦軸を表す. 

## パラメータ調整
experiment.pyを実行することで物体検出がうまく行われているか確認できる.
もし, パラメータの変更が必要であれば, roi_config.iniを変更することで, 条件を
変えることができる.

```
python experiment.py
```

## dataディレクトリ
収集した画像データをcsvファイルのラベルに従ってディレクトリ分割するコード
が入っている. データそのものは重いので入っていない. 

## 分類モデルの学習
finetuning_stock.ipynbは, torchvisionにあるモデルをロードし, 
finetuningを行うnotebookである. 全てのセルを実行することで, 訓練モデル
が出力される. パラメータセルの変数でモデルの選択, ハイパーパラメータの選択等
を行う. 行われる処理は以下の通りである.

1. torchvisionによるモデルのロード
2. pytorchによるモデルの学習
3. モデルを保存

## 訓練済みモデルによるRoIの分類
roi_double_description.pyは, bbox_searcherクラスで抽出したRoIを分類モデルに
入力し, 八重鑑別を行い, Bboxと確率値を描画した画像を出力するコードである. パラメータ等は
roi_config.iniを参照する. 
```
python roi_double_description.py
```