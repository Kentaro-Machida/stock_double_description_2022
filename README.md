# ストックの八重鑑別
## 概要
アブラナ科のストックという植物の八重鑑別を行う.  

## 処理の流れ
- 非NNの領域抽出により対象の葉の座標を獲得. 
- 獲得した領域をNNに入力. 分類を行う. 

## 環境構築
```
pip install opencv-python-headless==4.5.5.64
pip install numpy
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

bbox_searcher.py として実装されており, 以下のように使用する. 

```
import bbox_searcher.Bbox_Getter

b_thresh = (0, 255)
g_thresh = (128, 255)
r_thresh = (0, 255)

area_low_thresh_rate = 0.5
area_high_thresh_rate = 1.5

aspect_low_thresh=0.7
aspect_high_thresh=1.3

closing_ksize=(5, 5)
opening_ksize=(10, 10)

img_path = './data/test_img.jpg'
img = cv2.imread(img_path)
img = cv2.resize(img, (448, 448))

getter = bbox_searcher.Bbox_Getter(
    b_thresh, g_thresh, r_thresh,
    area_low_thresh_rate, area_high_thresh_rate,
    aspect_low_thresh, aspect_high_thresh,
    closing_ksize, opening_ksize
        )

boxes = getter.get_bbox(img)
```
boxesは, タプル(x1, y1, x2, y2)である. x1, y1はBboxの左上座標であり, 
x2, y2は右下座標である. xは横軸, yは縦軸を表す. 

## パラメータ調整
experiment.pyを実行することで物体検出がうまく行われているか確認できる.
もし, パラメータの変更が必要であれば, config.iniを変更することで, 条件を
変えることができる.

```
python experiment.py
```