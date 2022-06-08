from cv2 import COLOR_BGR2RGB
import cv2
import numpy as np
import torch
import torchvision.models as model
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchvision import transforms

from bbox_searcher import Bbox_Getter

def get_bbox(img:np.ndarray)->list:
    
    b_thresh = (0, 255)
    g_thresh = (128, 255)
    r_thresh = (0, 255)

    area_low_thresh_rate = 0.5
    area_high_thresh_rate = 1.5

    aspect_low_thresh=0.7
    aspect_high_thresh=1.3

    closing_ksize=(5, 5)
    opening_ksize=(10, 10)

    getter = Bbox_Getter(
        b_thresh, g_thresh, r_thresh,
        area_low_thresh_rate, area_high_thresh_rate,
        aspect_low_thresh, aspect_high_thresh,
        closing_ksize, opening_ksize
            )

    boxes = getter.get_bbox(img)

    return boxes

def get_RoIs(img:np.ndarray, boxes, out_size=224) -> list:
    """
    input
    image, list of bounding box
    output
    list of np.ndarray(W, H, C)
    """
    cut_img_list = []
    for x1, y1, x2, y2 in boxes:
        RoI = img[y1:y2, x1:x2]
        RoI = cv2.resize(RoI, (out_size, out_size))
        cut_img_list.append(RoI)

    return cut_img_list

def tensor_preprocess(img:np.ndarray)->torch.Tensor:
    normalizer = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tensorer = transforms.ToTensor()
    input = tensorer(img)
    input = normalizer(input)
    input = torch.unsqueeze(input, 0)
    return input

def main():
    img_size = (448, 448)
    img_path = './data/sample_images/test_img.jpg'

    img = cv2.imread(img_path)
    img = cv2.resize(img, img_size)
    img = cv2.cvtColor(img, COLOR_BGR2RGB)

    model_path = "./trained_models/Efficientnet_b3.pth"
    trained_model = torch.load(model_path ,map_location=torch.device('cpu'))

    boxes = get_bbox(img)

    # RoI画像を抽出
    RoIs = get_RoIs(img, boxes)

    # すべてのRoIに前処理関数をかます
    RoIs = list(map(tensor_preprocess, RoIs))

    # すべてのRoI4次元tensorを連結
    input_batch = torch.cat(RoIs)

    # 確率の計算
    logits = trained_model(input_batch)
    m = torch.nn.Softmax(dim=1)
    outs = m(logits)

    # 実際に描画を行ってみる
    num = 0
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    for x1, y1, x2, y2 in boxes:
        prob = float(outs[num][0])
        color = (1.0,1- prob, prob)
        bbox = mpatches.Rectangle(
            (x1, y1), (x2-x1), (y2-y1), fill=False, edgecolor=color, linewidth=3)
        ax.add_patch(bbox)
        ax.text(x1,y1-3, str(int(prob*100)), color=color)
        num +=1

    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()