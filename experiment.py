'''
This file is used for parameters adjustment of area extraction.
raad './coinfig.ini' file and get parameters.
'''

import configparser
import bbox_searcher
import cv2

config = configparser.ConfigParser()
config_path = './roi_config.ini'
config.read(config_path)

img_path=config['DEFAULT']['img_path']
img_size=(int(config['DEFAULT']['img_size']), int(config['DEFAULT']['img_size']))
img = cv2.imread(img_path)
img = cv2.resize(img, img_size)


b_thresh = (int(config['DEFAULT']['b_thresh_l']),
    int(config['DEFAULT']['b_thresh_h']))

g_thresh = (int(config['DEFAULT']['g_thresh_l']),
    int(config['DEFAULT']['g_thresh_h']))

r_thresh = (int(config['DEFAULT']['r_thresh_l']),
    int(config['DEFAULT']['r_thresh_h']))

area_low_thresh_rate=float(config['DEFAULT']['area_low_thresh_rate'])
area_high_thresh_rate=float(config['DEFAULT']['area_high_thresh_rate'])

aspect_low_thresh=float(config['DEFAULT']['aspect_low_thresh'])
aspect_high_thresh=float(config['DEFAULT']['aspect_high_thresh'])

closing_ksize = (int(config['DEFAULT']['closing_ksize']), int(config['DEFAULT']['closing_ksize']))
opening_ksize = (int(config['DEFAULT']['opening_ksize']), int(config['DEFAULT']['opening_ksize']))

getter = bbox_searcher.Bbox_Getter(
    b_thresh, g_thresh, r_thresh,
    area_low_thresh_rate, area_high_thresh_rate,
    aspect_low_thresh, aspect_high_thresh,
    closing_ksize, opening_ksize
        )

boxes = getter.get_bbox(img)

# テスト どれか一つだけ実行
getter.describe_bbox(img, boxes)
# getter.describe_closed(img)
# getter.describe_opened(img)