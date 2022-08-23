'''
This file is used for parameters adjustment of area extraction.
raad './coinfig.ini' file and get parameters.
'''

import configparser
import bbox_searcher
import cv2

# get Bounding Boxes with parameters in "roi_config.ini"
config = configparser.ConfigParser()
config_path = './roi_config.ini'
config.read(config_path)

img_path=config['DEFAULT']['img_path']
img_size=(int(config['DEFAULT']['img_size']), int(config['DEFAULT']['img_size']))
img = cv2.imread(img_path)
img = cv2.resize(img, img_size)

masking_type=config['DEFAULT']['masking_type']

b_thresh = (int(config['DEFAULT']['b_thresh_l']),
    int(config['DEFAULT']['b_thresh_h']))

g_thresh = (int(config['DEFAULT']['g_thresh_l']),
    int(config['DEFAULT']['g_thresh_h']))

r_thresh = (int(config['DEFAULT']['r_thresh_l']),
    int(config['DEFAULT']['r_thresh_h']))

h_thresh = (int(config['DEFAULT']['h_thresh_l']),
    int(config['DEFAULT']['h_thresh_h']))

s_thresh = (int(config['DEFAULT']['s_thresh_l']),
    int(config['DEFAULT']['s_thresh_h']))

v_thresh = (int(config['DEFAULT']['v_thresh_l']),
    int(config['DEFAULT']['v_thresh_h']))

area_low_thresh_rate=1/float(config['DEFAULT']['num_per_height_h'])
area_high_thresh_rate=1/float(config['DEFAULT']['num_per_height_l'])

aspect_low_thresh=float(config['DEFAULT']['aspect_low_thresh'])
aspect_high_thresh=float(config['DEFAULT']['aspect_high_thresh'])

closing_ksize = (int(config['DEFAULT']['closing_ksize']), int(config['DEFAULT']['closing_ksize']))
opening_ksize = (int(config['DEFAULT']['opening_ksize']), int(config['DEFAULT']['opening_ksize']))

getter = bbox_searcher.Bbox_Getter(
    b_thresh, g_thresh, r_thresh,
    area_low_thresh_rate, area_high_thresh_rate,
    masking_type,
    aspect_low_thresh, aspect_high_thresh,
    closing_ksize, opening_ksize,
    h_thresh, s_thresh, v_thresh
    )

boxes = getter.get_bbox(img)

# テスト どれか一つだけ実行
getter.describe_binary(img)
getter.describe_bbox(img, boxes)
getter.describe_closed(img)
getter.describe_opened(img)