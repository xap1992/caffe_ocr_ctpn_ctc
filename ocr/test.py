import sys
import cv2

sys.path.insert(0, './ctpn')
sys.path.insert(0, './ctc')
# sys.path.insert(0, '/home/xap/caffe/caffe-ctpn-ctc-ocr/python')
from math import *
import caffe
from ctc_ocr import ocr
from ctpn_demo import detector_lines

import datetime

caffe.set_mode_gpu()

CTPN_NET_DEF_FILE = "./ctpn/models/deploy.prototxt"
CTPN_MODEL_FILE = "./ctpn/models/ctpn_trained_model.caffemodel"



im = cv2.imread('./ctpn/1755.jpg')
# im = cv2.imread('./b.jpg')
starttime=datetime.datetime.now()
ctpn = detector_lines()
detetor = detector_lines()
text_lines = ctpn.get_text_lines(im, CTPN_NET_DEF_FILE, CTPN_MODEL_FILE)
text_recs=ctpn.boxes_rotate_rect(text_lines)

for rec in text_recs:
    pt1 = (rec[0], rec[1])
    pt2 = (rec[2], rec[3])
    pt3 = (rec[6], rec[7])
    pt4 = (rec[4], rec[5])
    partImg = ctpn.dumpRotateImage(im, degrees(atan2(pt2[1] - pt1[1], pt2[0] - pt1[0])), pt1, pt2, pt3, pt4)
    ctc = ocr()
    value = ctc.getImageValue(partImg)
    print value
print (datetime.datetime.now()-starttime)
ctpn.draw_boxes_rotate(im, text_recs)
