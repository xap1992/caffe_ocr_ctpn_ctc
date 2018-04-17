# coding:utf-8
import sys

import caffe
import cv2
import chardet
import numpy as np


def singleton(cls, *args, **kw):
    instances = {}

    def _singleton():
        global net
        global label

        def loadLabel(labelFile):
            with open(labelFile, 'r') as f:
                text = f.read()
            type = chardet.detect(text)
            allText = text.decode(type['encoding'])
            label = allText.split('\r\n')
            return label

        if cls not in instances:
            instances[cls] = cls(*args, **kw)
            label = loadLabel('./ctc/inception-bn-res-blstm/label.txt')
            model_def = './ctc/inception-bn-res-blstm/deploy.prototxt'
            model_weights = './ctc/inception-bn-res-blstm/model.caffemodel'
            net = caffe.Net(model_def, model_weights, caffe.TEST)
        return instances[cls]

    return _singleton


@singleton
class ocr(object):
    def doImage(self, image, dis=False):
        if dis:
            ele = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            image = cv2.erode(image, ele)
            image = 255 - image
        h, w, _ = image.shape
        nh = 32
        nw = int((32.0 * w) / h)
        image = cv2.resize(image, (nw, nh))
        if nw < 800:
            image = cv2.copyMakeBorder(image, 0, 0, 0, 800 - nw, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        elif nw > 800:
            image = image[:, :800, :]
        img = np.transpose(image, (2, 0, 1))
        return img

    def imageProcess(self, image):
        # 图片预处理
        img = self.doImage(image)

        net.blobs['data'].data[...] = img
        output = net.forward()
        value = output['result'][0].reshape([-1])
        return value

    def getImageValue(self, img):
        output_prob = self.imageProcess(img)
        line = ''
        for output in output_prob:
            index = int(output)
            if index != -1:
                line += label[index]
        return line
