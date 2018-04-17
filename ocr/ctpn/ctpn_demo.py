import cv2
import sys
import numpy as np
from math import *


from cfg import Config as cfg
from detectors import TextProposalDetector, TextDetector
from other import resize_im, CaffeModel


class detector_lines:
    def get_text_lines(self, im, NET_DEF_FILE, MODEL_FILE):
        # initialize the detectors
        text_proposals_detector = TextProposalDetector(CaffeModel(NET_DEF_FILE, MODEL_FILE))
        text_detector = TextDetector(text_proposals_detector)
        im, f = resize_im(im, cfg.SCALE, cfg.MAX_SCALE)
        text_lines = text_detector.detect(im)
        return text_lines / f

    def boxes_rotate_rect(self, bboxes):
        """
            boxes: bounding boxes
        """
        text_recs = np.zeros((len(bboxes), 8), np.int)
        index = 0
        for box in bboxes:
            b1 = box[6] - box[7] / 2
            b2 = box[6] + box[7] / 2
            x1 = box[0]
            y1 = box[5] * box[0] + b1
            x2 = box[2]
            y2 = box[5] * box[2] + b1
            x3 = box[0]
            y3 = box[5] * box[0] + b2
            x4 = box[2]
            y4 = box[5] * box[2] + b2

            disX = x2 - x1
            disY = y2 - y1
            width = np.sqrt(disX * disX + disY * disY)
            fTmp0 = y3 - y1
            fTmp1 = fTmp0 * disY / width
            x = np.fabs(fTmp1 * disX / width)
            y = np.fabs(fTmp1 * disY / width)
            if box[5] < 0:
                x1 -= x
                y1 += y
                x4 += x
                y4 -= y
            else:
                x2 += x
                y2 += y
                x3 -= x
                y3 -= y
            text_recs[index, 0] = x1
            text_recs[index, 1] = y1
            text_recs[index, 2] = x2
            text_recs[index, 3] = y2
            text_recs[index, 4] = x3
            text_recs[index, 5] = y3
            text_recs[index, 6] = x4
            text_recs[index, 7] = y4
            index = index + 1
        return text_recs

    def draw_boxes_rotate(self, im, text_recs):
        c = tuple((0,0,255))
        for rec in text_recs:
            pt1 = (rec[0], rec[1])
            pt2 = (rec[2], rec[3])
            pt3 = (rec[4], rec[5])
            pt4 = (rec[6], rec[7])
            cv2.line(im, pt1, pt2, c, 2)
            cv2.line(im, pt1, pt3, c, 2)
            cv2.line(im, pt4, pt2, c, 2)
            cv2.line(im, pt3, pt4, c, 2)
        cv2.imshow('result', im)
        cv2.waitKey(0)

    def dumpRotateImage(self, img, degree, pt1, pt2, pt3, pt4):
        height, width = img.shape[:2]
        heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
        widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
        matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
        matRotation[0, 2] += (widthNew - width) / 2
        matRotation[1, 2] += (heightNew - height) / 2
        imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
        pt1 = list(pt1)
        pt3 = list(pt3)

        [[pt1[0]], [pt1[1]]] = np.dot(matRotation, np.array([[pt1[0]], [pt1[1]], [1]]))
        [[pt3[0]], [pt3[1]]] = np.dot(matRotation, np.array([[pt3[0]], [pt3[1]], [1]]))
        imgOut = imgRotation[int(pt1[1]):int(pt3[1]), int(pt1[0]):int(pt3[0])]
        height, width = imgOut.shape[:2]
        return imgOut
