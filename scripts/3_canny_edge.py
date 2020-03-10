# coding:utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt

def nothing(x):
    pass

class Edge:
    def __init__(self, infile):
        self.infile = infile
        self.img = cv2.imread(self.infile)

    def canny(self):
        test=cv2.imread('../images/test2.jpg',0)
        cv2.namedWindow('image')
        cv2.createTrackbar('min','image',0,200,nothing)
        cv2.createTrackbar('max','image',100,300,nothing)
        while(1):
            min=cv2.getTrackbarPos('min','image')
            max=cv2.getTrackbarPos('max','image')
            edges = cv2.Canny(test, min, max)
            #【1】高斯平滑：去除噪声
            #【2】sobel梯度计算：求梯度大小和方向
            #【3】非极大值抑制：同梯度方向上的最大梯度才是边界
            #【4】滞后阈值：min梯度以下的不算边界。
            # max和min梯度之间的，和max以上梯度的像素相连的才算边界。
            cv2.imshow('image',edges)
            k=cv2.waitKey(1)&0xFF
            if k==27:
                break

if __name__ == '__main__':
    e = Edge('../images/test3.jpg')
    e.canny()
