# coding:utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt


class Threshold:
    def __init__(self, infile):
        self.infile = infile
        self.img = cv2.imread(self.infile)

    def simple_thresh(self):
        gray=cv2.imread(self.infile,0)
        ret,thresh1=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
        ret, thresh2 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        ret, thresh3= cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)
        ret, thresh4 = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO)
        ret, thresh5 = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO_INV)
        titles=['raw','binary','binary_inv','trunc','tozero','tozero_inv']
        images=[gray,thresh1,thresh2,thresh3,thresh4,thresh5]
        for i in range(6):
            plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
            plt.title(titles[i])
            plt.xticks([]),plt.yticks([])
        plt.show()

    def adaptive_threshold(self):
        gray=cv2.imread(self.infile,0)
        gray=cv2.medianBlur(gray,5)#中值滤波
        ret,th1=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
        th2=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                                  cv2.THRESH_BINARY,11,2)
        th3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                    cv2.THRESH_BINARY, 11, 2)
        titles=['raw','simple binary','adaptive mean','adaptive gaussian']
        images=[gray,th1,th2,th3]
        for i in range(4):
            plt.subplot(1,4,i+1),plt.imshow(images[i],'gray')
            plt.title(titles[i])
            plt.xticks([]),plt.yticks([])
        plt.show()

    def otsu_s(self):
        pass

if __name__ == '__main__':
    trans = Threshold('../images/test1.jpg')
    trans.adaptive_threshold()
