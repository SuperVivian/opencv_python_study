# coding:utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt


class Smoothing:
    def __init__(self, infile):
        self.infile = infile
        self.img = cv2.imread(self.infile)

    def blur(self):
        blur=cv2.blur(self.img,(5,5))
        gaussian = cv2.GaussianBlur(self.img, (5, 5), 0)
        median=cv2.medianBlur(self.img,5)
        bilateral=cv2.bilateralFilter(self.img,8,75,75)
        # 9 邻域直径，两个75 分别是空间/灰度值相似性高斯函数标准差
        titles = ['raw', 'mean','gaussian','median','bilateral']
        images = [self.img, blur,gaussian,median,bilateral]
        for i in range(5):
            img=cv2.cvtColor(images[i],cv2.COLOR_BGR2RGB)
            plt.subplot(2,3,i+1),plt.imshow(img)
            plt.title(titles[i])
            plt.xticks([]),plt.yticks([])
        plt.show()


if __name__ == '__main__':
    smooth = Smoothing('images/test1.jpg')
    smooth.blur()
