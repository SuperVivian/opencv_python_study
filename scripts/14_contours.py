# coding:utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt

class Contours:
    def __init__(self, infile):
        self.infile = infile
        self.img = cv2.imread(self.infile)

    def draw(self):
        #【1】把图像变灰，再二值化
        gray=cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        ret,thresh_img=cv2.threshold(gray,127,255,0)
        #【2】寻找轮廓：会改变原始图像
        image,contours,hierarchy=cv2.findContours(thresh_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        #retr_tree,chain_approx_simple
        #【3】绘制所有轮廓
        img1=cv2.drawContours(self.img,contours,-1,(0,255,0),3)
        #【4】绘制某个轮廓
        img2=cv2.drawContours(self.img,contours,1,(0,255,0),3)
        titles = ['all', 'one']
        imgs = [img1, img2]
        for i in range(2):
            plt.subplot(1, 2, i + 1), plt.imshow(cv2.cvtColor(imgs[i],cv2.COLOR_BGR2RGB))
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.show()

    def features(self):
        pass

    def properties(self):
        pass


if __name__ == '__main__':
    c = Contours('../images/test2.jpg')
    c.draw()
