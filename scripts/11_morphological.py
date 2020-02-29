# coding:utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt


class Morphological:
    def __init__(self, infile):
        self.infile = infile
        self.img = cv2.imread(self.infile)

    def erode_dilate(self):
        raw_gray=cv2.imread(self.infile,0)
        ret, raw_binary = cv2.threshold(raw_gray, 230, 255, cv2.THRESH_BINARY_INV)#230以下的变白
        inner_gray=cv2.imread('../images/i_inner.jpg',0)
        ret, inner_binnary = cv2.threshold(inner_gray, 230, 255, cv2.THRESH_BINARY_INV)#230以下的变白
        outer_gray=cv2.imread('../images/i_outer.jpg',0)
        ret, outer_binary = cv2.threshold(outer_gray, 230, 255, cv2.THRESH_BINARY_INV)#230以下的变白
        kernel=np.ones((5,5),np.uint8)
        erosion=cv2.erode(raw_binary,kernel,iterations=1)#腐蚀
        dilation=cv2.dilate(raw_binary,kernel,iterations=1)#膨胀
        closing=cv2.morphologyEx(inner_binnary,cv2.MORPH_CLOSE,kernel)#闭运算，先膨胀再腐蚀，去小孔
        opening=cv2.morphologyEx(outer_binary,cv2.MORPH_OPEN,kernel)#开运算，先腐蚀再膨胀，去噪声
        gradient=cv2.morphologyEx(raw_binary,cv2.MORPH_GRADIENT,kernel)#梯度：膨胀-腐蚀
        tophat=cv2.morphologyEx(outer_binary,cv2.MORPH_TOPHAT,kernel)#礼帽：开运算-原图
        blackhat=cv2.morphologyEx(inner_binnary,cv2.MORPH_BLACKHAT,kernel)#黑帽：闭运算-原图
        titles=['raw','erosion','dilation','gradient']
        images=[raw_binary,erosion,dilation,gradient]
        for i in range(4):
            plt.subplot(1,4,i+1),plt.imshow(images[i],'gray')
            plt.title(titles[i])
            plt.xticks([]),plt.yticks([])
        plt.show()
        titles=['inner','closing','blackhat','outer','opening','tophat']
        images=[inner_binnary,closing,blackhat,outer_binary,opening,tophat]
        for i in range(6):
            plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
            plt.title(titles[i])
            plt.xticks([]),plt.yticks([])
        plt.show()


    def get_kernel(self):
        #正方形
        print(cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)))
        #[[1 1 1 1 1]
        # [1 1 1 1 1]
        # [1 1 1 1 1]
        # [1 1 1 1 1]
        # [1 1 1 1 1]]
        #椭圆形
        print(cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
        #[[0 0 1 0 0]
        # [1 1 1 1 1]
        # [1 1 1 1 1]
        # [1 1 1 1 1]
        # [0 0 1 0 0]]
        #十字形
        print(cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5)))
        #[[0 0 1 0 0]
        # [0 0 1 0 0]
        # [1 1 1 1 1]
        # [0 0 1 0 0]
        # [0 0 1 0 0]]


if __name__ == '__main__':
    m = Morphological('../images/i.jpg')
    m.get_kernel()
