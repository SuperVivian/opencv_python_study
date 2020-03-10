# coding:utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt


class Gradients:
    def __init__(self, infile):
        self.infile = infile
        self.img = cv2.imread(self.infile)

    def gradient(self):
        gray=cv2.imread(self.infile,0)
        #参数1,0为只在x方向求一阶导数，最大可以求2阶导数。y方向同理。
        #参数cv2.CV_64F结果图像的深度，可以使用-1，与原图像保持一致（np.uint8)

        # sobelx=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)
        # sobely=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5)
        # laplacian=cv2.Laplacian(gray,cv2.CV_64F)

        sobelx=np.uint8(np.abs(cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)))
        sobely=np.uint8(np.abs(cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5)))
        laplacian=np.uint8(np.abs(cv2.Laplacian(gray,cv2.CV_64F)))

        titles = ['raw', 'sobelx','sobely','laplacian']
        images = [gray, sobelx,sobely,laplacian]
        for i in range(4):
            plt.subplot(1,4,i+1),plt.imshow(images[i],'gray')
            plt.title(titles[i])
            plt.xticks([]),plt.yticks([])
        plt.show()


if __name__ == '__main__':
    g = Gradients('../images/test3.jpg')
    g.gradient()
