# coding:utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt


class Histograms:
    def __init__(self, infile):
        self.infile = infile
        self.img = cv2.imread(self.infile)

    def accu_paint(self):
        raw_gray=cv2.imread(self.infile,0)
        raw_color=cv2.imread(self.infile)

        #【1】plot统计单通道直方图，用plot绘制
        plt.hist(raw_gray.ravel(),256,[0,256])
        plt.show()
        #【2】cv统计三通道直方图，用plot绘制
        color=('b','g','r')
        for i,col in enumerate(color):
            histr=cv2.calcHist([raw_color],[i],None,[256],[0,256])
            plt.plot(histr,color=col)
            plt.xlim([0,256])
        plt.show()
        #【3】numpy方法统计直方图，用plot绘制
        np_hist1, bins = np.histogram(raw_gray.ravel(), 256, [0, 256])  # cv函数比此函数快40倍
        # img.ravel()把图像转为一维数组
        np_hist2 = np.bincount(raw_gray.ravel(), minlength=256)  # 比histogram快十倍
        titles=['histogram','bincount']
        hists=[np_hist1,np_hist2]
        for i in range(2):
            plt.subplot(1,2,i+1),plt.plot(hists[i])
            plt.title(titles[i])
        plt.show()


    def mask_hist(self):
        raw_gray=cv2.imread(self.infile,0)
        mask=np.zeros(raw_gray.shape[:2],np.uint8)#mask为全黑的图像
        mask[100:500,100:600]=255#mask的该区域变白
        masked_img=cv2.bitwise_and(raw_gray,raw_gray,mask=mask)
        hist_full=cv2.calcHist([raw_gray],[0],None,[256],[0,256])
        hist_mask=cv2.calcHist([raw_gray],[0],mask,[256],[0,256])
        titles = ['raw_gray', 'mask','masked_img']
        imgs = [raw_gray, mask,masked_img]
        for i in range(3):
            plt.subplot(2, 2, i + 1), plt.imshow(imgs[i],'gray')
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.subplot(2,2,4),plt.plot(hist_full),plt.plot(hist_mask)
        plt.xlim([0,256])
        plt.show()


if __name__ == '__main__':
    h = Histograms('../images/test2.jpg')
    h.mask_hist()
