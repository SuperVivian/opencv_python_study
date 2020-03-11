#coding-utf8
import cv2
import matplotlib.pyplot as plt
import numpy as np

class Inpaint:
    def __init__(self, infile):
        self.infile = infile
        self.img = cv2.imread(self.infile)

    def inpaint_img(self):
        mask = cv2.imread('../images/mask2.png', 0)
        #用mask修补图像
        dst = cv2.inpaint(self.img, mask, 3, cv2.INPAINT_TELEA)
        dst2 = cv2.inpaint(self.img, mask, 3, cv2.INPAINT_NS)
        #画图
        titles = ['raw', 'mask','telea','ns']
        imgs = [self.img,mask,dst,dst2]
        for i in range(4):
            plt.subplot(2, 2, i + 1)
            if i==1:
                plt.imshow(imgs[i], 'gray')
            else:
                plt.imshow(cv2.cvtColor(imgs[i],cv2.COLOR_BGR2RGB))
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.show()

if __name__ == '__main__':
    d= Inpaint('../images/messi_2.jpg')
    d.inpaint_img()
