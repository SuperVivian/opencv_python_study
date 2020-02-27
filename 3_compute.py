import cv2
import numpy as np
import matplotlib.pyplot as plt

class Compute:
    def __init__(self, infile):
        self.infile = infile
        self.img=cv2.imread(self.infile)

    def add(self):
        x=np.uint8([250])
        y=np.uint8([10])
        print(cv2.add(x,y))#260=>255
        print(x+y)#260%256=4
        plt.subplot(131), plt.title('raw'), plt.imshow(cv2.cvtColor(self.img,cv2.COLOR_BGR2RGB)),plt.xticks([]), plt.yticks([])
        plt.subplot(132),plt.title('numpy'),plt.imshow(cv2.cvtColor(self.img+self.img,cv2.COLOR_BGR2RGB)),plt.xticks([]), plt.yticks([])
        plt.subplot(133), plt.title('add'), plt.imshow(cv2.cvtColor(cv2.add(self.img,self.img),cv2.COLOR_BGR2RGB)),plt.xticks([]), plt.yticks([])
        plt.show()

    def blend(self):
        img1=cv2.cvtColor(self.img,cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(cv2.imread('images/test2.jpg'),cv2.COLOR_BGR2RGB)
        dst=cv2.addWeighted(self.img,0.5,img2,0.5,0)#alpha,beta,gamma是混合参数
        dst1=cv2.cvtColor(dst,cv2.COLOR_BGR2RGB)
        plt.subplot(131), plt.title('raw1'), plt.imshow(img1), plt.xticks([]), plt.yticks([])
        plt.subplot(132), plt.title('raw2'), plt.imshow(img2), plt.xticks([]), plt.yticks([])
        plt.subplot(133), plt.title('blend'), plt.imshow(dst1), plt.xticks([]), plt.yticks([])
        plt.show()

    def bit_compute(self):
        raw=cv2.cvtColor(self.img,cv2.COLOR_BGR2RGB)
        img1=self.img
        img2=cv2.imread('images/logo.png')
        #让img2添加到img1上
        rows,cols,channels=img2.shape
        roi=img1[0:rows,0:cols]#在img上选取img2那么大的区域
        #创建mask
        img2gray=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        ret,mask=cv2.threshold(img2gray,175,255,cv2.THRESH_BINARY)
        mask_inv=cv2.bitwise_not(mask)
        print(type(ret))
        print(type(mask))
        img1_bg=cv2.bitwise_and(roi,roi,mask=mask)
        img2_fg=cv2.bitwise_and(img2,img2,mask=mask_inv)

        dst=cv2.add(img1_bg,img2_fg)
        img1[0:rows,0:cols]=dst
        img1=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)

        plt.subplot(231), plt.title('img1'), plt.imshow(raw), plt.xticks([]), plt.yticks([])
        plt.subplot(232), plt.title('img2'), plt.imshow(img2), plt.xticks([]), plt.yticks([])
        plt.subplot(234), plt.title('mask'), plt.imshow(mask,'gray'), plt.xticks([]), plt.yticks([])
        plt.subplot(235), plt.title('mask_inv'), plt.imshow(mask_inv,'gray'), plt.xticks([]), plt.yticks([])
        plt.subplot(236), plt.title('add'), plt.imshow(img1), plt.xticks([]), plt.yticks([])
        plt.show()


if __name__ == '__main__':
    compute = Compute('images/test1.jpg')
    compute.bit_compute()