# coding:utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt


class Pyramid:
    def __init__(self, infile):
        self.infile = infile
        self.img = cv2.imread(self.infile)

    def pyr_up_down(self):
        lower_reso=cv2.pyrDown(self.img)#降采样一次，长宽缩小为1/2
        higher_reso2=cv2.pyrUp(lower_reso)#放大到二倍
        cv2.imshow('raw', self.img)
        cv2.imshow('low',lower_reso)
        cv2.imshow('high', higher_reso2)
        cv2.waitKey(0)
        print(self.img.shape)#(640, 640, 3)
        print(lower_reso.shape)#(320, 320, 3)
        print(higher_reso2.shape)#(640, 640, 3)

    def img_blend(self):
        #【1】读入两幅图像
        A=cv2.imread('../images/apple.jpg')
        B=cv2.imread('../images/orange.jpg')
        # 【2】构建两幅图像的高斯金字塔（6层）
        G=A.copy()
        gpA=[G]
        for i in range(6):
            G=cv2.pyrDown(G)
            gpA.append(G)

        G=B.copy()
        gpB=[G]
        for i in range(6):
            G=cv2.pyrDown(G)
            gpB.append(G)

        # 【3】根据高斯金字塔计算拉普拉斯金字塔-计算公式?
        # Li = Gi -PyrUp(Gi+1)
        lpA=[gpA[5]]#L[5]=G[5]
        for i in range(5,0,-1):#倒着计算，至下而上
            L=cv2.subtract(gpA[i-1],cv2.pyrUp(gpA[i]))#L[4]=G[4]-pyrUp(G[5])
            lpA.append(L)

        lpB=[gpB[5]]
        for i in range(5,0,-1):
            L=cv2.subtract(gpB[i-1],cv2.pyrUp(gpB[i]))
            lpB.append(L)

        # 【4】在拉普拉斯的每一层进行图像融合（苹果的左边，橘子的右边）
        LS=[]
        for la,lb in zip(lpA,lpB):
            rows,cols,dpt=la.shape
            ls=np.hstack((la[:,0:cols//2],lb[:,cols//2:]))
            LS.append(ls)



        # 【5】根据融合后的图像金字塔重建原始图像
        res=[]
        ls_=LS[0]#LS是融合后的拉普拉斯图像，0-5是从小到大
        res.append(ls_)
        for i in range(1,6):
            ls_=cv2.pyrUp(ls_)
            # ls_=cv2.add(ls_,LS[i])
            res.append(ls_)

        for i in range(6):
            img=cv2.cvtColor(res[i],cv2.COLOR_BGR2RGB)
            plt.subplot(2,3,i+1),plt.imshow(img)
            plt.xticks([]),plt.yticks([])
        plt.show()

        real=np.hstack((A[:,:cols//2],B[:,cols//2:]))
        titles = ['apple', 'orange','real','blend']
        images = [A, B,real,ls_]
        for i in range(4):
            img=cv2.cvtColor(images[i],cv2.COLOR_BGR2RGB)
            plt.subplot(1,4,i+1),plt.imshow(img)
            plt.title(titles[i])
            plt.xticks([]),plt.yticks([])
        plt.show()

if __name__ == '__main__':
    p = Pyramid('../images/test1.jpg')
    p.img_blend()
