# coding:utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt


class GeoTransform:
    def __init__(self, infile):
        self.infile = infile
        self.img = cv2.imread(self.infile)

    def scale(self):
        res1=cv2.resize(self.img,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)

        height,width=self.img.shape[:2]
        res2=cv2.resize(self.img,(2*width,2*height),interpolation=cv2.INTER_CUBIC)

        print(self.img.shape)
        print(res1.shape)
        print(res2.shape)


    def translate(self):
        rows,cols,ch=self.img.shape
        M=np.float32([[1,0,100],[0,1,50]])
        dst=cv2.warpAffine(self.img,M,(cols,rows))
        print(self.img.shape)
        print(dst.shape)
        plt.subplot(121), plt.title('raw'), plt.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)), plt.xticks(
            []), plt.yticks([])
        plt.subplot(122), plt.title('translate'), plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)), plt.xticks(
            []), plt.yticks([])
        plt.show()

    def rotate(self):
        rows=self.img.shape[0]
        cols=self.img.shape[1]
        M=cv2.getRotationMatrix2D((cols/2,rows/2),45,0.6)
        dst=cv2.warpAffine(self.img,M,(2*cols,2*rows))
        print(self.img.shape)
        print(dst.shape)

        plt.subplot(121), plt.title('raw'), plt.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)), plt.xticks(
            []), plt.yticks([])
        plt.subplot(122), plt.title('rotate'), plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)), plt.xticks(
            []), plt.yticks([])
        plt.show()

    def affine(self):
        rows,cols,ch=self.img.shape
        pts1=np.float32([[50,50],[200,50],[50,200]])
        pts2=np.float32([[10,100],[200,50],[100,240]])
        M=cv2.getAffineTransform(pts1,pts2)
        dst=cv2.warpAffine(self.img,M,(cols,rows))
        print(self.img.shape)
        print(dst.shape)
        plt.subplot(121), plt.title('raw'), plt.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)), plt.xticks(
            []), plt.yticks([])
        plt.subplot(122), plt.title('warp'), plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)), plt.xticks(
            []), plt.yticks([])
        plt.show()

    def perspective(self):
        rows,cols,ch=self.img.shape
        pts1=np.float32([[56,65],[368,52],[28,387],[389,390]])
        pts2=np.float32([[0,0],[300,0],[0,300],[300,300]])
        M=cv2.getPerspectiveTransform(pts1,pts2)
        dst=cv2.warpPerspective(self.img,M,(300,300))
        print(self.img.shape)
        print(dst.shape)
        plt.subplot(121), plt.title('raw'), plt.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)), plt.xticks(
            []), plt.yticks([])
        plt.subplot(122), plt.title('perspective'), plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)), plt.xticks(
            []), plt.yticks([])
        plt.show()

if __name__ == '__main__':
    trans = GeoTransform('../images/test1.jpg')
    trans.translate()
