# coding:utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt


def nothing(x):
    pass

class Slider:
    def __init__(self, infile):
        self.infile = infile
        self.img = cv2.imread(self.infile)

    def slider(self):
        img=np.zeros((300,513,3),np.uint8)
        cv2.namedWindow('image')
        cv2.createTrackbar('R','image',0,255,nothing)#0是滑动条默认位置，255是滑动条最大值。
        cv2.createTrackbar('G','image',0,255,nothing)#nothing是回调函数，默认参数是滑动条位置。
        cv2.createTrackbar('B','image',0,255,nothing)
        switch='0:0FF\n1:ON'
        cv2.createTrackbar(switch,'image', 0, 1, nothing)
        while(1):
            cv2.imshow('image',img)
            k=cv2.waitKey(1)&0xFF
            #在某些系统上，waitKey（）可能会返回一个不仅仅编码ASCII密钥的值。
            #当OpenCV使用GTK作为其后端GUI时，已知在Linux上发生了一个错误库。
            #在所有系统上，我们可以通过读取返回值中的最后一个字节来确保我们只提取SCII键代码
            if k==27:
                break
            r=cv2.getTrackbarPos('R','image')
            g = cv2.getTrackbarPos('G', 'image')
            b = cv2.getTrackbarPos('B', 'image')
            s=cv2.getTrackbarPos(switch,'image')
            if s==0:
                img[:]=0
            else:
                img[:]=[b,g,r]
        cv2.destroyAllWindows()
if __name__ == '__main__':
    s = Slider('../images/test3.jpg')
    s.slider()
