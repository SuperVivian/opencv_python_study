# coding:utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt

class Draw:
    def __init__(self, infile):
        self.infile = infile
        self.img = cv2.imread(self.infile)
        self.drawing = False  # true if mouse is pressed
        self.mode = True  # if True, draw rectangle. Press 'm' to toggle to curve
        self.ix, self.iy = -1, -1

    def draw(self):
        img = np.zeros((512, 512, 3), np.uint8)
        # 【1】线
        cv2.line(img, (0, 0), (511, 511), (255, 0, 0), 5)  # 原图、起点、终点、蓝色、粗细
        #【2】长方形
        cv2.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)  # 原图，左上角、右下角、绿色、粗细
        #【3】圆
        cv2.circle(img, (447, 63), 63, (0, 0, 255), -1)  # 原图、圆心、半径、红色、粗细
        #【4】椭圆
        cv2.ellipse(img, (256, 256), (100, 50), 0, 0, 180, (255,0,255),-1)#紫色
        # 原图、中心点、长轴/短轴长度、椭圆沿逆时针方向旋转的角度、
        # 椭圆弧沿顺时针方向起始的角度和结结束角度（0-360是整个椭圆）、颜色、粗细
        #【5】多边形
        pts = np.array([[100, 200], [200, 350], [300, 202], [500, 100]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        # 这里reshape 的第一个参数为-1, 表明这一维的长度是根据后面的维度的计算出来的。
        cv2.polylines(img,[pts],True,(0,255,255),2)#黄色
        # 【6】文字
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(img, 'OpenCV', (10, 500), font, 4, (255, 255, 255), 2)#白色
        cv2.imshow('image',img)
        cv2.waitKey(0)

if __name__ == '__main__':
    d = Draw('../images/test2.jpg')
    d.draw()
