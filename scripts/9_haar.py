#coding-utf8
import cv2
import matplotlib.pyplot as plt
import numpy as np

class Haar:
    def __init__(self, infile):
        self.infile = infile
        self.img = cv2.imread(self.infile)

    def classify(self):
        copy=self.img.copy()
        #【1】加载XML分类器
        face_cascade = cv2.CascadeClassifier('E:\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('E:\opencv\sources\data\haarcascades\haarcascade_eye.xml')
        #【2】以灰度格式加载输入图像或视频
        gray = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)
        #【3】在图像中检测面部。如果检测到面部，会返回面部所在区域Rect(x,y,w,h)。
        #       一旦获得这个位置，我们可以创建一个ROI并在其中进行眼部检测。
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)#img、ScaleFactor、minNeighbors
        for (x, y, w, h) in faces:
            #画脸(彩色图像）
            copy = cv2.rectangle(copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
            #确定脸部ROI
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = copy[y:y + h, x:x + w]
            #【4】检测眼睛(灰度图像)
            eyes = eye_cascade.detectMultiScale(roi_gray)#img、ScaleFactor、minNeighbors
            for (ex, ey, ew, eh) in eyes:
                #画眼睛（彩色图像)
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        titles = ['raw', 'faces&eyes']
        imgs = [self.img,copy]
        for i in range(2):
            plt.subplot(1, 2, i + 1), plt.imshow(cv2.cvtColor(imgs[i],cv2.COLOR_BGR2RGB))
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.show()

if __name__ == '__main__':
    d= Haar('../images/people.jpg')
    d.classify()
