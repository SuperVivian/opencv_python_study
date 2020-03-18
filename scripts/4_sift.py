# coding:utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt


class Sift:
    def __init__(self, infile):
        self.infile = infile
        self.img = cv2.imread(self.infile)

    def sift(self):
        img=self.img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp = sift.detect(gray, None)
        # 计算关键点描述符
        # 使用函数 sift.compute() 来 计算 些关键点的描述符。例如
        # kp, des = sift.compute(gray, kp)
        kp, des = sift.detectAndCompute(gray, None)
        img = cv2.drawKeypoints(gray, kp, img)
        plt.subplot(121), plt.imshow(gray,'gray'), plt.title('raw'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(img,'gray'), plt.title('keypoints'), plt.xticks([]), plt.yticks([])
        plt.show()

    def surf(self):
        img = cv2.imread('../images/white_point.jpg',0)#读入灰度图像
        surf = cv2.xfeatures2d.SURF_create(400)#创建surf对象
        print(surf.getHessianThreshold())
        #【1】当HessianThreshold为400时
        kp, des = surf.detectAndCompute(img, None)
        img2 = cv2.drawKeypoints(img, kp, None, (255, 0, 0), 4)
        print(len(kp))#285个关键点
        #【2】增大HessianThreshold
        surf.setHessianThreshold(20000)
        kp, des = surf.detectAndCompute(img, None)
        img3 = cv2.drawKeypoints(img, kp, None, (255, 0, 0), 4)
        print(len(kp))#151
        #【3】u-surf：所有的关键点的朝向都是一致的。它比前面的快很多
        print(surf.getUpright())#Check upright flag, if it False, set it to True
        surf.setUpright(True)
        kp = surf.detect(img, None)
        img4 = cv2.drawKeypoints(img, kp, None, (255, 0, 0), 4)
        #【4】改描述符64维为128维
        print(surf.descriptorSize())#64
        print(surf.getExtended())#如果结果是64维，就代表这个bool值为false
        surf.setExtended(True)
        kp, des = surf.detectAndCompute(img, None)
        print(surf.descriptorSize())
        #展示结果
        plt.subplot(221), plt.imshow(img,'gray'), plt.title('raw'), plt.xticks([]), plt.yticks([])
        plt.subplot(222), plt.imshow(img2,'gray'), plt.title('keypoints1'), plt.xticks([]), plt.yticks([])
        plt.subplot(223), plt.imshow(img3, 'gray'), plt.title('keypoints2'), plt.xticks([]), plt.yticks([])
        plt.subplot(224), plt.imshow(img4, 'gray'), plt.title('keypoints3'), plt.xticks([]), plt.yticks([])
        plt.show()

    def fast(self):
        img = cv2.imread('../images/blox.jpg', 0)
        #【1】用默认值初始化fast对象
        fast = cv2.FastFeatureDetector_create()
        #【2】检测并绘制关键点
        kp = fast.detect(img, None)
        img2 = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0))
        #【3】打印所有默认参数
        print("Threshold: ", fast.getThreshold())
        print("nonmaxSuppression: ", fast.getNonmaxSuppression())
        print("neighborhood: ", fast.getType())
        print("Total Keypoints with nonmaxSuppression: ", len(kp))
        #【4】不使用非最大值抑制
        #使用极大值抑制的方法可以解决检测到的特征点相连的问题
        fast.setNonmaxSuppression(0)
        kp = fast.detect(img, None)
        print("Total Keypoints without nonmaxSuppression: ", len(kp))
        img3 = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0))
        #展示结果
        plt.subplot(131), plt.imshow(img, 'gray'), plt.title('raw'), plt.xticks([]), plt.yticks([])
        plt.subplot(132), plt.imshow(img3,'gray'), plt.title('more'), plt.xticks([]), plt.yticks([])
        plt.subplot(133), plt.imshow(img2,'gray'), plt.title('less'), plt.xticks([]), plt.yticks([])
        plt.show()

    def brief(self):
        img = cv2.imread('../images/blox.jpg', 0)
        #【1】初始化FAST检测器
        star = cv2.xfeatures2d.StarDetector_create()
        #【2】初始化BRIEF提取器
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        #【3】用star找到关键点
        kp = star.detect(img, None)
        #【4】用BRIEF计算描述符
        kp, des = brief.compute(img, kp)
        #函数brief:getInt(′bytes′) 会以字节格式给出nd 的大小，默认值为32。
        print("brief.descriptorSize: ",brief.descriptorSize())

    def orb(self):
        img = cv2.imread('../images/blox.jpg', 0)
        #【1】初始化ORB 特征检测器
        orb = cv2.ORB_create()
        #【2】用ORB找到关键点
        kp = orb.detect(img, None)
        #【3】用ORB计算描述符
        kp, des = orb.compute(img, kp)
        #【4】只画出关键点位置。不考虑大小、方向。
        img2 = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
        #展示结果
        plt.subplot(121), plt.imshow(img, 'gray'), plt.title('raw'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(img2,'gray'), plt.title('ret'), plt.xticks([]), plt.yticks([])
        plt.show()


if __name__ == '__main__':
    c = Sift('../images/home.jpg')
    c.orb()
