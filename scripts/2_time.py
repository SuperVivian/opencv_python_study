import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
class Time:
    def __init__(self, infile):
        self.infile = infile
        self.img=cv2.imread(self.infile)


    def get_time(self):
        e1 = time.time()#从参考点到这个函数被执行的时钟数
        for i in range(5, 49, 2):#用不同的核来进行中值模糊
            img=cv2.medianBlur(self.img, i)
        e2 =  time.time()#从参考点到这个函数被执行的时钟数
        t = (e2 - e1)
        print(t)  # 1.193786859512329


    def get_cv_time(self):
        e1 = cv2.getTickCount()#从参考点到这个函数被执行的时钟数
        for i in range(5, 49, 2):#用不同的核来进行中值模糊
            img=cv2.medianBlur(self.img, i)
        e2 = cv2.getTickCount()#从参考点到这个函数被执行的时钟数
        t = (e2 - e1) / cv2.getTickFrequency()  # 时钟频率 或者 每秒钟的时钟数
        print(t)  # 1.3738476076698682



    #：一般情况下OpenCV 的函数要比Numpy 函数快。所以对于相同的操
    #作最好使用OpenCV 的函数。当然也有例外，尤其是当使用Numpy 对视图
    #（而非复制）进行操作时。


    def IPy_Time(self):
        cv2.useOptimized()#检测是否开启了默认优化
        cv2.setUseOptimized(False)#关闭默认优化
        #%timeit 魔法命令



if __name__ == '__main__':
    t = Time('../images/test1.jpg')
    t.get_time()