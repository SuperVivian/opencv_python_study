import cv2
import matplotlib.pyplot as plt
import numpy as np

class Shift:
    def __init__(self, infile):
        self.infile = infile
        self.img = cv2.imread(self.infile)

    def meanshift(self):
        #【1】创建cap对象，从文件打开视频
        cap = cv2.VideoCapture('../images/slow.flv')
        #【2】读取视频第一帧
        ret, frame = cap.read()
        #【3】设置窗口的起始位置
        print(frame.shape)#640x360
        r, h, c, w = 186, 25, 305, 78  #硬编码
        track_window = (c, r, w, h)
        #【4】设置追踪对象的区域
        roi = frame[r:r + h, c:c + w]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)#转到HSV颜色空间
        mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))# 将低亮度的值忽略掉
        roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])#计算直方图
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        #【5】设置终止条件：精度先达到1或者迭代次数先达到10次时，停止迭代
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        #【6】分析视频
        while True:
            ret, frame = cap.read()
            if ret is True:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)#直方图反向投影
                #应用meanshift，获得新的窗口位置
                ret, track_window = cv2.meanShift(dst, track_window, term_crit)
                #把新窗口在图像上画出来
                x, y, w, h = track_window
                print(track_window)
                img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0), 2)
                cv2.imshow('img2', img2)

                k = cv2.waitKey(60)  # & 0xff
                if k == 27:
                    break
            else:#最后一帧
                break
        cv2.destroyAllWindows()
        cap.release()

    def camshift(self):
        cap = cv2.VideoCapture('../images/slow.flv')
        ret, frame = cap.read()
        #确定跟踪对象
        r, h, c, w = 186, 25, 305, 78  #硬编码
        track_window = (c, r, w, h)
        roi = frame[r:r + h, c:c + w]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        #计算直方图
        roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        # 终止条件
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        #在视频中跟踪目标
        while True:
            ret, frame = cap.read()
            if ret is True:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
                #CamShift
                ret, track_window = cv2.CamShift(dst, track_window, term_crit)
                #画出来
                pts = cv2.boxPoints(ret)
                pts = np.int0(pts)
                print('len pts:', len(pts), pts)
                img2 = cv2.polylines(frame, [pts], True, (255, 0, 0), 2)
                cv2.imshow('img2', img2)
                k = cv2.waitKey(60)  # & 0xff
                if k == 27:
                    break
            else:
                break
        cv2.destroyAllWindows()
        cap.release()

if __name__ == '__main__':
    s = Shift('../images/test1.jpg')
    s.camshift()
