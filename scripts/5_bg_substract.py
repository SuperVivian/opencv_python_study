import cv2
import matplotlib.pyplot as plt
import numpy as np

class Bg_sub:
    def __init__(self, infile):
        self.infile = infile
        self.img = cv2.imread(self.infile)

    def mog(self):
        cap = cv2.VideoCapture("../images/vtest.avi")
        fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
        # 可选参数 比如 进行建模场景的时间长度 高斯混合成分的数量-阈值等
        while True:
            ret, frame = cap.read()
            fgmask = fgbg.apply(frame)
            cv2.imshow('frame', fgmask)
            k = cv2.waitKey(1)  # & 0xff
            if k == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


    def mog2(self):
        cap = cv2.VideoCapture("../images/vtest.avi")
        fgbg = cv2.createBackgroundSubtractorMOG2()
        while True:
            ret, frame = cap.read()
            fgmask = fgbg.apply(frame)
            cv2.imshow('frame', fgmask)
            k = cv2.waitKey(30)  # & 0xff
            if k == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def mor(self):
        cap = cv2.VideoCapture("../images/vtest.avi")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
        counter = 0
        while True:
            ret, frame = cap.read()
            fgmask = fgbg.apply(frame)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            cv2.imshow('frame', fgmask)  # 前 120 帧
            counter += 1
            print(counter)
            k = cv2.waitKey(1)  # & 0xff
            if k == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    b= Bg_sub('../images/test1.jpg')
    b.mor()
