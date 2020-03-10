import cv2
import matplotlib.pyplot as plt
import numpy as np

class Flow:
    def __init__(self, infile):
        self.infile = infile
        self.img = cv2.imread(self.infile)

    def lucas(self):
        cap = cv2.VideoCapture('../images/slow.flv')
        #ShiTomasi角点检测所需参数
        feature_params = dict(maxCorners=100,
                              qualityLevel=0.3,
                              minDistance=7,
                              blockSize=7)
        #lucas kanade光流所需参数
        # maxLevel 为使用的图像金字塔层数
        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        #随机颜色
        color = np.random.randint(0, 255, (100, 3))
        #读取第一帧，并进行角点检测
        ret, old_frame = cap.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)
        while True:
            ret, frame = cap.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # calculate optical flow能够获取点的新位置
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            # Select good points
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            # draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
                frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
            img = cv2.add(frame, mask)
            cv2.imshow('frame', img)
            k = cv2.waitKey(30)  # & 0xff
            if k == 27:
                break
            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
        cv2.destroyAllWindows()
        cap.release()

    def dense(self):
        cap = cv2.VideoCapture("../images/vtest.avi")
        ret, frame1 = cap.read()
        #对整幅图像计算
        prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[..., 1] = 255
        while True:
            ret, frame2 = cap.read()
            next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            # 稠密光流
            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            #返回带有光流向量（u，v）的双通道数组。
            #通过计算我们能得到光流的大小和方向。
            #我们使用颜色对结果进行编码以便于更好的观察。
            #方向对应于H（Hue）通道，大小对应于V（Value）通道。
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            #显示帧
            cv2.imshow('frame2', frame2)
            cv2.imshow('flow', bgr)
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break
            elif k == ord('s'):
                cv2.imwrite('opticalfb.png', frame2)
                cv2.imwrite('opticalhsv.png', bgr)
            #更新图像
            prvs = next

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    f= Flow('../images/test1.jpg')
    f.dense()
