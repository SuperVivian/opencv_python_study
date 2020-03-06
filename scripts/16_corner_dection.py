# coding:utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt


class Corner:
    def __init__(self, infile):
        self.infile = infile
        self.img = cv2.imread(self.infile)

    def harris(self):
        # 【1】把原图处理为float32的灰度图像
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        # 输入图像必须是float32，最后一个参数在0.04 到0.05 之间
        # 【2】Harris角点检测
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)  # 膨胀，为了突出角点
        copy1 = self.img.copy()
        copy2 = self.img.copy()
        copy1[dst > 0.01 * dst.max()] = [0, 255, 0]  # 大于某个阈值（最大值的0.01）的角点变成緑色
        copy2[dst > 0.5 * dst.max()] = [0, 255, 0]
        # 【3】画图
        titles = ['raw', 'low_thresh', 'hign_thresh']
        imgs = [self.img, copy1, copy2]
        for i in range(3):
            plt.subplot(1, 3, i + 1), plt.imshow(cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB))
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.show()

    def subfix(self):
        copy = self.img.copy()
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # 【1】Harris角点检测
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)
        ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
        dst = np.uint8(dst)
        # 【2】找到重心
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
        # 【3】设置迭代终止条件
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        # 【4】亚像素级角点检测
        # 返回值由角点坐标组成的一个数组（而非图像）
        corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)
        # 画图
        res = np.hstack((centroids, corners))
        # np.int0 可以用来省略小数点后面的数字（非四􃮼五入）。
        res = np.int0(res)
        copy[res[:, 1], res[:, 0]] = [0, 0, 255]
        copy[res[:, 3], res[:, 2]] = [0, 255, 0]
        plt.imshow(cv2.cvtColor(copy, cv2.COLOR_BGR2RGB))
        plt.show()

    def track(self):
        copy = self.img.copy()
        gray = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
        # 返回的结果是[[ 311., 250.]] 两层括号的数组。
        corners = np.int0(corners)
        for i in corners:
            x, y = i.ravel()
            cv2.circle(copy, (x, y), 3, (0,0,255), 3)
        plt.imshow(cv2.cvtColor(copy, cv2.COLOR_BGR2RGB)), plt.show()


if __name__ == '__main__':
    c = Corner('../images/board.jpg')
    c.track()
