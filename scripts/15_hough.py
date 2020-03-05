# coding:utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt


class Hough:
    def __init__(self, infile):
        self.infile = infile
        self.img = cv2.imread(self.infile)

    def Line(self):
        img_copy1 = self.img.copy()
        img_copy2 = self.img.copy()
        img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        for rho, theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(img_copy1, (x1, y1), (x2, y2), (0, 0, 255), 2)
        for line in lines:
            for rho, theta in line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(img_copy2, (x1, y1), (x2, y2), (0, 0, 255), 2)
        titles = ['raw', 'line_one', 'line_all']
        imgs = [self.img, img_copy1, img_copy2]
        for i in range(3):
            plt.subplot(1, 3, i + 1), plt.imshow(cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB))
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.show()

    def LineP(self):
        img_copy1 = self.img.copy()
        img_copy2 = self.img.copy()
        img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)
        minLineLength = 100  # 线的最短长度
        maxLineGap = 10  # 两条线段之间的最大间隔。如果小于此值，这两条直线就被看成是一条直线。
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)
        for x1, y1, x2, y2 in lines[0]:
            cv2.line(img_copy1, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img_copy2, (x1, y1), (x2, y2), (0, 255, 0), 2)
        titles = ['raw', 'line_p_one', 'line_p_all']
        imgs = [self.img, img_copy1, img_copy2]
        for i in range(3):
            plt.subplot(1, 3, i + 1), plt.imshow(cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB))
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.show()

    def circle(self):
        img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.medianBlur(img_gray, 5)
        cimg = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=50, param2=30, minRadius=0, maxRadius=0)
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
        titles = ['raw', 'circle']
        imgs = [self.img, cimg]
        for i in range(2):
            plt.subplot(1, 2, i + 1), plt.imshow(cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB))
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.show()

    def water_shed(self):
        copy=self.img.copy()
        gray = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)
        # 【1】Otsu's二值化
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # 【2】用开运算去除白噪声
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        # 【3】用膨胀确定背景区域
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        # 【4】确定前景区域
        # 第二个参数0,1,2 分别表示CV_DIST_L1, CV_DIST_L2 , CV_DIST_C
        dist_transform = cv2.distanceTransform(opening, 1, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        # 【5】Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        # 【6】创建标签，标记标签
        ret, markers1 = cv2.connectedComponents(sure_fg)
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers1 + 1
        # Now, mark the region of unknown with zero
        markers[unknown == 255] = 0
        # 【7】实施分水岭算法
        markers3 = cv2.watershed(copy, markers)
        copy[markers3 == -1] = [255, 0, 0]
        #### 画图
        titles = ['raw', 'thresh','opening','sure_bg','dist_transform','sure_fg',
                  'unknown','markers1','markers','markers3','result']
        imgs = [self.img,thresh, opening,sure_bg,dist_transform,sure_fg,
                unknown,markers1,markers,markers3,copy]
        for i in range(11):
            plt.subplot(2, 6, i + 1), plt.imshow(imgs[i],'gray')
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.show()


if __name__ == '__main__':
    h = Hough('../images/water_coins.jpg')
    h.water_shed()
