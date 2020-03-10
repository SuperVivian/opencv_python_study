# coding:utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt


class Contours:
    def __init__(self, infile):
        self.infile = infile
        self.img = cv2.imread(self.infile)

    def draw(self):
        # 【1】把图像变灰，再二值化
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        ret, thresh_img = cv2.threshold(gray, 127, 255, 0)
        # 【2】寻找轮廓：会改变原始图像
        image, contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # retr_tree,chain_approx_simple
        # 【3】绘制所有轮廓
        img1 = cv2.drawContours(self.img, contours, -1, (0, 255, 0), 3)
        # 【4】绘制某个轮廓
        img2 = cv2.drawContours(self.img, contours, 1, (0, 255, 0), 3)
        titles = ['all', 'one']
        imgs = [img1, img2]
        for i in range(2):
            plt.subplot(1, 2, i + 1), plt.imshow(cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB))
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.show()

    def features(self):
        # 【0】检测图像轮廓
        img = cv2.imread('../images/shandian.jpg', 0)
        cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        ret, thresh = cv2.threshold(img, 127, 255, 0)
        img, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # 【1】轮廓的矩
        cnt = contours[0]
        M = cv2.moments(cnt)  # 矩以一个字典的形式返回。可以求出质心坐标。
        # cx = int(M['m10']/M['m00'])
        # cy = int(M['m01']/M['m00'])
        # 【2】轮廓面积：可以用函数cv2.contourArea，也可以使用矩（0 阶矩）['m00']
        area = cv2.contourArea(cnt)
        # 【3】轮廓周长：弧长。第二参数指定对象的形状是闭合的（True），还是打开的
        perimeter = cv2.arcLength(cnt, True)
        # 【4】轮廓近似：近似到另外一种由更少点组成的轮廓形状，新轮廓的点的数目由我们设定的准确度来决定
        epsilon = 0.1 * cv2.arcLength(cnt, True)  # 从原始轮廓到近似轮廓的最大距离
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        # 【5】凸包
        hull = cv2.convexHull(cnt)  # points轮廓、hull输出、clockwise方向标志、returnPoints默认返回凸包上的点。否则返回轮廓点的索引。
        # 【6】凸性检测：检测一个曲线是不是凸的。
        k = cv2.isContourConvex(cnt)
        # 【7】直边界矩形
        x, y, w, h = cv2.boundingRect(cnt)  # x,y是左上角坐标
        img_rect = cv2.rectangle(cimg.copy(), (x, y), (x + w, y + h), (0, 255, 0), 2)
        # 【8】旋转的边界矩形
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        img_min_rect = cv2.drawContours(cimg.copy(), [box], 0, (0, 0, 255), 2)
        # 【9】最小外接圆
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        img_circle = cv2.circle(cimg.copy(), center, radius, (0, 255, 0), 2)
        # 【10】椭圆拟合：返回旋转边界矩形的内切圆
        ellipse = cv2.fitEllipse(cnt)
        img_ellip = cv2.ellipse(cimg.copy(), ellipse, (0, 255, 0), 2)
        # 【11】直线拟合
        rows, cols = img.shape[:2]
        [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
        lefty = int((-x * vy / vx) + y)
        righty = int(((cols - x) * vy / vx) + y)
        img_line = cv2.line(cimg.copy(), (cols - 1, righty), (0, lefty), (0, 255, 0), 2)
        titles = ['raw', 'img_rect', 'img_min_rect', 'img_circle', 'img_ellip', 'img_line']
        imgs = [img, img_rect, img_min_rect, img_circle, img_ellip, img_line]
        for i in range(6):
            plt.subplot(2, 3, i + 1), plt.imshow(imgs[i], 'gray')
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.show()

    def properties(self):
        img = cv2.imread('../images/test2.jpg', 0)
        ret, thresh = cv2.threshold(img, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, 1, 2)
        cnt = contours[0]

        # 【1】边界矩形的长宽比
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        # 【2】Extent：轮廓面积与边界矩形面积的比
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        rect_area = w * h
        extent = float(area) / rect_area
        # 【3】Solidity：轮廓面积与凸包面积的比
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area
        # 【4】Equivalent Diameter：与轮廓面积相等的圆形的直径
        equi_diameter = np.sqrt(4 * area / np.pi)
        # 【5】方向：对象的方向，下面的方法还会返回长轴和短轴的长度
        (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
        # 【6】掩模和像素点
        mask = np.zeros(img.shape, np.uint8)
        # 这里一定要使用参数-1, 绘制填充的的轮廓
        cv2.drawContours(mask, [cnt], 0, 255, -1)
        pixelpoints = np.transpose(np.nonzero(mask))
        # pixelpoints = cv2.findNonZero(mask)
        # 【7】最大值和最小值及它们的位置
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(img, mask=mask)
        # 【8】平均颜色及平均灰度
        mean_val = cv2.mean(img, mask=mask)
        # 【9】极点：一个对象最上面，最下面，最左边，最右边的点
        leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
        rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
        topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
        bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])

    def find_defects(self):
        img = self.img
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img_gray, 127, 255, 0)
        image,contours, hierarchy = cv2.findContours(thresh, 2, 1)
        cnt = contours[0]#第一个轮廓
        # 【1】找到凸缺陷
        hull = cv2.convexHull(cnt, returnPoints=False)
        defects = cv2.convexityDefects(cnt, hull)
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]#起点，终点，最远的点，到最远点的近似距离
            #返回结果的前三个值是轮廓点的索引，还要到轮廓点中去找它们
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            cv2.line(img, start, end, [0, 255, 0], 2)#将起点和终点用一条绿线连接
            cv2.circle(img, far, 5, [0, 0, 255], -1)#在最远点画一个圆圈
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # 【2】求解图像中的一个点到一个对象轮廓的最短距离
        # 如果点在轮廓的外部，返回值为负。如果在轮廓上，返回值为0。如果在轮廓内部，返回值为正。
        dist = cv2.pointPolygonTest(cnt, (50, 50), True)#轮廓、point
        print(dist)
        # 此函数的第三个参数是measureDist。如果设置为True，就会计算最短距离。
        # 如果是False，只会判断这个点与轮廓之间的位置关系（返回值为+1，-1，0）。

    def match(self):
        #【1】得到二值化图像
        img1 = cv2.imread('../images/shandian.jpg', 0)
        img2 = cv2.imread('../images/shandian2.jpg', 0)
        img3 = cv2.imread('../images/shandian3.jpg', 0)
        ret1, thresh1 = cv2.threshold(img1, 127, 255, 0)
        ret2, thresh2 = cv2.threshold(img2, 127, 255, 0)
        ret3, thresh3 = cv2.threshold(img3, 127, 255, 0)
        #【2】检测轮廓
        image1,contours, hierarchy = cv2.findContours(thresh1, 2, 1)
        cnt1 = contours[0]
        image2,contours, hierarchy = cv2.findContours(thresh2, 2, 1)
        cnt2 = contours[0]
        image3,contours, hierarchy = cv2.findContours(thresh3, 2, 1)
        cnt3 = contours[0]
        #【3】用轮廓进行匹配
        _1_1 = cv2.matchShapes(cnt1, cnt1, 1, 0.0)
        _1_2 = cv2.matchShapes(cnt1, cnt2, 1, 0.0)
        _1_3 = cv2.matchShapes(cnt1, cnt3, 1, 0.0)
        print(_1_1,_1_2,_1_3)
        #画图
        titles = ['1', '2', '3']
        imgs = [img1, img2, img3]
        for i in range(3):
            plt.subplot(2, 3, i + 1), plt.imshow(imgs[i], 'gray')
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.show()

if __name__ == '__main__':
    c = Contours('../images/shandian.jpg')
    c.match()
