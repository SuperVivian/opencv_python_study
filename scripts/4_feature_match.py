# coding:utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt


class FeatureMatch:
    def __init__(self, infile):
        self.infile = infile
        self.img = cv2.imread(self.infile)

    def bf_orb(self):
        img1 = cv2.imread('../images/box.png', 0)  # queryImage
        img2 = cv2.imread('../images/box_in_scene.png', 0)  # trainImage
        #【1】初始ORB特征检测器
        orb = cv2.ORB_create()
        #【2】用ORB找到两幅图像的关键点和描述符
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        #【3】创建BFMatcher对象
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        #【4】匹配两幅图像的描述符
        matches = bf.match(des1, des2)
        #【5】根据描述符之间的距离来排序
        matches = sorted(matches, key=lambda x: x.distance)
        #【6】画出前10匹配的特征点
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)  # 前10个匹配
        plt.imshow(img3), plt.xticks([]), plt.yticks([]),plt.show()

    def bf_sift(self):
        img1 = cv2.imread('../images/box.png', 0)# queryImage
        img2 = cv2.imread('../images/box_in_scene.png', 0)  # trainImage
        #【1】初始化SIFT对象
        sift = cv2.xfeatures2d.SIFT_create()
        #【2】用SIFT找到关键点和描述符
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        #【3】创建BFMatcher对象，使用默认参数
        bf = cv2.BFMatcher()
        #【4】用knn方法获得k 对最佳匹配
        matches = bf.knnMatch(des1, des2, k=2)
        #【5】比值测试
        # 首先获取与 A距离最近的点 B （最近）和 C （次近），
        # 只有当 B/C 小于阀值时（0.75）才被认为是匹配，
        # 因为假设匹配是一一对应的，真正的匹配的理想距离为0
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])
        #【6】画出匹配点：参数是match list
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
        plt.imshow(img3), plt.xticks([]), plt.yticks([]),plt.show()

    def flann(self):
        img1 = cv2.imread('../images/box.png', 0)  # queryImage
        img2 = cv2.imread('../images/box_in_scene.png', 0)  # trainImage
        #【1】特征检测、特征描述
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        #【2】准备FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary
        #【3】FLANN匹配
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        # Need to draw only good matches, so create a mask
        matchesMask = [[0, 0] for i in range(len(matches))]
        # 比值测试
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                matchesMask[i] = [1, 0]
        #【4】画出匹配点
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matchesMask,
                           flags=0)

        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
        plt.imshow(img3), plt.xticks([]), plt.yticks([]),plt.show()

    def homo(self):
        #【1】先在图像中来找到SIFT 特征点，然后再使用比值测试找到最佳匹配。
        img1 = cv2.imread('../images/box.png', 0)  # queryImage
        img2 = cv2.imread('../images/box_in_scene.png', 0)  # trainImage
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
        #【2】只有存在10 个以上匹配时才去查找目标，否则显示警告消息：“现在匹配不足！”
        MIN_MATCH_COUNT = 10
        if len(good) > MIN_MATCH_COUNT:
            #【3】提取两幅图像中匹配点的坐标
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            #【4】传入到函数中计算透视变换矩阵
            # 第三个参数Method used to computed a homography matrix. The following methods are possible:
            # 0 - a regular method using all the points
            # CV_RANSAC - RANSAC-based robust method
            # CV_LMEDS - Least-Median robust method
            # 第四个参数取值范围在1 到10，拒绝一个点对的阈值。原图像的点经过变换后点与目标图像上对应点的误差
            # 超过误差就认为是outlier
            # 返回值中M 为变换矩阵。
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
            h, w = img1.shape# 获得原图像的高和宽
            #【5】使用得到的变换矩阵对原图像的四个角进行变换，获得在目标图像上对应的坐标。
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            #【6】在train image中画出变换后的白色对象框
            img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        else:
            print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
            matchesMask = None
        #【7】绘制inliers（如果能成功的找到目标图像的话）或者匹配的关键点（如果失败）。
        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matchesMask,  # draw only inliers
                           flags=2)
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
        plt.imshow(img3), plt.xticks([]), plt.yticks([]),plt.show()


if __name__ == '__main__':
    c = FeatureMatch('../images/board.jpg')
    c.homo()
