#coding-utf8
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob

def draw(img, corners, imgpts):#参数：输入图像、棋盘上的角点、要绘制的3D 坐标轴上的点
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img

def draw_cube(img,imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)
    #用绿色画底面
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)
    #用蓝色画柱子
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)
    #用红色画顶面
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)
    return img

class Camera:
    def __init__(self, infile):
        self.infile = infile
        self.img = cv2.imread(self.infile)
        self.objpoints = []
        self.imgpoints = []

    def find_corners(self):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)#终止条件
        # 对象点：like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6 * 7, 3), np.float32)
        objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
        images = glob.glob('../images/boards/left*.jpg')
        for fname in images:#遍历所有图像
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#变灰度
            # 找到棋盘角点
            ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)
            #如果找到，添加对象点、图像点
            if ret == True:
                self.objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)#亚像素精确化角点
                self.imgpoints.append(corners)
                #画出角点
                cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
                cv2.imshow('img', img)
                cv2.waitKey(1)
        cv2.destroyAllWindows()

    def calib(self):
        # 【1】读入一张图片进行准备进行校正
        img = cv2.imread('../images/boards/left12.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 变灰度

        # 【2】进行摄像机标定，得到摄像机矩阵、畸变系数等
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1], None, None)

        # 【3】优化摄像机矩阵
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))#返回自由缩放系数alpha和ROI图像
        #alpha = 0，返回的非畸变图像会带有最少量的不想要的像素
        #alpha = 1，所有的像素都会被返回，还有一些黑图像
        #ROI 图像，我们可以用来对结果进行裁剪

        # 【4】校正图像：两种方法结果相同
        # 方法一：undistort函数
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]# 结合上边得到的ROI 对结果进行裁剪
        # 方法二：remapping：
        mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
        dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]

        # 【5】画出结果
        titles = ['raw', 'dst']
        imgs = [img, dst]
        for i in range(2):
            plt.subplot(1, 2, i + 1), plt.imshow(cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB))
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.show()

        #【6】计算误差
        mean_error = 0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(self.objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        print("total error: ", mean_error / len(self.objpoints))

        return mtx, dist, rvecs, tvecs

    def gestrue(self,mtx, dist):
        #【1】加载前面结果中摄像机矩阵和畸变系数(本函数使用参数传递)
        #【2】设置终止条件、对象点(棋盘上的3D角点)和坐标轴点
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((6 * 7, 3), np.float32)#初始化对象点为0
        objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)#mgrid函数返回多维结构

        axis1 = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
        #X 轴从（0,0,0）绘制到（3,0,0），Y 轴从（0,0,0）绘制到（0,3,0）
        #Z 轴从（0,0,0）绘制到（0,0,-3）。负值表示它是朝着（垂直于）摄像机方向。

        #3D 空间中的一个立方体的8 个角点
        axis2 = np.float32([[0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0],
                           [0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3]])

        #【3】加载图像，检测角点，计算变换矩阵并投影
        img = cv2.imread('../images/boards/left12.jpg')
        copy1=img.copy()
        copy2=img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)#检测角点
        if ret == True:#检测到角点
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)#亚像素精确化
            ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)#计算旋转和平移变换矩阵
            #【4】利用变换矩阵把坐标轴点投影到平面，并调用绘图函数
            #3个坐标轴
            imgpts1, jac = cv2.projectPoints(axis1, rvecs, tvecs, mtx, dist)
            draw(copy1, corners2, imgpts1)
            #立方体
            imgpts2, jac = cv2.projectPoints(axis2, rvecs, tvecs, mtx, dist)
            draw_cube(copy2,imgpts2)
            #【5】展示图像
            titles = ['raw', 'dst_axis','dst_cube']
            imgs = [img, copy1,copy2]
            for i in range(3):
                plt.subplot(1, 3, i + 1), plt.imshow(cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB))
                plt.title(titles[i])
                plt.xticks([]), plt.yticks([])
            plt.show()


if __name__ == '__main__':
    d= Camera('../images/home.jpg')
    d.find_corners()
    mtx, dist, rvecs, tvecs=d.calib()
    d.gestrue(mtx, dist)




