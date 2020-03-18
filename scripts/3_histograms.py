# coding:utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt


class Histograms:
    def __init__(self, infile):
        self.infile = infile
        self.img = cv2.imread(self.infile)

    def accu_paint(self):
        raw_gray=cv2.imread(self.infile,0)
        raw_color=cv2.imread(self.infile)

        #【1】plot统计单通道直方图，用plot绘制
        plt.hist(raw_gray.ravel(),256,[0,256])
        plt.show()
        #【2】cv统计三通道直方图，用plot绘制
        color=('b','g','r')
        for i,col in enumerate(color):
            histr=cv2.calcHist([raw_color],[i],None,[256],[0,256])
            plt.plot(histr,color=col)
            plt.xlim([0,256])
        plt.show()
        #【3】numpy方法统计直方图，用plot绘制
        #方法一：
        np_hist1, bins = np.histogram(raw_gray.ravel(), 256, [0, 256])  # cv函数比此函数快40倍
        # img.ravel()把图像转为一维数组
        #方法二：
        np_hist2 = np.bincount(raw_gray.ravel(), minlength=256)  # 比histogram快十倍

        titles=['histogram','bincount']
        hists=[np_hist1,np_hist2]
        for i in range(2):
            plt.subplot(1,2,i+1),plt.plot(hists[i])
            plt.title(titles[i])
        plt.show()

    def mask_hist(self):
        raw_gray=cv2.imread(self.infile,0)
        mask=np.zeros(raw_gray.shape[:2],np.uint8)#mask为全黑的图像
        mask[100:500,100:600]=255#mask的该区域变白
        masked_img=cv2.bitwise_and(raw_gray,raw_gray,mask=mask)
        hist_full=cv2.calcHist([raw_gray],[0],None,[256],[0,256])
        hist_mask=cv2.calcHist([raw_gray],[0],mask,[256],[0,256])
        titles = ['raw_gray', 'mask','masked_img']
        imgs = [raw_gray, mask,masked_img]
        for i in range(3):
            plt.subplot(2, 2, i + 1), plt.imshow(imgs[i],'gray')
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.subplot(2,2,4),plt.plot(hist_full),plt.plot(hist_mask)
        plt.xlim([0,256])
        plt.show()

    def np_equalize(self):
        img = cv2.imread('../images/wiki.jpg', 0)
        #【1】计算原始图像的直方图及累积分布图
        hist1, bins = np.histogram(img.flatten(), 256, [0, 256])# 直方图
        cdf1 = hist1.cumsum()
        cdf_normalized1 = cdf1 * hist1.max() / cdf1.max()# 计算累积分布图
        #【2】对原始图像进行均衡化
        # 构建Numpy 掩模数组，cdf 为原数组，当数组元素为0 时，掩盖（计算时被忽略）。
        cdf_m = np.ma.masked_equal(cdf1, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        # 对被掩盖的元素赋值，这里赋值为0
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        img2 = cdf[img]#得到均衡化后的图像
        #【3】计算均衡化后图像的直方图及累积分布图
        hist2, bins = np.histogram(img2.flatten(), 256, [0, 256])
        cdf2 = hist2.cumsum()# 计算累积分布图
        cdf_normalized2 = cdf2 * hist2.max() / cdf2.max()
        #【4】展示：用plot
        plt.subplot(121),plt.title('original hist'),plt.hist(img.flatten(), 256, [0, 256], color='r')
        plt.plot(cdf_normalized1, color='b'),plt.xlim([0, 256]),plt.legend(('cdf', 'histogram'), loc='upper left')
        plt.subplot(122),plt.title('dst hist'),plt.hist(img2.flatten(), 256, [0, 256], color='r')
        plt.plot(cdf_normalized2, color='b'),plt.xlim([0, 256]),plt.legend(('cdf', 'histogram'), loc='upper left')
        plt.show()

    def cv_equalize(self):
        img = cv2.imread('../images/wiki.jpg', 0)
        equ = cv2.equalizeHist(img)#均衡化
        res = np.hstack((img, equ))  # 拼接图像
        #展示
        plt.imshow(res, 'gray')
        plt.xticks([]), plt.yticks([])
        plt.show()

    def clahe(self):
        img = cv2.imread('../images/tsukuba_l.png', 0)
        # 普通均衡化
        equ = cv2.equalizeHist(img)
        #自适应均衡化
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))# create a CLAHE object (Arguments are optional).
        cl1 = clahe.apply(img)
        #展示
        titles = ['raw', 'equ','clahe']
        imgs = [img, equ,cl1]
        for i in range(3):
            plt.subplot(1, 3, i + 1), plt.imshow(imgs[i],'gray')
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.show()

    def two_d_hist(self):
        img = cv2.imread('../images/home.jpg')
        copy=img.copy()
        #【1】把图像转为HSV模式
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        #【2】计算二维直方图
        #cv方法：
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        # Numpy方法：
        h, s, v = cv2.split(hsv)
        hist, xbins, ybins = np.histogram2d(h.ravel(), s.ravel(), [180, 256], [[0, 180], [0, 256]])
        #【3】绘制2D直方图
        plt.subplot(121), plt.imshow(cv2.cvtColor(copy,cv2.COLOR_BGR2RGB))
        plt.title('raw'),plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(hist,interpolation = 'nearest'),plt.title('2D hist')
        plt.show()

    def numpy_back_project(self):
        # Numpy 中的算法
        #【1】roi是要查找的目标区域
        roi = cv2.imread('../images/grass.jpg')
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        #【2】target是我们用来查找的图像：在messi图像中查找草地
        target = cv2.imread('../images/messi.jpg')
        hsvt = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
        #【3】得到两幅图像的直方图
        M = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        I = cv2.calcHist([hsvt], [0, 1], None, [180, 256], [0, 180, 0, 256])
        #【4】反向投影
        R=cv2.divide(M,I)#根据R 这个”调色板“创建一副新的图像，其中的每一个像素代表这个点就是目标的概率。
        h, s, v = cv2.split(hsvt)
        B = R[h.ravel(), s.ravel()]#h 为点（x，y）处的hue 值，s 为点（x，y）处的saturation 值。
        B = np.minimum(B, 1)#B (x,y) = min [B (x,y),1]。
        B = B.reshape(hsvt.shape[:2])
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        B = cv2.filter2D(B, -1, disc)#使用一个圆盘算子做卷积，B = D x B，其中D 为卷积核
        B = np.uint8(B)#输出图像中灰度值最大的地方就是我们要查找的目标的位置
        cv2.normalize(B, B, 0, 255, cv2.NORM_MINMAX)
        #【5】对输出图像做二值化
        ret, thresh = cv2.threshold(B, 50, 255, 0)
        #展示
        plt.subplot(131), plt.imshow(cv2.cvtColor(roi,cv2.COLOR_BGR2RGB))
        plt.title('roi'),plt.xticks([]), plt.yticks([])
        plt.subplot(132), plt.imshow(cv2.cvtColor(target,cv2.COLOR_BGR2RGB))
        plt.title('target'),plt.xticks([]), plt.yticks([])
        plt.subplot(133), plt.imshow(thresh,'gray')
        plt.title('thresh'),plt.xticks([]), plt.yticks([])
        plt.show()

    def cv_back_project(self):
        #【1】roi是要查找的目标区域
        roi = cv2.imread('../images/grass.jpg')
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        #【2】target是我们用来查找的图像：在messi图像中查找草地
        target = cv2.imread('../images/messi.jpg')
        hsvt = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
        #【3】得到roi的直方图
        roihist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        #【4】归一化直方图并进行反向投影
        # 归一化之后的直方图便于显示，归一化之后就成了0 到255 之间的数了。
        cv2.normalize(roihist, roihist, 0, 255, cv2.NORM_MINMAX)
        #输入参数为：原始图像，结果图像，映射到结果图像中的最小值，最大值，归一化类型
        # cv2.NORM_MINMAX 对数组的所有值进行转化，使它们线性映射到最小值和最大值之间

        dst = cv2.calcBackProject([hsvt], [0, 1], roihist, [0, 180, 0, 256], 1)

        #【5】用圆盘算子做卷积，把分散的点连在一起
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dst = cv2.filter2D(dst, -1, disc)
        #【6】对结果图像二值化
        ret, thresh = cv2.threshold(dst, 50, 255, 0)
        #【7】合并为三通道
        thresh = cv2.merge((thresh, thresh, thresh))
        res = cv2.bitwise_and(target, thresh)# 按位操作
        res = np.hstack((target, thresh, res))#拼接三幅图像
        # 展示
        plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
        plt.xticks([]), plt.yticks([]), plt.show()

if __name__ == '__main__':
    h = Histograms('../images/test2.jpg')
    h.cv_back_project()
