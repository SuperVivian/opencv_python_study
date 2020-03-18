#coding-utf8
import cv2
import matplotlib.pyplot as plt
import numpy as np

class KMeans:
    def __init__(self, infile):
        self.infile = infile
        self.img = cv2.imread(self.infile)

    def one_feature(self):
        #随机产生数据并绘制
        x = np.random.randint(25, 100, 25)
        y = np.random.randint(175, 255, 25)
        z = np.hstack((x, y))
        z = z.reshape((50, 1))#长度为50，取值范围为0 到255
        z = np.float32(z)
        plt.hist(z, 256, [0, 256]), plt.show()
        #使用KMeans
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0) #终止条件：迭代10次货精确度epsilon=1.0
        flags = cv2.KMEANS_RANDOM_CENTERS
        compactness, labels, centers = cv2.kmeans(z, 2, None, criteria, 10, flags)
        A = z[labels == 0]
        B = z[labels == 1]
        plt.hist(A, 256, [0, 256], color='r')
        plt.hist(B, 256, [0, 256], color='b')
        plt.hist(centers, 32, [0, 256], color='y')#重心
        plt.show()

    def multi_feature(self):
        X = np.random.randint(25, 50, (25, 2))
        Y = np.random.randint(60, 85, (25, 2))
        Z = np.vstack((X, Y))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = cv2.kmeans(Z, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        A = Z[label.ravel() == 0]
        B = Z[label.ravel() == 1]
        plt.scatter(A[:, 0], A[:, 1])
        plt.scatter(B[:, 0], B[:, 1], c='r')
        plt.scatter(center[:, 0], center[:, 1], s=80, c='y', marker='s')
        plt.xlabel('Height'), plt.ylabel('Weight')
        plt.show()

    def color(self):
        img = self.img
        #转换数据格式
        Z = img.reshape((-1, 3))
        Z = np.float32(Z)
        #用kmeans分成14类
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 14
        ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))#转回原来的图像格式
        #画出结果
        titles = ['raw', 'kmeans']
        imgs = [self.img,res2]
        for i in range(2):
            plt.subplot(1, 2, i + 1), plt.imshow(cv2.cvtColor(imgs[i],cv2.COLOR_BGR2RGB))
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.show()

if __name__ == '__main__':
    d= KMeans('../images/home.jpg')
    d.color()

