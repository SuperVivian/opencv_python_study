# coding-utf8
import cv2
import matplotlib.pyplot as plt
import numpy as np

SZ = 20
bin_n = 16  # Number of bins
affine_flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR


# 使用图像的二阶矩对其进行抗扭斜（deskew）处理
def deskew(img):
    m = cv2.moments(img)  # 求矩
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=affine_flags)
    return img


# 使用方向梯度直方图Histogram of Oriented Gradients  HOG 作为特征向量
# 计算图像 X 方向和 Y 方向的 Sobel 导数。然后计算得到每个像素的梯度的方向和大小。把这个梯度转换成16 位的整数。
# 将图像分为4 个小的方块，对每一个小方块计算它们的朝向直方图（16 个bin），使用梯度的大小做权重。
# 这样每一个小方块都会得到一个含有16 个成员的向量。
# 4 个小方块的4 个向量就组成了这个图像的特征向量（包含64 个成员）
def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n * ang / (2 * np.pi))# quantizing binvalues in (0...16)
    bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]
    mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)# hist is a 64 bit vector
    return hist


class SVM:
    def __init__(self, infile):
        self.infile = infile
        self.img = cv2.imread(self.infile, 0)

    def opencv_svm(self):
        img = self.img
        # 【1】切割图像
        cells = [np.hsplit(row, 100) for row in np.vsplit(img, 50)]
        # 【2】确定trainData和testData
        train_cells = [i[:50] for i in cells]
        test_cells = [i[50:] for i in cells]
        # 【3】对所有训练图像做抗扭斜处理
        deskewed = [map(deskew, row) for row in train_cells]
        # 【4】计算所有训练图像的hog
        hogdata = [map(hog, row) for row in deskewed]
        # 【5】训练数据的特征值、标签
        trainData = np.float32(hogdata).reshape(-1, 64)
        responses = np.float32(np.repeat(np.arange(10), 250)[:, np.newaxis])
        # 【6】svm训练
        svm = cv2.ml.SVM_create()  # 创建对象
        svm_params = dict(kernel_type=cv2.ml.SVM_LINEAR,
                          svm_type=cv2.ml.SVM_C_SVC,
                          C=2.67, gamma=5.383)
        svm.train(trainData, cv2.ml.ROW_SAMPLE, responses, params=svm_params)
        svm.save('svm_data.dat')
        # 【7】用同样的方法处理测试集
        deskewed = [map(deskew, row) for row in test_cells]
        hogdata = [map(hog, row) for row in deskewed]
        testData = np.float32(hogdata).reshape(-1, bin_n * 4)
        # 【8】预测
        result = svm.predict(testData)
        mask = result == responses
        correct = np.count_nonzero(mask)
        print(correct * 100.0 / result.size)


if __name__ == '__main__':
    d = SVM('../images/digits.png')
    d.opencv_svm()
