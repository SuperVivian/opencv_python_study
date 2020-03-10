import cv2
import matplotlib.pyplot as plt
import numpy as np

class DFT:
    def __init__(self, infile):
        self.infile = infile
        self.img = cv2.imread(self.infile)

    def numpy_fft(self):
        #【1】转灰度图
        img = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)

        #【2】fft2：对信号进行频率转换，输出结果是一个复杂的数组。
        #第二个参数是可选的, 决定输出数组的大小。输出数组的大小和输入图
        #像大小一样。如果输出结果比输入图像大，输入图像就需要在进行FFT 前补
        #0。如果输出结果比输入图像小的话，输入图像就会被切割。
        f = np.fft.fft2(img)

        #【3】fftshift
        fshift = np.fft.fftshift(f)#频率为0 的部分（直流分量）在输出图像的左上角
        #让它（直流分量）在输出图像的中心，我们还需要将结果沿两个方向平移N/2

        #【4】构建振幅图
        magnitude_spectrum = 20 * np.log(np.abs(fshift))#构建振幅图的公式
        #画图
        plt.subplot(121), plt.imshow(img, cmap='gray')
        plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
        plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
        plt.show()

    def numpy_fft_abs(self):
        img = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        #可以在频域对图像进行一些操作:
        #【1】使用一个60x60 的矩形窗口对图像进行掩模操作从而去除低频分量
        rows, cols = img.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        fshift[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0
        #【2】
        f_ishift = np.fft.ifftshift(fshift)#进行逆平移操作，使直流分量回到左上角
        img_back = np.fft.ifft2(f_ishift)#进行FFT 逆变换
        # 取绝对值
        img_back = np.abs(img_back)
        #结果显示高通滤波其实是一种边界检测操作
        plt.subplot(131), plt.imshow(img, cmap='gray')
        plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(132), plt.imshow(img_back, cmap='gray')
        plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
        plt.subplot(133), plt.imshow(img_back)
        plt.title('Result in JET'), plt.xticks([]), plt.yticks([])
        plt.show()

    def dft(self):
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        #构建振幅图
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
        #画图
        plt.subplot(121), plt.imshow(img, cmap='gray')
        plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
        plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
        plt.show()

    def reverse_dft(self):
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        rows, cols = img.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        # 首先创建一个mask, 中心为1，其余为0
        mask = np.zeros((rows, cols, 2), np.uint8)
        mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1

        # 应用mask，inverse DFT
        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        #画图
        plt.subplot(121), plt.imshow(img, cmap='gray')
        plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(img_back, cmap='gray')
        plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
        plt.show()

    def optimize_dft(self):
        img = cv2.imread('messi5.jpg', 0)
        # 图像原始大小
        rows, cols = img.shape
        print(rows, cols)  # 342 548
        # 优化后的图像大小
        nrows = cv2.getOptimalDFTSize(rows)
        ncols = cv2.getOptimalDFTSize(cols)
        print(nrows, ncols)  # 360 57
        # 补零
        nimg = np.zeros((nrows, ncols))
        nimg[:rows, :cols] = img
        # Numpy：
        fft1 = np.fft.fft2(img)
        fft2 = np.fft.fft2(nimg, [nrows, ncols])
        # OpenCV：
        dft1 = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft2 = cv2.dft(np.float32(nimg), flags=cv2.DFT_COMPLEX_OUTPUT)

    def test_algorithm(self):
        mean_filter = np.ones((3, 3))# 均值过滤
        x = cv2.getGaussianKernel(5, 10)# 创建一个高斯核
        gaussian = x * x.T#矩阵转置
        # 不同算子
        # scharr：x方向
        scharr = np.array([[-3, 0, 3],
                           [-10, 0, 10],
                           [-3, 0, 3]])
        # sobel：x方向
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])
        # sobel：y方向
        sobel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]])
        # laplacian：
        laplacian = np.array([[0, 1, 0],
                              [1, -4, 1],
                              [0, 1, 0]])
        #不同的过滤器
        filters = [mean_filter, gaussian, laplacian, sobel_x, sobel_y, scharr]
        filter_name = ['mean_filter', 'gaussian', 'laplacian', 'sobel_x', 'sobel_y', 'scharr_x']
        #求不同核作用下的图像的振幅图
        fft_filters = [np.fft.fft2(x) for x in filters]
        fft_shift = [np.fft.fftshift(y) for y in fft_filters]
        mag_spectrum = [np.log(np.abs(z) + 1) for z in fft_shift]
        #画图
        for i in range(6):
            plt.subplot(2, 3, i + 1), plt.imshow(mag_spectrum[i], cmap='gray')
            plt.title(filter_name[i]), plt.xticks([]), plt.yticks([])
        plt.show()


if __name__ == '__main__':
    d = DFT('../images/messi.jpg')
    d.test_algorithm()
