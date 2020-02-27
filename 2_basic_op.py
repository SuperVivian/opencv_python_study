import cv2
import numpy as np
import matplotlib.pyplot as plt


class Basic:
    def __init__(self, infile):
        self.infile = infile
        self.img = cv2.imread(self.infile)

    def get_pixel(self):
        # 根据行列坐标获取像素值，可以直接修改该值
        px1 = self.img[100, 100, 0]
        px2 = self.img[100, 100]  # 返回bgr值。如果是灰度图像返回灰度值。
        px3 = self.img[100:103, 100:103]  # 9个点
        self.img[100, 100] = [255, 255, 255]  # 修改像素值
        print(px1)  # 229
        print(px2)  # [229 224 225]
        print(px3)
        print(self.img[100, 100])

    def get_pixel_by_numpy(self):
        print(self.img.item(10, 10, 2))  # item()获取一个像素值
        self.img.itemset((10, 10, 2), 100)  # itemset()修改像素值
        print(self.img.item(10, 10, 2))

    def get_img_properties(self):
        print(self.img.shape)  # (640, 640, 3)
        print(self.img.size)  # 1228800
        print(self.img.dtype)  # uint8

    def copy_roi(self):
        cv2.imshow('image', self.img)
        copy = self.img
        x = copy[280:340, 330:390]
        copy[273:333, 100:160] = x
        cv2.imshow('copy', copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def split_merge_channel(self):
        b, g, r = cv2.split(self.img)  # 比较耗时，尽量用索引
        img_s = cv2.merge((b, g, r))  # 参数是一个vector

        b = self.img[:, :, 0]  # 拿到第一个通道的
        copy = self.img
        copy[:, :, 2] = 0  # 红色通道置为0
        cv2.imshow('image', copy)
        cv2.waitKey(0)

    def make_border(self):
        # 四个值的顺序是：上下左右
        top = bottom = left = right = 100
        replicate = cv2.copyMakeBorder(self.img, top, bottom, left, right, cv2.BORDER_REPLICATE)
        reflect = cv2.copyMakeBorder(self.img, top, bottom, left, right, cv2.BORDER_REFLECT)
        reflect101 = cv2.copyMakeBorder(self.img, top, bottom, left, right, cv2.BORDER_REFLECT_101)
        wrap = cv2.copyMakeBorder(self.img, top, bottom, left, right, cv2.BORDER_WRAP)
        constant = cv2.copyMakeBorder(self.img, top, bottom, left, right, cv2.BORDER_CONSTANT)

        plt.subplot(231), plt.imshow(self.img, 'gray'), plt.title('original'),plt.xticks([]), plt.yticks([])
        plt.subplot(232), plt.imshow(replicate, 'gray'), plt.title('replicate'),plt.xticks([]), plt.yticks([])
        plt.subplot(233), plt.imshow(reflect, 'gray'), plt.title('reflect'),plt.xticks([]), plt.yticks([])
        plt.subplot(234), plt.imshow(reflect101, 'gray'), plt.title('reflect101'),plt.xticks([]), plt.yticks([])
        plt.subplot(235), plt.imshow(wrap, 'gray'), plt.title('wrap'),plt.xticks([]), plt.yticks([])
        plt.subplot(236), plt.imshow(constant, 'gray'), plt.title('constant'),plt.xticks([]), plt.yticks([])

        plt.show()


if __name__ == '__main__':
    basic = Basic('images/test1.jpg')
    basic.make_border()
