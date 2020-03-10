#coding-utf8
import cv2
import matplotlib.pyplot as plt
import numpy as np

class Denoise:
    def __init__(self, infile):
        self.infile = infile
        self.img = cv2.imread(self.infile)

    def denoise_color(self):
        img = cv2.cvtColor(self.img, code=cv2.COLOR_BGR2RGB)
        dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        plt.subplot(121), plt.imshow(img)
        plt.subplot(122), plt.imshow(dst)
        plt.show()


    def denoise_video(self):
        cap = cv2.VideoCapture('../images/vtest.avi')
        img = [cap.read()[1] for i in range(5)]# 创建一个前5帧的list
        gray = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in img]# 全部转为灰度图
        gray = [np.float64(i) for i in gray]# 全部转为float64类型
        noise = np.random.randn(*gray[1].shape) * 10#创建一个方差为25的噪声
        noisy = [i + noise for i in gray]#向所有图像添加噪声
        noisy = [np.uint8(np.clip(i, 0, 255)) for i in noisy]#转为uint8类型
        # 根据这5帧图像，对第三帧去噪
        dst = cv2.fastNlMeansDenoisingMulti(noisy, 2, 5, None, 4, 7, 35)
        titles = ['3rd frame', '3rd frame+noise','3rd denoise']
        imgs = [gray[2], noisy[2],dst]
        for i in range(3):
            plt.subplot(1, 3, i + 1), plt.imshow(imgs[i], 'gray')
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.show()

    def mor(self):
        cap = cv2.VideoCapture("../images/vtest.avi")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
        counter = 0
        while True:
            ret, frame = cap.read()
            fgmask = fgbg.apply(frame)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            cv2.imshow('frame', fgmask)  # 前 120 帧
            counter += 1
            print(counter)
            k = cv2.waitKey(1)  # & 0xff
            if k == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    d= Denoise('../images/die.png')
    d.denoise_video()
