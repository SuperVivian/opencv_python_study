import cv2
import matplotlib.pyplot as plt


class GUI:
    def __init__(self, infile):
        self.infile = infile

    def read_show_save_img(self, outfile):
        """
        读取、显示、存储图像
        """
        # 【1】读入图像
        img = cv2.imread(self.infile, 0)  # 0是以灰度形式读入。1是彩色，-1是包含alpha通道
        # 【2】显示图像+窗口操作
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        # 第二个参数，默认是窗口不可改。normal代表窗口可以调整大小
        cv2.imshow('image', img)  # 窗口会自动调整为图像大小。第一个参数是窗口的名字
        k = cv2.waitKey(0)
        # 等待特定的几毫秒，看是否有键盘输入。有输入，返回键ascll码，无输入，返回-1.
        # 若参数为0，则无限期等待键盘输入。
        if k == 27:  # esc键
            cv2.destroyWindow('image')  # 清除某个窗口
            cv2.destroyAllWindows()  # 清除所有窗口
        elif k == ord('s'):
            # 【3】保存图像
            cv2.imwrite(outfile, img)
            cv2.destroyAllWindows()  # 清除所有窗口

    def show_gray_img_in_plt(self,img):
        img = cv2.imread(self.infile, 0)  # 【1】opencv读入图片
        plt.imshow(img, cmap='gray', interpolation='bicubic')  # 【2】plt读入图片
        plt.show()  # 【3】plt显示图片

    def show_color_img_in_plt(self,img):
        # opencv读取图片。若彩色，是BGR模式。
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换模式
        plt.imshow(img2)  # plt是RGB模式。
        plt.xticks([]), plt.yticks([])  # 隐藏tick
        plt.title('RGB')
        plt.show()


if __name__ == '__main__':
    gui = GUI('images/test1.jpg')
    gui.show_color_img_in_plt()
