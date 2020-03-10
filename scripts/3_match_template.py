# coding:utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt


class Template:
    def __init__(self, infile):
        self.infile = infile
        self.img = cv2.imread(self.infile)

    def match_one(self):
        raw_gray=cv2.imread(self.infile,0)
        gray_2=raw_gray.copy()
        template=cv2.imread('../images/messi_face.jpg',0)
        w,h=template.shape[::-1]
        print(template.shape)#(52, 40)
        print(w,h)#40 52

        methods=['cv2.TM_CCOEFF','cv2.TM_CCOEFF_NORMED','cv2.TM_CCORR',
                 'cv2.TM_CCORR_NORMED','cv2.TM_SQDIFF','cv2.TM_SQDIFF_NORMED']
        for meth in methods:
            img=gray_2.copy()
            method=eval(meth)#eval用来计算存储在字符串中的有效python表达式
            #exec语句执行存储在字符串或文件中的Python语句
            res=cv2.matchTemplate(img,template,method)
            min_val,max_val,min_loc,max_loc=cv2.minMaxLoc(res)
            #使用不同的比较方法，对结果的解释不同
            if method in [cv2.TM_SQDIFF,cv2.TM_SQDIFF_NORMED]:#这两种选min
                top_left=min_loc
            else:
                top_left=max_loc
            bottom_right=(top_left[0]+w,top_left[1]+h)
            cv2.rectangle(img,top_left,bottom_right,255,2)#画一个矩形
            titles = ['Matching Result', 'Detected Point']
            imgs = [res,img]
            for i in range(2):
                plt.subplot(1, 2, i + 1), plt.imshow(imgs[i], 'gray')
                plt.title(titles[i])
                plt.xticks([]), plt.yticks([])
            plt.suptitle(meth)
            plt.show()



    def match_more(self):
        img_rgb=cv2.imread('../images/mario.jpg')
        img_gray=cv2.cvtColor(img_rgb,cv2.COLOR_BGRA2GRAY)
        template=cv2.imread('../images/mario_coin.jpg',0)
        w,h=template.shape[::-1]
        res=cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
        threshold=0.7
        loc=np.where(res>=threshold)
        rgb_copy=img_rgb.copy()
        template_rgb=cv2.imread('../images/mario_coin.jpg')
        for pt in zip(*loc[::-1]):#获取每个区域左上角的点
            cv2.rectangle(rgb_copy,pt,(pt[0]+w,pt[1]+h),(0,0,255),2)
        titles = ['raw', 'template','match loc']
        imgs = [img_rgb, template_rgb,rgb_copy]
        for i in range(3):
            plt.subplot(1, 3, i + 1), plt.imshow(cv2.cvtColor(imgs[i],cv2.COLOR_BGR2RGB))
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.show()


if __name__ == '__main__':
    t = Template('../images/messi.jpg')
    t.match_more()
