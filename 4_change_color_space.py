# coding:utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt


class ChangeColorSpace:
    def __init__(self, infile):
        self.infile = infile
        self.img = cv2.imread(self.infile)

    def change_color_space(self):
        gray=cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        print(gray.shape)
        print(hsv.shape)
        # cv2.imshow('gray',gray)
        # cv2.imshow('hsv',hsv)
        # cv2.waitKey(0)




if __name__ == '__main__':
    change = ChangeColorSpace('images/test1.jpg')
    change.change_color_space()
