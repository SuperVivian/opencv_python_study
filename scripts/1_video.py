import cv2
import matplotlib.pyplot as plt


class Video:
    def __init__(self, infile):
        self.infile = infile
        self.img = cv2.imread(self.infile)

    def capture(self):
        #【1】创建Capture对象
        cap = cv2.VideoCapture(0)#参数为设备索引号或视频文件。0为笔记本内置摄像头
        if cap.isOpened():#检验摄像头是否打开
            print("镜头是打开的")
        else:
            cap.open(0)#打开摄像头
        print(cap.get(0))#参数是0-18的整数，每个数代表视频的一个属性
        w=cap.get(3)
        h=cap.get(4)
        ret=cap.set(3,320)#用set函数来设置某些属性值
        ret=cap.set(4,240)
        while (True):
            #【2】一帧一帧捕获
            ret, frame = cap.read()#帧读取正确就返回True。最后检查返回值看是否到视频结尾
            # 【3】把该帧图像转为灰度图像
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 【4】显示结果
            cv2.imshow('frame', gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # 【5】释放Capture对象
        cap.release()
        cv2.destroyAllWindows()

    def save_video(self):
        cap = cv2.VideoCapture(0)
        # 定义codec，创建VideoWriter对象
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #FourCC 就是一个4 字节码，用来确定视频的编码格式。
        out = cv2.VideoWriter('../res_images/output.avi', fourcc, 20.0, (640, 480))
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                frame = cv2.flip(frame, 0)
                # 写入flip的帧
                out.write(frame)
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        # Release everything if job is finished
        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    v = Video('../images/test1.jpg')
    v.save_video()
