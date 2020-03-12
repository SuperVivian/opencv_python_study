#coding-utf8
import cv2
import matplotlib.pyplot as plt
import numpy as np

class KNN:
    def __init__(self, infile):
        self.infile = infile
        self.img = cv2.imread(self.infile)

    def opencv_knn(self):
        # 包含(x,y)值的25个点作为训练数据
        trainData = np.random.randint(0, 100, (25, 2)).astype(np.float32)#low,hign,size,dtype
        # 把25个点用数字0和1分别标记红色或蓝色
        responses = np.random.randint(0, 2, (25, 1)).astype(np.float32)#low,hign,size,dtype
        # 找出红色点并画出
        red = trainData[responses.ravel() == 0]#中括号里的表达式返回indexes
        plt.scatter(red[:, 0], red[:, 1], 80, 'r', '^')#x,y,s,c,marker
        # 找出蓝色点并画出
        blue = trainData[responses.ravel() == 1]
        plt.scatter(blue[:, 0], blue[:, 1], 80, 'b', 's')
        newcomer = np.random.randint(0, 100, (1, 2)).astype(np.float32)#一个新的点
        plt.scatter(newcomer[:, 0], newcomer[:, 1], 80, 'g', 'o')#画出这个点
        #【1】创建knn对象
        knn = cv2.ml.KNearest_create()
        #【2】训练数据
        knn.train(trainData, cv2.ml.ROW_SAMPLE,responses)
        #【3】用新的点进行测试
        ret, results, neighbours, dist = knn.findNearest(newcomer, 3)
        print("result: ", results, "\n")
        print("neighbours: ", neighbours, "\n")
        print("distance: ", dist)
        plt.show()

    def ocr(self):
        #【1】读图，并转为灰度图
        img =self.img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #【2】切割大图为小图：把图像分成20x20的5000个小块（row：1000、col：2000
        cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]
        #【3】转为numpy数组，它的size为(50,100,20,20)
        x = np.array(cells)
        #【4】准备train_data 和 test_data.
        train = x[:, :50].reshape(-1, 400).astype(np.float32)  # Size = (2500,400)
        test = x[:, 50:100].reshape(-1, 400).astype(np.float32)  # Size = (2500,400)
        #【5】为train 和 test data准备标签
        k = np.arange(10)#0-9
        train_labels = np.repeat(k, 250)[:, np.newaxis]
        test_labels = train_labels.copy()
        #【6】初始化kNN, 训练数据, 测试数据for k=1，得到预测结果
        knn = cv2.ml.KNearest_create()
        knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
        ret, result, neighbours, dist = knn.findNearest(test, k=5)
        #【7】计算分类器的准确率
        # 比较预测标签和test_data的实际标签
        matches = (result == test_labels)
        correct = np.count_nonzero(matches)
        accuracy = correct * 100.0 / result.size
        print('准确率', accuracy)  # 准确率91.76%
        # 保存数据
        np.savez('knn_data.npz', train=train, train_labels=train_labels, test=test, test_labels=test_labels)
        # 加载数据
        with np.load('knn_data.npz') as data:
            print(data.files)#['train', 'train_labels', 'test', 'test_labels']
            train = data['train']
            train_labels = data['train_labels']
            test = data['test']
            test_labels = data['test_labels']

    def en_ocr(self):
        #【1】加载数据，converters把字母转换为数字
        data = np.loadtxt('../images/letter-recognition.data', dtype='float32', delimiter=',',
                          converters={0: lambda ch: ord(ch) - ord('A')})  # 20000个
        #【2】把数据分成2部分，train和test各10000
        train, test = np.vsplit(data, 2)
        #【3】把数据集分成特征值和标签
        responses, trainData = np.hsplit(train, [1])
        labels, testData = np.hsplit(test, [1])
        #【4】初始化KNN，分类，计算准确率
        knn = cv2.ml.KNearest_create()
        knn.train(trainData, cv2.ml.ROW_SAMPLE, responses)
        ret, result, neighbours, dist = knn.findNearest(testData, k=5)
        correct = np.count_nonzero(result == labels)
        accuracy = correct * 100.0 / 10000
        print('准确率', accuracy)  # 93.06
        #存储数据
        np.savez('knn_data_alphabet.npz', train_alphabet=train, train_labels_alphabet=responses, test_alphabet=testData,
                 test_labels_alphabet=labels)

if __name__ == '__main__':
    d= KNN('../images/digits.png')
    d.en_ocr()

