import numpy as np
import pandas as pd
import operator
from os import listdir
from sklearn.svm import SVC


def handwritingClassTest():
    train = pd.read_csv('d:/train.csv')
    # pd inner join
    trainArr = train.as_matrix()
    # 测试集的Labels
    hwLabels = []
    m = len(trainArr)
    # 初始化训练的Mat矩阵,测试集
    trainingMat = np.zeros((m, len(trainArr[0]) - 1))
    # 从文件名中解析出训练集的类别
    for i in range(m):
        classNumber = trainArr[i][-1]
        # 将获得的类别添加到hwLabels中
        hwLabels.append(classNumber)
        # 将每一个文件的1x1024数据存储到trainingMat矩阵中
        trainingMat[i, :] = [hash(i) for i in trainArr[i][:-1]]
    clf = SVC(C=200, kernel='rbf')
    clf.fit(trainingMat, hwLabels)
    # 返回testDigits目录下的文件列表

    test = pd.read_csv('d:/test.csv')
    # pd inner join
    testArr = train.as_matrix()
    # 错误检测计数
    errorCount = 0.0
    # 测试数据的数量
    mTest = len(testArr)
    # 从文件中解析出测试集的类别并进行分类测试
    for i in range(mTest):
        classNumber = testArr[i][-1]
        vectorUnderTest = [hash(i) for i in testArr[i][:-1]]
        # 获得预测结果
        # classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        classifierResult = clf.predict(vectorUnderTest)
        print("分类返回结果为%d  真实结果为%d" % (classifierResult, classNumber))
        if (classifierResult != classNumber):
            errorCount += 1.0
    print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount / mTest * 100))


if __name__ == '__main__':
    handwritingClassTest()
