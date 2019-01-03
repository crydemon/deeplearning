import numpy as np
import operator
from os import listdir


def creatDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']  # 分类变量A与B
    return group, labels


group, labels = creatDataSet()  # 变量赋值


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # 通过重复reps次A来构造出一个新数组。
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    # 每一个样本集与输入向量距离的平方值
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    # print(distances)
    sortedDistIndicies = np.argsort(distances)  # 将distance中的元素从小到大排列，提取其对应的index(索引)，然后输出到sortedDistIndicies中
    classCount = {}
    # print(distances)
    # print(sortedDistIndicies)
    # 查看前k个距离最小的分类，选择最大的类作为此项的分类
    for i in range(k):
        # print(i)
        # print(sortedDistIndicies[i])
        voteIlabel = labels[sortedDistIndicies[i]]
        # print(voteIlabel)
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        # print(classCount[voteIlabel])
        # print("----------")
    # print(classCount)
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # operator.itemgetter(1)      //定义函数，获取对象的第1个域的值
    # operator.itemgetter(1,0)  //定义函数b，获取对象的第1个域和第0个的值
    # print(sortedClassCount)
    return sortedClassCount[0][0]


print(classify0([0, 0], group, labels, 3))


def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())  # get the number of lines in the file
    returnMat = np.zeros((numberOfLines, 3))  # prepare matrix to return
    classLabelVector = []  # prepare labels return
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))  # 通过索引值-1将列表的最后一列元素存储到向量classLabelVector中
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))  # 将变量内容变成与输入矩阵同样大小的矩阵
    normDataSet = normDataSet / np.tile(ranges, (m, 1))  # element wise divide  # 特征值归一化公式
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.50  # hold out 10%
    datingDataMat, datingLabels = file2matrix('D:\paper\kNN-master\机器学习实战（中文版+英文版+源代码）\机器学习实战源代码\machinelearninginaction\Ch02\datingTestSet2.txt')  # load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 6)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
    print(errorCount)


datingClassTest()


def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])  # 向量转换
    return returnVect


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('D:/paper/machinelearninginaction/Ch02/digits/trainingDigits')  # load the training set
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))  # 创建m*1024的训练矩阵
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)  # 从文件名中切割出分类的数字并加入在向量hwLables中
        trainingMat[i, :] = img2vector('D:/paper/machinelearninginaction/Ch02/digits/trainingDigits/%s' % fileNameStr)
    testFileList = listdir('D:/paper/machinelearninginaction/Ch02/digits/testDigits')  # iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('D:/paper/machinelearninginaction/Ch02/digits/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount / float(mTest)))

# handwritingClassTest()
