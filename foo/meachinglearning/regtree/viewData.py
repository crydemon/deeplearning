import matplotlib.pyplot as plt
import numpy as np


def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        cur_line = line.strip().split('\t')
        flt_line = list(map(float, cur_line))
        dataMat.append(flt_line)
    return dataMat


def plotDataSet(fileName):
    dataMat = loadDataSet(fileName)
    n = len(dataMat)
    xcord = []
    ycord = []
    for i in range(n):
        xcord.append(dataMat[i][0])
        ycord.append(dataMat[i][1])
    fig = plt.figure()
    ax = fig.add_subplot(111)  # 添加subplot
    ax.scatter(xcord, ycord)  # 绘制样本点
    plt.title('DataSet')  # 绘制title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1


def regLeaf(dataSet):
    return np.mean(dataSet[:, -1])


def regErr(dataSet):
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]


def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    import types
    tolS = ops[0]
    tolN = ops[1]
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m, n = np.shape(dataSet)
    S = errType(dataSet)
    bestS = float('inf')
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n - 1):
        for splitVal in set(dataSet[:, featIndex].T.A.tolist()[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if min(np.shape(mat0)[0], np.shape(mat1)[0]) < tolN:
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if S - bestS < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if min(np.shape(mat0)[0], np.shape(mat1)[0]) < tolN:
        return None, leafType(dataSet)
    return None, bestValue


if __name__ == '__main__':
    myDat = loadDataSet('ex00.txt')
    myMat = np.mat(myDat)
    feat, val = chooseBestSplit(myMat, regLeaf, regErr, (1, 4))
    print(feat)
    print(val)
