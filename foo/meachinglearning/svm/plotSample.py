import matplotlib.pyplot as plt
import numpy as np


def loadDataSet(fileName):
    dataMat = [];
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


def showDataSet(dataMat, labelMat):
    dataPlus = []
    dataMinus = []
    for i in range(len(labelMat)):
        dataPlus.append(dataMat[i]) if labelMat[i] > 0 else dataMinus.append(dataMat[i])

    dataPlusNp = np.array(dataPlus)
    dataMinusNp = np.array(dataMinus)
    plt.scatter(np.transpose(dataPlusNp)[0], np.transpose(dataPlusNp)[1],  marker='s')
    plt.scatter(np.transpose(dataMinusNp)[0], np.transpose(dataMinusNp)[1],  marker='o', c='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


if __name__ == '__main__':
    dataMat, labelMat = loadDataSet('testSet.txt')
    showDataSet(dataMat, labelMat)
