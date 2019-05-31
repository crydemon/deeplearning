from numpy import *


def loadDataSet(fileName):  # general function to parse tab -delimited floats
    dataMat = []  # assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)  # map all elements to float()
        dataMat.append(fltLine)
    return dataMat


def distEclud(vecA, vecB):
    return sqrt(sum(pow(vecA - vecB, 2)))


# 为给定数据集构建一个包含 k 个随机质心的集合。随机质心必须要在整个数据集的边界之内，这可以通过找到数据集每一维的最小和最大值来完成。然后生成 0~1.0 之间的随机数并通过取值范围和最小值，以便确保随机点在数据的边界之内。

def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))
    for j in range(n):  # create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = mat(minJ + rangeJ * random.rand(k, 1))
    return centroids


def kMeans(dataSet, k, disMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = disMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:  # 如果距离比 minDist（最小距离）还小，更新 minDist（最小距离）和最小质心的 index（索引）
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:  # 簇分配结果改变
                clusterChanged = True  # 簇改变
                clusterAssment[i, :] = minIndex, minDist ** 2  # 更新簇分配结果为最小质心的 index（索引），minDist（最小距离）的平方
        print(centroids)
        for cent in range(k):  # 更新质心
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]  # 获取该簇中的所有点
            centroids[cent, :] = mean(ptsInClust, axis=0)  # 将质心修改为簇中所有点的平均值，mean 就是求平均值的
    return centroids, clusterAssment


def testBasicFunc():
    # 加载测试数据集
    dataMat = mat(loadDataSet('testSet.txt'))

    # 测试 randCent() 函数是否正常运行。
    # 首先，先看一下矩阵中的最大值与最小值
    print('min(dataMat[:, 0])=', min(dataMat[:, 0]))
    print('min(dataMat[:, 1])=', min(dataMat[:, 1]))
    print('max(dataMat[:, 1])=', max(dataMat[:, 1]))
    print('max(dataMat[:, 0])=', max(dataMat[:, 0]))

    # 然后看看 randCent() 函数能否生成 min 到 max 之间的值
    print('randCent(dataMat, 2)=', randCent(dataMat, 2))

    # 最后测试一下距离计算方法
    print(' distEclud(dataMat[0], dataMat[1])=', distEclud(dataMat[0], dataMat[1]))