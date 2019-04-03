from numpy import *


# 词表到向量的转换函数
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1,侮辱  0,正常
    return postingList, classVec


def createVocabList(dataSet):
    vocabSet = set([])  # 调用set方法,创建一个空集
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 创建两个集合的并集
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)  # 创建一个所含元素都为0的向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word:%s is not in my Vocabulary" % word)
    return returnVec


def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)  # 创建一个所含元素都为0的向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


# 朴素贝叶斯分类器训练集
def trainNB0(trainMatrix, trainCategory):  # 传入参数为文档矩阵，每篇文档类别标签所构成的向量
    numTrainDocs = len(trainMatrix)  # 文档矩阵的长度
    numWords = len(trainMatrix[0])  # 第一个文档的单词个数
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 任意文档属于侮辱性文档概率
    # p0Num = zeros(numWords);p1Num = zeros(numWords)        #初始化两个矩阵，长度为numWords，内容值为0
    p0Num = ones(numWords)
    p1Num = ones(numWords)  # 初始化两个矩阵，长度为numWords，内容值为1
    # p0Denom = 0.0;p1Denom = 0.0                         #初始化概率
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # p1Vect = p1Num/p1Denom #对每个元素做除法
    # p0Vect = p0Num/p0Denom
    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


# 朴素贝叶斯分类函数
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):

    p1 = sum(vec2Classify * p1Vec) + log(pClass1)  # 元素相乘
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


# def testingNB():
#     listOPosts, listClasses = loadDataSet()  # 产生文档矩阵和对应的标签
#     myVocabList = createVocabList(listOPosts)  # 创建并集
#     trainMat = []  # 创建一个空的列表
#     for postinDoc in listOPosts:
#         trainMat.append(setOfWords2Vec(myVocabList, postinDoc))  # 使用词向量来填充trainMat列表
#     p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))  # 训练函数
#     testEntry = ['love', 'my', 'dalmation']  # 测试文档列表
#     thisDoc = array(setOfWords2Vec(myVocabList, testEntry))  # 声明矩阵
#     print(testEntry, 'classified as:', classifyNB(thisDoc, p0V, p1V, pAb))
#
#     testEntry = ['stupid', 'garbage']
#     thisDoc = array(setOfWords2Vec(myVocabList, testEntry))  # 声明矩阵
#     print(testEntry, 'classified as:', classifyNB(thisDoc, p0V, p1V, pAb))


def textParse(bigString):  # 将字符串转换为字符列表
    import re
    # * 会匹配0个或多个规则，split会将字符串分割成单个字符【python3.5+】; 这里使用\W 或者\W+ 都可以将字符数字串分割开，产生的空字符将会在后面的列表推导式中过滤掉
    listOfTokens = re.split(r'\W+', bigString)  # 将特殊符号作为切分标志进行字符串切分，即非字母、非数字
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]  # 除了单个字母，例如大写的I，其它单词变成小写


def spamTest():
    docList = [];
    classList = [];
    fullText = []
    f_path = 'e:/machinelearninginaction/Ch04/email'
    for i in range(1, 26):  # 遍历25个txt文件
        wordList = textParse(open(f_path + '/spam/%d.txt' % i, encoding='utf-8', errors='ignore').read())  # 读取每个垃圾邮件，并字符串转换成字符串列表
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(1)  # 标记垃圾邮件，1表示垃圾文件
        wordList = textParse(open(f_path + '/ham/%d.txt' % i, encoding='utf-8', errors='ignore').read())  # 读取每个非垃圾邮件，并字符串转换成字符串列表
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(0)  # 标记非垃圾邮件，1表示垃圾文件
    vocabList = createVocabList(docList)  # 创建词汇表，不重复
    trainingSet = list(range(50))
    testSet = []  # 创建存储训练集的索引值的列表和测试集的索引值的列表
    for i in range(10):  # 从50个邮件中，随机挑选出40个作为训练集,10个做测试集
        randIndex = int(random.uniform(0, len(trainingSet)))  # 随机选取索索引值
        testSet.append(trainingSet[randIndex])  # 添加测试集的索引值
        del (trainingSet[randIndex])  # 在训练集列表中删除添加到测试集的索引值
    trainMat = []
    trainClasses = []  # 创建训练集矩阵和训练集类别标签系向量
    for docIndex in trainingSet:  # 遍历训练集
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))  # 将生成的词集模型添加到训练矩阵中
        trainClasses.append(classList[docIndex])  # 将类别添加到训练集类别标签系向量中
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))  # 训练朴素贝叶斯模型
    print(p0V)
    print(p1V)
    print(pSpam)
    errorCount = 0  # 错误分类计数
    for docIndex in testSet:  # 遍历测试集
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])  # 测试集的词集模型
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:  # 如果分类错误
            errorCount += 1  # 错误计数加1
            print("分类错误的测试集：", docList[docIndex])
    print('错误率：%.2f%%' % (float(errorCount) / len(testSet) * 100))


if __name__ == '__main__':
    spamTest()
