import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split

"""
knn
具体参数含义如下： 
n_neighbors:一个整数，指定k值 
weights:一个字符串或者可调用对象，指定投票权重类型。 
如果为’uniform’:本节点的所有邻居节点投票权重相同 
如果为’distance’：本节点所有邻居节点的投票权重与距离成反比 
[callable]:一个可调用对象，传入距离的数组，返回同样形状的权重数组 
algorithm:一个字符串，指定计算最近邻的算法 
‘ball_tree’:使用BallTree算法 
‘kd_tree’:使用KdTree算法 
‘brute’:使用暴力搜索法 
‘auto’:自动选择最合适的算法 
leaf_size:一个整数，指定BallTree或者KdTree叶节点的规模，它影响树的构建和查询速度 
mertic:一个字符串，指定距离的度量。默认为’minkowski’ 
p：整数值，指定在’Minkowski’度量的指数，为1，则代表曼哈顿距离，为2代表欧式距离 
n_jobs:并行性。若为-1，则将派发到所有CPU上 
KNN回归 
class sklearn.neighbor.KNeighborsRegresssor(n_neighbors=5,weights=’uniform’, 
algorithm=’auto’,leaf_size=30,p=2,mertic=’minkowski’, 
mertic_params=None,n_jobs=1,**kwargs) 
具体参数含义如下： 
n_neighbors:一个整数，指定k值 
weights:一个字符串或者可调用对象，指定投票权重类型。 
如果为’uniform’:本节点的所有邻居节点投票权重相同 
如果为’distance’：本节点所有邻居节点的投票权重与距离成反比 
[callable]:一个可调用对象，传入距离的数组，返回同样形状的权重数组 
algorithm:一个字符串，指定计算最近邻的算法 
‘ball_tree’:使用BallTree算法 
‘kd_tree’:使用KdTree算法 
‘brute’:使用暴力搜索法 
‘auto’:自动选择最合适的算法 
leaf_size:一个整数，指定BallTree或者KdTree叶节点的规模，它影响树的构建和查询速度 
mertic:一个字符串，指定距离的度量。默认为’minkowski’ 
p：整数值，指定在’Minkowski’度量的指数，为1，则代表曼哈顿距离，为2代表欧式距离 
n_jobs:并行性。若为-1，则将派发到所有CPU上

"""

def load_data():
    digits = datasets.load_digits()
    X_train = digits.data
    y_train = digits.target
    return train_test_split(X_train, y_train, test_size=0.25,
                            random_state=0, stratify=y_train)


def create_regression_data(n):
    X = 5 * np.random.rand(n, 1)
    y = np.sin(X).ravel()
    y[::5] += 1 * (0.5 - np.random.rand(int(n / 5)))
    return train_test_split(X, y, test_size=0.25, random_state=0)


def test_KNeighorsClassifier(*data):
    X_train, X_test, y_train, y_test = data
    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_train, y_train)
    print("training score：%f" % clf.score(X_train, y_train))
    print("testing score:%f" % clf.score(X_test, y_test))


def test_KNeighorsClassifier_K_w(*data):
    X_train, X_test, y_train, y_test = data
    Ks = np.linspace(1, y_train.size, num=100, endpoint=False, dtype='int')
    weights = ['uniform', 'distance']

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for weight in weights:
        training_scores = []
        testing_scores = []
        for K in Ks:
            clf = neighbors.KNeighborsClassifier(weights=weight, n_neighbors=K)
            clf.fit(X_train, y_train)
            testing_scores.append(clf.score(X_test, y_test))
            training_scores.append(clf.score(X_train, y_train))
        ax.plot(Ks, testing_scores, label="testing score:weight=%s" % weight)
        ax.plot(Ks, training_scores, label="training score:weight=%s" % weight)
    ax.legend(loc='best')
    ax.set_xlabel("K")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.set_title("KNeighborsClassifier")
    plt.show()


def test_KNeighborsClassifier_k_p(*data):
    X_train, X_test, y_train, y_test = data
    Ks = np.linspace(1, y_train.size, endpoint=False, dtype='int')
    Ps = [1, 2, 10]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for P in Ps:
        training_scores = []
        testing_scores = []
        for K in Ks:
            clf = neighbors.KNeighborsClassifier(p=P, n_neighbors=K)
            clf.fit(X_train, y_train)
            testing_scores.append(clf.score(X_test, y_test))
            training_scores.append(clf.score(X_train, y_train))
        ax.plot(Ks, testing_scores, label="testing score:p=%d" % P)
        ax.plot(Ks, training_scores, label="training score:p=%d" % P)
    ax.legend(loc='best')
    ax.set_xlabel("K")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.set_title("KNeighborsClassifier")
    plt.show()


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()
    test_KNeighborsClassifier_k_p(X_train, X_test, y_train, y_test)
    # test_KNeighorsClassifier(X_train,X_test,y_train,y_test)
    # test_KNeighorsClassifier_K_w(X_train,X_test,y_train,y_test)
