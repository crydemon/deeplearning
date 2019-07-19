import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    data_set = pd.read_csv("Social_Network_Ads.csv")
    X = data_set.iloc[:, [2, 3]].values
    Y = data_set.iloc[:, 4].values
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    from sklearn.linear_model import LogisticRegression

    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, y_pred)

    from matplotlib.colors import ListedColormap

    X_set, y_set = X_train, y_train
    # 生成网格数据
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                         np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))

    # 填充等高线的颜色, 8是等高线分为几部分
    # mp.contourf(x, y, z, 等高线条数，cmap=颜色映射)# 等高线填充
    # mp.contour(x, y, z, 等高线条数，colors=颜色, linewidth=线宽)#等高线绘制
    # alpha 控制颜色深浅
    # 画出所有预测点
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green')))
    # index, value
    for i, j in enumerate(np.unique(y_set)):
        print(i)
        # y_set == 0 是index 的筛选条件
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c=ListedColormap(('red', 'green'))(i), label=j)

    plt.title(' LOGISTIC(Training set)')
    plt.xlabel(' Age')
    plt.ylabel(' Estimated Salary')
    plt.legend()
    plt.show()

    # ravel()：如果没有必要，不会产生源数据的副本
    # flatten()：返回源数据的副本
    # squeeze()：只能对维数为1的维度降维
    X_set, y_set = X_test, y_test
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                         np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))

    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green')))
    plt.show()
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c=ListedColormap(('red', 'green'))(i), label=j)

    plt.title(' LOGISTIC(Test set)')
    plt.xlabel(' Age')
    plt.ylabel(' Estimated Salary')
    plt.legend()
    plt.show()
