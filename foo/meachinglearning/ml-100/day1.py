import numpy as np
import pandas as pd

if __name__ == '__main__':
    dataset = pd.read_csv('Data.csv')
    X = dataset.iloc[:, :-1].values
    Y = dataset.iloc[:, 3].values
    print("Step 2: Importing dataset")
    print("X")
    print(X)
    print("Y")
    print(Y)
    # Step 3: Handling the missing data
    from sklearn.impute import SimpleImputer as Imputer

    imputer = Imputer(missing_values=np.nan, strategy='mean')
    imputer = imputer.fit(X[:, 1:3])
    X[:, 1:3] = imputer.transform(X[:, 1:3])
    print("---------------------")
    print("Step 3: Handling the missing data")
    print("step2")
    print("X")
    print(X)
    # Step 4: Encoding categorical data
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    from sklearn.compose import ColumnTransformer

    labelencoder_X = LabelEncoder()
    X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
    # Creating a dummy variable
    # 将离散型特征使用one-hot编码，确实会让特征之间的距离计算更加合理。
    # 比如，有一个离散型特征，代表工作类型，该离散型特征，共有三个取值，不使用one-hot编码，其表示分别是x_1 = (1), x_2 = (2), x_3 = (3)。
    # 两个工作之间的距离是，(x_1, x_2) = 1, d(x_2, x_3) = 1, d(x_1, x_3) = 2。那么x_1和x_3工作之间就越不相似吗？
    # 显然这样的表示，计算出来的特征的距离是不合理。
    # 那如果使用one-hot编码，则得到x_1 = (1, 0, 0), x_2 = (0, 1, 0), x_3 = (0, 0, 1)，
    # 那么两个工作之间的距离就都是sqrt(2).即每两个工作之间的距离是一样的，显得更合理。


    # 首先，one-hot编码是N位状态寄存器为N个状态进行编码的方式 
    # eg：高、中、低不可分，→ 用0 0 0 三位编码之后变得可分了，并且成为互相独立的事件 
    # → 类似 SVM中，原本线性不可分的特征，经过project之后到高维之后变得可分了 
    # GBDT处理高维稀疏矩阵的时候效果并不好，即使是低维的稀疏矩阵也未必比SVM好。

    # Tree Model不太需要one-hot编码
    # 对于决策树来说，one-hot的本质是增加树的深度 
    # tree-model是在动态的过程中生成类似 One-Hot + Feature Crossing 的机制 
    # 1. 一个特征或者多个特征最终转换成一个叶子节点作为编码 ，one-hot可以理解成三个独立事件 
    # 2. 决策树是没有特征大小的概念的，只有特征处于他分布的哪一部分的概念 
    # one-hot可以解决线性可分问题 但是比不上label econding 
    # one-hot降维后的缺点： 
    # 降维前可以交叉的降维后可能变得不能交叉 

    onehotencoder = OneHotEncoder(categorical_features=[0])
    onehotencoder = ColumnTransformer(
        [('one_hot_encoder', OneHotEncoder(), [0])],
        # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
        remainder='passthrough'  # Leave the rest of the columns untouched
    )
    X = onehotencoder.fit_transform(X)
    labelencoder_Y = LabelEncoder()
    Y = labelencoder_Y.fit_transform(Y)
    print("---------------------")
    print("Step 4: Encoding categorical data")
    print("X")
    print(X)
    print("Y")
    print(Y)

    # Step 5: Splitting the datasets into training sets and Test sets
    from sklearn.model_selection import train_test_split

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    print("---------------------")
    print("Step 5: Splitting the datasets into training sets and Test sets")
    print("X_train")
    print(X_train)
    print("X_test")
    print(X_test)
    print("Y_train")
    print(Y_train)
    print("Y_test")
    print(Y_test)

    # Step 6: Feature Scaling
    from sklearn.preprocessing import StandardScaler
    # 特征量化
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    print("---------------------")
    print("Step 6: Feature Scaling")
    print("X_train")
    print(X_train)
    print("X_test")
    print(X_test)
