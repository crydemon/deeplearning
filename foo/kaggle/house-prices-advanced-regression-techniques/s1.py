# 导入需要的模块
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 用来绘图的，封装了matplot
# 要注意的是一旦导入了seaborn，
# matplotlib的默认作图风格就会被覆盖成seaborn的格式
import seaborn as sns

from scipy import stats
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 1000)
    data_train = pd.read_csv('train.csv')
    # print(data_train.sample(n=10))
    # print(data_train['SalePrice'].describe()) # SalePrice”没有无效或者其他非数值的数据
    sns.distplot(data_train['SalePrice'])

    # 偏度（Skewness）是描述某变量取值分布对称性的统计量。
    print("Skewness: %f" % data_train['SalePrice'].skew())
    # 峰度（Kurtosis）是描述某变量所有取值分布形态陡缓程度的统计量。
    print("Kurtosis: %f" % data_train['SalePrice'].kurtosis())
    # CentralAir
    var = 'CentralAir'
    data = pd.concat([data_train['SalePrice'], data_train[var]], axis=1)
    fig = sns.boxplot(x=var, y="SalePrice", data=data)
    fig.axis(ymin=0, ymax=800000)
    plt.show()
    var = 'OverallQual'
    data = pd.concat([data_train['SalePrice'], data_train[var]], axis=1)
    fig = sns.boxplot(x=var, y="SalePrice", data=data)
    fig.axis(ymin=0, ymax=800000)
    plt.show()
    var = 'YearBuilt'
    data = pd.concat([data_train['SalePrice'], data_train[var]], axis=1)
    f, ax = plt.subplots(figsize=(26, 12))
    fig = sns.boxplot(x=var, y="SalePrice", data=data)
    fig.axis(ymin=0, ymax=800000);
    plt.show()
    var = 'YearBuilt'
    data = pd.concat([data_train['SalePrice'], data_train[var]], axis=1)
    data.plot.scatter(x=var, y="SalePrice", ylim=(0, 800000))
    plt.show()
    corrmat = data_train.corr()
    f, ax = plt.subplots(figsize=(20, 9))
    sns.heatmap(corrmat, vmax=0.8, square=True)
    plt.show()
    from sklearn import preprocessing
    f_names = ['CentralAir', 'Neighborhood']
    for x in f_names:
        label = preprocessing.LabelEncoder()
        data_train[x] = label.fit_transform(data_train[x])
    corrmat = data_train.corr()
    f, ax = plt.subplots(figsize=(20, 9))
    sns.heatmap(corrmat, vmax=0.8, square=True)
    plt.show()
    k = 10  # 关系矩阵中将显示10个特征
    cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
    cm = np.corrcoef(data_train[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, \
                     square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
    plt.show()
    sns.set()
    cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd',
            'YearBuilt']
    sns.pairplot(data_train[cols], size=2.5)
    plt.show()
