from sklearn import preprocessing
import numpy as np

x = np.arange(4).reshape(2, 2)
print(x)
print(np.sum(x, axis=0))
x_scale = preprocessing.scale(x, axis=0)
print(x_scale.std(axis=0))
print(x_scale)
print(x_scale.mean(axis=0))
# 调用fit方法，根据已有的训练数据创建一个标准化的转换器
scaler = preprocessing.StandardScaler().fit(x)
print(scaler)
# 使用上面这个转换器去转换训练数据x,调用transform方法
print(scaler.transform(x))
min_max_scaler = preprocessing.MinMaxScaler()
x_minmax = min_max_scaler.fit_transform(x)
x_minmax
x_normalized = preprocessing.normalize(x, norm='l2')
print(x)
print(x_normalized)
# 根据训练数据创建一个正则器
normalizer = preprocessing.Normalizer().fit(x)
print(normalizer)
# 对训练数据进行正则
normalizer.transform(x)

from sklearn.preprocessing import FunctionTransformer

transformer = FunctionTransformer(np.log1p)

x = np.array([[0, 1], [2, 3]])

print(transformer.transform(x))
# 如果你的数据有许多异常值，那么使用数据的均值与方差去做标准化就不行了。
# 在这里，你可以使用robust_scale 和 RobustScaler这两个方法。它会根据中位数或者四分位数去中心化数据。
scaler = preprocessing.RobustScaler(with_centering=False)
scaler.fit(x)
