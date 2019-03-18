import numpy as np
import pandas as pd
import sys

test_path = 'D:/bigdata/ML2017FALL-master/hw1/test.csv'
train_path = 'D:/bigdata/ML2017FALL-master/hw1/train.csv'
output_path = 'D:/bigdata/ML2017FALL-master/hw1/out.csv'

train = pd.read_csv(train_path, encoding='big5')
test = pd.read_csv(test_path, encoding='big5', header=None)

mask = []
for i in range(len(train)):
    fil_bool = train['日期'][i][5] != '7' and train['日期'][i][5] != '8'
    mask.append(fil_bool)

# print(mask)
# Access group of values using labels
# Boolean list with the same length as the row axis
train_filt = train.loc[mask]
# print(train_filt)
variables = list(train['測項'][:18])
print(variables)


def create_period(data, seq_len):
    sequence_length = seq_len
    result = []
    for i in range(len(data) - sequence_length):
        result.append(data[i:i + sequence_length])
    return result


feature_set = []

# 使用0值表示沿着每一列或行标签\索引值向下执行方法
# 使用1值表示沿着每一行或者列标签模向执行对应的方法
for var in variables:
    var_list = []
    # iterrows()返回值为元组,(index,row)
    for index, row in train_filt.iterrows():
        if row['測項'] == var:
            # list() 方法用于将元组转换为列表。
            var_list += list(row[3:])
    if var == 'PM2.5':
        new_PM = []
        # enumerate在字典上是枚举、列举的意思
        # 对于一个可迭代的（iterable） / 可遍历的对象（如列表、字符串），enumerate将其组成一个索引序列，利用它可以同时获得索引和值
        # enumerate多用于在for循环中得到计数
        for index, pm in enumerate(np.array(var_list, dtype=float)):
            if pm < 0:
                new_PM.append(new_PM[index - 1])
            else:
                new_PM.append(pm)
        var_ts = np.array(new_PM).reshape(10, int(len(new_PM) / 10))
    # 失效值
    elif var == 'RAINFALL':
        var_list = np.array(var_list)
        # print(var_list)
        # list中的每一个进行判断
        # print(var_list == 'NR')
        var_list[var_list == 'NR'] = 0
        var_ts = np.array(var_list, dtype=float)
        var_ts = var_ts.reshape(10, 480)
    else:
        var_ts = np.array(var_list, dtype=float)
        var_ts = var_ts.reshape(10, 480)
    F = []
    for i in range(var_ts.shape[0]):
        F += create_period(var_ts[i], 9)
    feature_set.append(F)
feature_set = np.array(feature_set)
print(feature_set)
# 合并
feature_set = np.concatenate(feature_set, axis=1)

# extract ground truth
var_list = []
for index, row in train_filt.iterrows():
    if row["測項"] == "PM2.5":
        var_list += list(row[3:])
new_PM = []
for index, pm in enumerate(np.array(var_list, dtype=float)):
    if pm < 0:
        new_PM.append(new_PM[index - 1])
    else:
        new_PM.append(pm)

var_ts = np.array(new_PM).reshape((10, 480))

ground_truth = var_ts[:, 9:]
ground_truth = ground_truth.flatten()  # 產生ground truth #

# 平方PM2.5

print(feature_set[:, 81:90])
PM_sq = feature_set[:, 81:90] ** 2
feature_set = np.concatenate((feature_set, PM_sq), axis=1)


# Linear regression using Gradient Descent and ADAGrad #

class LinearRegressionGD_ADA(object):

    def __init__(self, eta=1, n_iter=10000, random_state=1, shuffle=True, alpha=0):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.shuffle = shuffle
        self.alpha = alpha

    def fit(self, X, y):
        print("fdskkk")
        print(1 + X.shape[1])
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.1, size=1 + X.shape[1])
        self.cost_ = []
        lr_b = 0
        lr_w = np.zeros(X.shape[1])
        for i in range(self.n_iter):
            b_grad = 0.0
            w_grad = np.zeros(X.shape[1])
            if self.shuffle:
                X, y = self._shuffle(X, y)
            for xi, target in zip(X, y):  # iterate on single sample
                cost = []  # record cost for each sample
                output = self.net_input(xi)
                error = (target - output)

                w_grad = w_grad - 2 * xi.dot(error)
                b_grad = b_grad - 2 * error
            #                 self.w_[1:] += 2* self.eta * xi.dot(error)
            #                 self.w_[0] += 2*self.eta * error
            lr_b = lr_b + b_grad ** 2
            lr_w = lr_w + w_grad ** 2
            self.w_[1:] = self.w_[1:] - self.eta / np.sqrt(lr_w) * w_grad + self.alpha * self.w_[1:]
            self.w_[0] = self.w_[0] - self.eta / np.sqrt(lr_b) * b_grad
            # calculate RMSE for an epoch
            errors = (sum((y - (self.net_input(X))) ** 2) / len(y)) ** 0.5
            self.cost_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return self.net_input(X)

    @staticmethod
    def _shuffle(X, y):
        r = np.random.permutation(len(y))
        return X[r], y[r]


# normalize feature
feature_df = pd.DataFrame(feature_set)
normalized_feature = np.array((feature_df - feature_df.mean()) / feature_df.std())


def build_feature_index(feature_order, period):
    index = []
    for i in feature_order:
        index += list(range(i * period, (i + 1) * period))
    return index


feature_index = build_feature_index([7, 9, 12, 18], 9)


# function for cross validation
def cross_validation(model_in, X, y, feature_index=feature_index,
                     times=5, proportion=0.5, feature_select=True):
    loss = []
    for _ in range(times):
        model = model_in
        # proportion 比例
        msk = np.random.rand(len(y)) < proportion

        train_X = X[msk]
        train_y = y[msk]

        test_X = X[~msk]
        test_y = y[~msk]

        if feature_select:
            train_X = train_X[:, feature_index]
            test_X = test_X[:, feature_index]

        model.fit(train_X, train_y)
        valid_loss = (((model.predict(test_X) - test_y) ** 2).sum() / len(test_y)) ** 0.5
        loss.append(valid_loss)
    return (sum(loss) / times, np.std(loss), loss)


# fit model
lr_ada_shf = LinearRegressionGD_ADA(eta=1.25 * 1, n_iter=10000, shuffle=True)

print("Fitting model....")
cross_loss = cross_validation(lr_ada_shf, normalized_feature, ground_truth, times=1)
print("LR_ada_shf validation Loss: ", cross_loss)

# arrange test.csv feature

# create testing feature
test_set = []
s_arr = np.array(range(0, len(test), 18))
e_arr = s_arr + 18
for start, end in zip(s_arr, e_arr):
    a = np.array(test.iloc[start:end, 2:]).flatten()
    a[a == "NR"] = 0
    a[a == "-1"] = 4
    a = np.array(a, dtype=float)
    test_set.append(a)
test_set = np.array(test_set)

# PM2.5 平方
testPM_sq = test_set[:, 81:90] ** 2
test_set = np.concatenate((test_set, testPM_sq), axis=1)
test_df = pd.DataFrame(test_set)
normalized_test = np.array((test_df - feature_df.mean()) / feature_df.std())

test_pred = lr_ada_shf.predict(normalized_test[:, feature_index])
submit_df = pd.DataFrame({"id": ["id_" + str(i) for i in range(240)], "value": test_pred})

print("save csv")
submit_df.to_csv(output_path, index=False)

# save model
# np.save('model.npy',lr_ada_shf.w_)
