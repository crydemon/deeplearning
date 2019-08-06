# !/usr/bin/python
# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import sklearn as sk
import math
# import time
# import gc
import xgboost as xgb
# from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
# from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_log_error
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, chi2, SelectPercentile, GenericUnivariateSelect
from sklearn.preprocessing import Imputer, OneHotEncoder, PolynomialFeatures

way_data = ''
train0 = pd.read_csv(way_data + 'train.csv')
train0 = train0.set_index('Id')
test0 = pd.read_csv(way_data + 'test.csv')
test0 = test0.set_index('Id')

train0_x = train0.iloc[:, :-1]
train0_y = train0.SalePrice
test0_x = test0.copy()

all_0 = pd.concat([train0_x, test0_x])

###
# 把nan换成-999
all_0.fillna(-999, inplace=True)
###

# 分定性和定量
qualitative_0 = pd.DataFrame()
quantitative_0 = pd.DataFrame()
for k in all_0.columns:
    dtype_k = all_0[k].dtype
    if dtype_k == 'object':  # 定性
        qualitative_0[k] = all_0[k]
    else:
        quantitative_0[k] = all_0[k]

qualitative_1 = qualitative_0.copy()
quantitative_1 = quantitative_0.copy()

# 定性编码
# for k in qualitative_1.columns:
#     le = sk.preprocessing.LabelEncoder()
#     qualitative_1[k][qualitative_1[k].notnull()] = le.fit_transform(qualitative_1[k][qualitative_1[k].notnull()])
# 定性补缺
# imputer=Imputer(strategy='most_frequent')
# qualitative_1 = pd.DataFrame(imputer.fit_transform(qualitative_1),index=qualitative_1.index)
# 定性哑编码
encoder = OneHotEncoder(sparse=False, dtype=np.int)
qualitative_1 = pd.DataFrame(encoder.fit_transform(qualitative_1), index=qualitative_1.index)

# 定量补缺
# imputer=Imputer(strategy='mean')
# quantitative_1 = pd.DataFrame(imputer.fit_transform(quantitative_1),index=quantitative_1.index)
# 数据变换（升维）
pf = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
quantitative_1 = pd.DataFrame(pf.fit_transform(quantitative_1), index=quantitative_1.index)

# 定性和定量合并
all_1 = pd.concat([qualitative_1, quantitative_1], axis=1)
all_1.columns = np.arange(all_1.shape[1])

# 分成train和test
train1_x = all_1.loc[train0_x.index, :]
test1_x = all_1.loc[test0_x.index, :]
train1_y = train0_y.copy()
# train1_y=train0_y.apply(float)/800000
# train1_y=train0_y.apply(float)
'''
#判别分析（降维）
skb=SelectKBest(chi2,k=400)
all_2=skb.fit_transform(train1_x,train1_y)
'''
########################
train_x, test_x, train_y, test_y_real = sk.model_selection.train_test_split(train1_x, train1_y, test_size=0.3)
########################
'''
#判别分析（降维）
skb=SelectKBest(chi2,k=3000)        ###
train_x=skb.fit_transform(train_x,train_y)
test_x=skb.transform(test_x)
'''

dtrain = xgb.DMatrix(train_x, train_y)
dtest = xgb.DMatrix(test_x, test_y_real)  ###

watchlist = [(dtrain, 'train'), (dtest, 'test')]


# num_class=train_y.max()+1

def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    # return a pair metric_name, result
    # since preds are margin(before logistic transformation, cutoff at 0)
    return 'error', math.sqrt(mean_squared_log_error(preds, labels))


#############

params = {
    'objective': 'reg:gamma',
    'eta': 0.1,
    'seed': 0,
    'missing': -999,
    # 'num_class':num_class,
    'silent': 1,
    'gamma': 0.5,
    'subsample': 0.5,
    'alpha': 0.5,
    'max_depth': 5,
    'min_child_weight': 1,
    # 'colsample_bytree':0.6,
    # 'colsample_bylevel':1
    # 'max_delta_step':0.8
}
num_rounds = 3000
clf = xgb.train(params, dtrain, num_rounds, watchlist, feval=evalerror)
