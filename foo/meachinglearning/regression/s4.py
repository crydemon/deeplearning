from sklearn import svm
import pandas as pd
import numpy as np

if __name__ == '__main__':
    row_data = pd.read_csv('d:/gmv.csv')
    train = row_data.loc[row_data['day_time'] <= 43575]
    test = row_data.loc[row_data['day_time'] > 43575]
    print(train)
    X = []
    y = []
    for index, row in train.iterrows():
        if row is None:
            continue
        line = []
        for key in train.columns.values:
            val = row[key]
            if val is None:
                val = 0
            if key == 'gmv':
                y.append(val)
            else:
                line.append(val)
        X.append(line)

    clf = svm.SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.2, gamma='scale', kernel='poly', max_iter=500)

    clf.fit(X, y)
    print(clf.score(X, y))

    for index, row in test.iterrows():
        if row is None:
            continue
        line = []
        for key in test.columns.values:
            val = row[key]
            if val is None:
                val = 0
            if key == 'gmv':
                print('real', val)
            else:
                line.append(val)
        result = clf.predict([line])
        print('predict', result[0])
