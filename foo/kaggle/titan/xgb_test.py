import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
train_df = pd.read_csv("train.csv", header=0)
test_df = pd.read_csv("test.csv", header=0)

n_folds = 12
kf = KFold(n_splits=12, random_state=42, shuffle=True)


class DataFrameImputer(TransformerMixin):
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
                               if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
                              index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


feature_columns_to_use = ['Pclass', 'Sex', 'Age', 'Fare', 'Parch']
nonnumeric_columns = ['Sex']

big_X = train_df[feature_columns_to_use].append(test_df[feature_columns_to_use])
big_X_imputed = DataFrameImputer().fit_transform(big_X)
le = LabelEncoder()
for feature in nonnumeric_columns:
    big_X_imputed[feature] = le.fit_transform(big_X_imputed[feature])

train_X = big_X_imputed[0:train_df.shape[0]]
test_X = big_X_imputed[train_df.shape[0]::]
train_y = train_df['Survived']


def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


def cv_rmse(model, X, y):
    rmse = np.sqrt(-cross_val_score(model, X.values, y, scoring="neg_mean_squared_error", cv=kf))
    return rmse


gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.01).fit(train_X, train_y)

score = cv_rmse(gbm, train_X, train_y)
# 很慢
print("xgboost: {:.4f} ({:.4f})".format(score.mean(), score.std()))

predictions = gbm.predict(test_X)

submission = pd.DataFrame({'PassengerId': test_df['PassengerId'],
                           'Survived': predictions})
submission.to_csv("submission.csv", index=False)
print("done")
