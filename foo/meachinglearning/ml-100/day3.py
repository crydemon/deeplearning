import pandas as pd
import numpy as np

dataset = pd.read_csv('50_Startups.csv')
# print(dataset)
X = dataset.iloc[:, :-1].values
# print(X)
Y = dataset.iloc[:, 4].values

print(dataset.dtypes)
print(dataset['State'].drop_duplicates())
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Encoding Categorical data
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), [3])],
    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'  # Leave the rest of the columns untouched
)
X = np.array(ct.fit_transform(X), dtype=float)
print(X[:, 3])
# Avoiding Dummy Variable Trap
X = X[:, 1:]
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1 / 5, random_state=0)
# Step 2: Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, Y_train)
# Predicting the Test set results
y_pred = regressor.predict(X_test)
print(y_pred)
# Returns the coefficient of determination R^2 of the prediction
# 1 - 预测值的均值的计算的方差residual/真实均值的方差
print(regressor.score(X_test, Y_test))
