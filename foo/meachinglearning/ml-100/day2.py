import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dataset = pd.read_csv('studentscores.csv')
    #Data Preprocessing
    X = dataset.iloc[:, : 1].values
    print(X)
    Y = dataset.iloc[:, 1].values
    from sklearn.model_selection import train_test_split

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1 / 4, random_state=0)

    # Fitting Simple Linear Regression Model to the training set
    from sklearn.linear_model import LinearRegression

    regressor = LinearRegression()
    regressor = regressor.fit(X_train, Y_train)

    # Predecting the Result
    Y_pred = regressor.predict(X_test)
    print(Y_pred)
    # Visualising the Training results
    # scatter 分散
    plt.scatter(X_train, Y_train, color='red')
    plt.plot(X_train, regressor.predict(X_train), color='blue')

    # Visualizing the test results
    plt.scatter(X_test, Y_test, color='black')
    plt.plot(X_test, regressor.predict(X_test), color='black')
    plt.show()
