import pandas as pd
import numpy as np

import os


# 目标函数 y = ax + b
def compute_grad(beta, x, y):
    grad = [0, 0]
    grad[0] = 2. * np.mean(beta[0] + beta[1] * x - y)
    grad[1] = 2. * np.mean(x * (beta[0] + beta[1] * x - y))
    return np.array(grad)


def compute_grad_SGD(beta, x, y):
    grad = [0, 0]
    r = np.random.randint(0, len(x))
    grad[0] = 2. * np.mean(beta[0] + beta[1] * x[r] - y[r])
    grad[1] = 2. * np.mean(x * (beta[0] + beta[1] * x[r] - y[r]))
    return np.array(grad)


def compute_grad_batch(beta, batch_size, x, y):
    grad = [0, 0]
    r = np.random.choice(range(len(x)), batch_size)
    grad[0] = 2. * np.mean(beta[0] + beta[1] * x[r] - y[r])
    grad[1] = 2. * np.mean(x * (beta[0] + beta[1] * x[r] - y[r]))
    return np.array(grad)


def update_beta(beta, alpha, grad):
    new_beta = np.array(beta) - alpha * grad
    return new_beta


def rmse(beta, x, y):
    squared_err = (beta[0] + beta[1] * x - y) ** 2
    res = np.sqrt(np.mean(squared_err))
    return res


if __name__ == '__main__':
    resource_path = os.path.abspath('../../../resources/grdient-data')
    train = pd.read_csv(resource_path + '/train.csv')
    test = pd.read_csv(resource_path + '/train.csv')
    submit = pd.read_csv(resource_path + '/sample_submit.csv')

    beta = [1, 1]
    alpha = 0.2
    tol_L = 0.01

    max_x = max(train['id'])
    x = train['id'] / max_x
    y = train['questions']
    loss = 1
    i = 0
    while True:
        grad = compute_grad(beta, x, y)
        beta = update_beta(beta, alpha, grad)
        loss_new = rmse(beta, x, y)
        if abs(loss_new - loss) <= tol_L:
            loss = loss_new
            break
        else:
            i += 1
            loss = loss_new
    print('Coef: %s \nIntercept %s' % (beta[1] / max_x, beta[0]))
    print('Round %s Diff RMSE %s' % (i, loss))

    print("---------use SGD--------------------")

    while True:
        grad = compute_grad_SGD(beta, x, y)
        beta = update_beta(beta, alpha, grad)
        loss_new = rmse(beta, x, y)
        if abs(loss_new - loss) <= tol_L:
            loss = loss_new
            break
        else:
            i += 1
            # print('Round %s Diff RMSE %s' % (i, abs(loss_new - loss)))
            loss = loss_new

    print('Coef: %s \nIntercept %s' % (beta[1] / max_x, beta[0]))
    print('Round %s Diff RMSE %s' % (i, loss))

    print("---------use Batch_SGD--------------------")

    while True:
        batch_size = np.random.randint(2, 10)
        grad = compute_grad_batch(beta, batch_size, x, y)
        beta = update_beta(beta, alpha, grad)
        loss_new = rmse(beta, x, y)
        if abs(loss_new - loss) <= tol_L:
            loss = loss_new
            break
        else:
            i += 1
            # print('Round %s Diff RMSE %s' % (i, abs(loss_new - loss)))
            loss = loss_new

    print('Coef: %s \nIntercept %s' % (beta[1] / max_x, beta[0]))
    print('Round %s Diff RMSE %s' % (i, loss))

    print("---------use sklearn--------------------")

    from sklearn.linear_model import LinearRegression

    lr = LinearRegression()
    lr.fit(train[['id']], train[['questions']])
    print('Sklearn Coef: %s' % lr.coef_[0][0])
    print('Sklearn Coef: %s' % lr.intercept_[0])
    res = rmse([lr.intercept_[0], lr.coef_[0][0]], train['id'], y)
    print('Sklearn RMSE: %s' % res)
