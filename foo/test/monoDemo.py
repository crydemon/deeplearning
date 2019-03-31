from numpy import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    t = arange(-20.0, 20.0, 0.0001)
    s = sin(2 * pi * t)
    logS = log(s)

    fig = plt.figure()
    ax = fig.add_subplot(311)  # 两行一列， 第一行第一列的子图
    ax.plot(t, s)
    ax.set_ylabel('f(x)')
    ax.set_xlabel('x')

    ax = fig.add_subplot(312)
    ax.plot(t, logS)
    ax.set_ylabel('ln(f(x))')
    ax.set_xlabel('x')

    sigmoid = 1 / (1 + exp(-t))
    ax = fig.add_subplot(313)
    ax.plot(t, sigmoid)
    ax.set_ylabel('sigmoid(x)')
    ax.set_xlabel('x')
    plt.show()
