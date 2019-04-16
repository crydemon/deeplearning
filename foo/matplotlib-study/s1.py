import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl

if __name__ == '__main__':
    # 直接使用Artists创建图表的标准流程如下：
    #     #
    #     # 创建Figure对象
    #     # 用Figure对象创建一个或者多个Axes或者Subplot对象
    #     # 调用Axies等对象的方法创建各种简单类型的Artists
    plt.figure(1)  # 创建图表1
    plt.figure(2)  # 创建图表2
    ax1 = plt.subplot(211)  # 在图表2中创建子图1
    ax2 = plt.subplot(212)  # 在图表2中创建子图2

    x = np.linspace(0, 3, 100)
    for i in range(5):
        plt.figure(1)  # ❶ # 选择图表1
        plt.plot(x, np.exp(i * x / 3))
        plt.sca(ax1)  # ❷ # 选择图表2的子图1
        plt.plot(x, np.sin(i * x))
        plt.sca(ax2)  # 选择图表2的子图2
        plt.plot(x, np.cos(i * x))

    plt.show()

    X1 = range(0, 50)
    Y1 = [num ** 2 for num in X1]  # y = x^2 X2 = [0, 1] Y2 = [0, 1] # y = x

    Fig = plt.figure(figsize=(8,
                              4))  # Create a `figure' instance Ax = Fig.add_subplot(111) # Create a `axes' instance in the figure Ax.plot(X1, Y1, X2, Y2) # Create a Line2D instance in the axes
    plt.plot(x, np.exp(7 * x / 3))
    Fig.show()
    Fig.savefig("test.pdf")

    x = [1, 2, 3, 4, 5]  # Make an array of x values
    y = [1, 4, 9, 16, 25]  # Make an array of y values for each x value

    # Example data
    a = np.arange(0, 3, .02)
    b = np.arange(0, 3, .02)
    c = np.exp(a)
    d = c[::-1]

    # Create plots with pre-defined labels.
    # Alternatively, you can pass labels explicitly when calling `legend`.
    fig, ax = plt.subplots()
    ax.plot(a, c, 'k--', label='Model length') # 虚线
    ax.plot(a, d, 'k:', label='Data length') # 	点 line style
    ax.plot(a, c + d, 'r', label='Total message length') # k black

    # Now add the legend with some customizations.
    legend = ax.legend(loc='upper center', shadow=True)

    # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
    frame = legend.get_frame()
    frame.set_facecolor('0.90')

    # Set the fontsize
    for label in legend.get_texts():
        label.set_fontsize('large')

    for label in legend.get_lines():
        label.set_linewidth(1.5)  # the legend line width
    plt.show()

    # make an array of random numbers with a gaussian distribution with
    # mean = 5.0 均值
    # rms = 3.0 标准差
    # number of points = 1000
    data = np.random.normal(5.0, 3.0, 1000)

    # make a histogram of the data array
    pl.hist(data)

    # make plot labels
    pl.xlabel('data')
    pl.show()
