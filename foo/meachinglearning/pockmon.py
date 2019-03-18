import numpy as np
import matplotlib.pyplot as plt

x_data = [338., 333., 328., 207., 226., 25., 179., 60., 208., 606.]
y_data = [640., 633., 619., 393., 428., 27., 193., 66., 226., 1591.]
# y_data = b + w * x_data

x = np.arange(-200, -100, 1)
y = np.arange(-5, 5, 0.1)
Z = np.zeros((len(x), len(y)))
[X, Y] = np.meshgrid(len(x), len(y))
for i in range(len(x)):
    for j in range(len(y)):
        b = x[i]
        w = y[j]
        Z[j][i] = 0
        for n in range(len(x_data)):
            Z[j][i] = Z[j][i] + (y_data[n] - b - w * x_data[n]) ** 2
        Z[j][i] = Z[j][i] / len(x_data)

b = -120
w = -4
lr = 0.000001
iteration = 1000000

b_history = [b]
w_history = [w]

for i in range(iteration):
    b_grad = 0.0
    w_grad = 0.0
    for n in range(len(x_data)):
        b_grad = b_grad - 2.0 * (y_data[n] - b - w * x_data[n]) * 1.0
        w_grad = w_grad - 2.0 * (y_data[n] - b - w * x_data[n]) * x_data[n]

    b = b - lr * b_grad
    w = w - lr * w_grad

    b_history.append(b)
    w_history.append(w)

# 等高线 填充等高线的颜色, 50是等高线分为几部分 透明度0.5 并将 Z 的值对应到color map的暖色组中寻找对应颜色。
# contour和contourf都是画三维等高线图的，不同点在于contourf会对等高线间的区域进行填充
plt.contourf(x, y, Z, 50, alpha=0.5, cmap=plt.get_cmap('jet'))

# 使用plt.contour函数划线。位置参数为：X, Y, f(X,Y)。颜色选黑色，线条宽度选0.5
# use plt.contour to add contour lines
# C = plt.contour(X, Y, f(X, Y), 8, colors='black', linewidth=.5)

# marker -- 折点形状
# markersize 或 ms --折点大小
# linewidth(或lw)可以改变线的粗细，其值为浮点数
# 'o' 圆形
# 'x' x号标记
plt.plot([-188.4], [2.67], 'x', ms=6, marker=6, color='orange')

# ms和marker分别代表指定点的长度和宽度。
plt.plot(b_history, w_history, 'o-', ms=6, lw=1.0, color='black')
plt.xlim(-200, -100)
plt.ylim(-5, 5)
plt.xlabel(r'$b$', fontsize=16)
plt.ylabel(r'$w$', fontsize=16)
plt.show()
