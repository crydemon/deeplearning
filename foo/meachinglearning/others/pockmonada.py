import numpy as np
import matplotlib.pyplot as plt

x_data = [338.,226., 25., 179., 60., 208., 333., 328., 207., 606.]
y_data = [640.,428., 27., 193., 66., 226., 633., 619., 393., 1591.]
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
lr = 1
iteration = 100000

b_history = [b]
w_history = [w]

lr_b = 0
lr_w = 0

for i in range(iteration):
    b_grad = 0.0
    w_grad = 0.0
    for n in range(len(x_data)):
        b_grad -= 2.0 * (y_data[n] - b - w * x_data[n]) * 1.0
        w_grad -= 2.0 * (y_data[n] - b - w * x_data[n]) * x_data[n]

    lr_b += b_grad ** 2
    lr_w += w_grad ** 2

    b -= lr / np.sqrt(lr_b) * b_grad
    w -= lr / np.sqrt(lr_w) * w_grad
    b_history.append(b)
    w_history.append(w)

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
plt.plot(b_history, w_history, 'o-', ms=2, lw=1.0, color='black')
plt.xlim(-200, -100)
plt.ylim(-5, 5)
plt.xlabel(r'$b$', fontsize=16)
plt.ylabel(r'$w$', fontsize=16)
plt.show()
