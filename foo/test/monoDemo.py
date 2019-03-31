from numpy import *
import matplotlib.pyplot as plt

t = arange(-1.0, 1.0, 0.0001)
s = sin(2*pi*t)
logS = log(s)

fig = plt.figure()
ax = fig.add_subplot(211) # 两行一列， 第一行第一列的子图
ax.plot(t,s)
ax.set_ylabel('f(x)')
ax.set_xlabel('x')

ax = fig.add_subplot(212)
ax.plot(t,logS)
ax.set_ylabel('ln(f(x))')
ax.set_xlabel('x')
plt.show()