import numpy as np
import math

if __name__ == '__main__':
    arr = []
    for i in range(4):
        arr.append([np.random.randint(i, 100) for i in range(5)])
    print(arr)
    a = np.array(arr)
    print(a)
    print(a.ndim)  # 数组的轴（维度）的个数。在Python世界中，维度的数量被称为rank。
    print(a.size)
    print(a.shape)
    print(a.dtype)
    print(a.itemsize)
    print(np.arange(30).reshape(5, 6))
    print(np.array(arr, dtype=complex))
    print(np.zeros((2, 3)))
    print(np.ones((2, 4)))
    print(np.arange(0, 2, 0.3).__class__)
    print(np.arange(0, 2, 0.3))
    # 当arange与浮点参数一起使用时，由于浮点数的精度是有限的，通常不可能预测获得的元素数量。出于这个原因，通常最好使用函数
    # linspace ，它接收我们想要的元素数量而不是步长作为参数：
    print(np.linspace(0, 2, 5))
    x = np.linspace(0, 2 * math.pi, 100)
    print(x)
    y = np.sin(x)
    print(y)
    a = np.arange(48).reshape(2, 3, 8)
    print(a)
    print(a.ndim)  # 数组的轴（维度）的个数。在Python世界中，维度的数量被称为rank。
    print(a.size)
    print(a.shape)
    print(a.dtype)
    print(a.itemsize)
