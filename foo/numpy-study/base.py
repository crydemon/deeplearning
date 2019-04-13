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
    a = []
    a.append([1, 1])
    a.append([0, 1])
    b = []
    b.append([2, 0])
    b.append([3, 4])
    a = np.array(a)
    b = np.array(b)
    print(a.dot(b))
    print(np.dot(a, b))

    a = np.ones((2, 3), dtype=int)
    b = np.random.random((2, 3))
    print(a * 3 + b + a)
    print("-----")
    # a += b # b is not automatically converted to integer type

    # size = (a, b, c)
    # 维度 == size中参数的个数
    # # a指最外层1（只看最外层括号）看本层元素个数，本例中指有两层括号的元素个数
    # b指只看次层2层括号内本层元素个数，本例中指有一层括号的元素个数
    # c指只看3层最内层本层元素个数。，无括号的元素个数。
    # ** [] ** 代表维度，增加一层括号即增加一维。

    a = np.arange(24).reshape(2, 3, 4)
    print(a)
    # axis = 0 剩下的矩阵为（3， 4），沿0相加
    print(a.sum(axis=0))
    print(a.sum(axis=0).sum(axis=0))
    print(a.sum(axis=0).sum(axis=0).sum(axis=0))
    # axis = 0 剩下的矩阵为（2， 4）
    print(a.sum(axis=1))
    print(a.sum(axis=1).sum(axis=1))
    print(a.sum(axis=1).sum(axis=1).sum(axis=0))
    # axis = 0 剩下的矩阵为（2， 3）
    print(a.sum(axis=2))
    print(a.sum(axis=2).sum(axis=1))
    print(a.sum(axis=2).sum(axis=1).sum(axis=0))

    a = np.arange(10) ** 3
    print(a)
    print(a[2])
    print(a[2:5])
    print(a[::2])
    print(a[::-1])
    a = np.arange(24).reshape(2, 3, 4)
    print(a[:, :, ::2])
