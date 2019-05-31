import  numpy as np

if __name__ == '__main__':
    a = np.arange(24).reshape(2, 3, 4)
    print(a.shape)
    print('a')
    print(a)
    print('axis = 0')
    print(a.sum(axis = 0))
    print('axis = 1')
    print(a.sum(axis=1))
    print('axis = 2')
    print(a.sum(axis=2))