import time


def work1():
    for i in range(5):
        print('work1', i)
        yield
        time.sleep(1)


def work2():
    for i in range(5):
        print('work2', i)
        yield
        time.sleep(1)


w1 = work1()
w2 = work2()
for i in range(5):
    next(w1)
    next(w2)
    print('-----------')
