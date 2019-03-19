def sqrt(x, precsion, guess):
    cur_precsion = 1.0
    d = 1.0
    while cur_precsion >= precsion:
        d, cur_precsion = guess(x, d)
    return d


def guess2(x, y):
    g = x * 1.0 / y
    b = (g + y) / 2
    return b, abs(g - b)


def guess3(x, y):
    g = x * 1.0 / (y * y)
    b = (x / (y * y) + 2 * y) / 3
    return b, abs(g - b)


def sqrt1(x, precsion):
    def good_enough(guess):
        return abs(guess * guess - x) < precsion

    def improve(guess):
        return (guess + x / guess) / 2

    def sqrt_iter(guess):
        if good_enough(guess):
            return guess
        else:
            return sqrt_iter(improve(guess))

    return sqrt_iter(1.0)


def factorial(n):
    if n == 1:
        return 1
    else:
        return n * factorial(n - 1)


def factorial1(n):
    # 迭代计算，尾递归
    def fact_iter(product, counter):
        if counter == 1:
            return product
        else:
            return fact_iter(counter * product, counter - 1)

    return fact_iter(1, n)


def fib(n):
    def fib_iter(a, b, count):
        if count == 0:
            return b
        else:
            return fib_iter(a + b, a, count - 1)

    return fib_iter(1, 0, n)


def count_change(amount1):
    memory = [[0] * 6 for i in range(amount1 + 1)]

    def cc(amount, kinds_of_coins):
        if memory[amount][kinds_of_coins] != 0:
            return memory[amount][kinds_of_coins]
        if amount == 0:
            return 1
        elif amount < 0 or kinds_of_coins == 0:
            return 0
        else:
            memory[amount][kinds_of_coins] = cc(amount, kinds_of_coins - 1) + cc(amount - first_denomination(kinds_of_coins), kinds_of_coins)
            return memory[amount][kinds_of_coins]

    def first_denomination(kinds_of_coins):
        conds = {1: 1, 2: 5, 3: 10, 4: 25, 5: 50}
        return conds.get(kinds_of_coins)

    return cc(amount1, 5)


def calucalate_f(n):
    if n < 3:
        return n
    else:
        return calucalate_f(n - 1) + 2 * calucalate_f(n - 2) + 3 * calucalate_f(n - 3)


def calucalate_f1(n1):
    def cal_iter(f1, f2, f3, n):
        if n < 3:
            return n
        else:
            return cal_iter(f1 + 2 * f2 + 3 * f3, f1, f2, n - 1)

    if n1 < 3:
        return n1
    else:
        return cal_iter(2, 1, 0, n1)


def triangle(n):
    memory = [[0] * n for i in range(n)]

    def recursive(i, j):
        if memory[i][j] != 0:
            return memory[i][j]
        if j == 0 or j == i:
            memory[i][j] = 1
            return 1
        else:
            return recursive(i - 1, j) + recursive(i - 1, j - 1)

    for i in range(0, n):
        xs = []
        for j in range(0, i + 1):
            xs.append(recursive(i, j))
        print(xs)

# yield在函数中的功能类似于return，不同的是yield每次返回结果之后函数并没有退出，
# 而是每次遇到yield关键字后返回相应结果，并保留函数当前的运行状态，等待下一次的调用。
# 如果一个函数需要多次循环执行一个动作，并且每次执行的结果都是需要的，这种场景很适合使用yield实现。
# 包含yield的函数成为一个生成器，生成器同时也是一个迭代器，支持通过next方法获取下一个值。

def triangles1(n1):
    def answer():
        xs = [1]
        while True:
            yield xs
            xs = [1] + [xs[i - 1] + xs[i] for i in range(1, len(xs))] + [1]

    def println(n):
        for t in answer():
            print(t)
            n -= 1
            if n == 0:
                break

    println(n1)


if __name__ == '__main__':
    triangles1(10)
    # print(calucalate_f1(200))
    # print(count_change(2000))
    # print(fib(100))
    # print(factorial1(20))
    # print(sqrt1(9, 0.00000001))
    # print(sqrt(9, 0.00000001, guess2))
    # print(sqrt(9, 0.00000001, guess3))
