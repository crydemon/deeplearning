import math


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


def expt(b1, n1):
    def fast_expt(b, n, a):
        if n == 0:
            return a
        elif n & 1 == 0:
            return fast_expt(b * b, n >> 1, a)
        else:
            return fast_expt(b, n - 1, b * a)

    return fast_expt(b1, n1, 1)


def multi(a1, b1):
    def double(n):
        return n << 1

    def halve(n):
        return n >> 1

    def multi_iter(a, b, p):
        if b == 0:
            return p
        elif b & 1 == 0:
            return multi_iter(double(a), halve(b), p)
        else:
            return multi_iter(a, b - 1, p + a)

    return multi_iter(a1, b1, 0)


def fib_1(n):
    def fib_iter(a, b, p, q, count):
        if count == 0:
            return b
        elif count & 1 == 0:
            return fib_iter(a, b, p * p + q * q, 2 * p * q + q * q, count >> 1)
        else:
            return fib_iter(b * q + a * p + a * q, b * p + a * q, p, q, count - 1)

    return fib_iter(1, 0, 0, 1, n)


# 求和公式
def sum_1(term, next_a, a, b):
    # Normal recursion depth maxes out at 10000
    def sum_iter(cur_a, result):
        if cur_a > b:
            return result
        else:
            return sum_iter(next_a(cur_a), term(cur_a) + result)

    return sum_iter(a, 0)


# 积分求法
def integral(f, a, b, dx):
    return sum_1(f, lambda x: x + dx, a + dx / 2, b) * dx


# 积分求法
def simpson(f, a, b, n):
    h = (b - a) / n

    def yk(k):
        if k == 0 or k == n:
            m = 1
        elif k & 1 == 1:
            m = 4
        else:
            m = 2
        return m * f(a + k * h)

    return h / 3 * sum_1(yk, lambda x: x + 1, 0, n)


def product(term, next_a, a, b):
    def iter_product(cur_a, result):
        if cur_a > b:
            return result
        else:
            return iter_product(next_a(cur_a), term(cur_a) * result)

    if a == 0:
        return 0
    else:
        return iter_product(a, 1)


def factorial_2(n):
    return product(lambda x: x, lambda x: x + 1, 1, n)


def pi(n):
    def yk(k):
        if k & 1 == 1:
            m = 3 + k - 1
            return (m - 1) / m
        else:
            m = 3 + k - 2
            return (m + 1) / m

    return 4 * product(yk, lambda k: k + 1, 1, n)


def accumulate(combiner, null_value, term, netx_a, a, b):
    def iter_acc(cur_a, result):
        if cur_a > b:
            return result
        else:
            return iter_acc(netx_a(cur_a), combiner(term(cur_a), result))

    return iter_acc(a, null_value)


def sum_2(term, next_a, a, b):
    return accumulate(lambda x, y: x + y, 0, term, next_a, a, b)


def search(f, neg_point, pos_point):
    def mid_point(x1, x2):
        return (x2 - x1) / 2 + x1

    def close_enough(x, y):
        return abs(x - y) < 0.001

    def iter_search(x1, x2):
        mid = mid_point(x1, x2)
        if close_enough(f(x1), f(x2)):
            return mid
        elif f(mid) < 0:
            return search(f, mid, x2)
        else:
            return search(f, x1, mid)

    if f(neg_point) * f(pos_point) > 0:
        return "fool init"
    else:
        return iter_search(neg_point, pos_point)


def fixed_point(f, first_guess):
    tolerance = 0.00001

    def close_enough(v1, v2):
        return abs(v1 - v2) < tolerance

    def go(guess):
        next_g = f(guess)
        if close_enough(next_g, guess):
            return next_g
        else:
            return go(next_g)

    return go(first_guess)


# (x / y + y) /2 控制振荡， 平均阻尼技术
def sqrt2(x):
    return fixed_point(lambda y: (x / y + y) / 2, 1.0)


def golden_rate1():
    return fixed_point(lambda x: 1 + 1 / x, 1.0)


def average_damp(f):
    return lambda x: (f(x) + x) / 2


def sqrt3(x):
    return fixed_point(average_damp(lambda y: x / y), 1.0)


def cube_root(x):
    return fixed_point(average_damp(lambda y: x / (y * y)), 1.0)


def deriv(g):
    dx = 0.000001
    return lambda x: (g(x + dx) - g(x)) / dx


def newton_tranform(g):
    return lambda x: (x - g(x) / deriv(g)(x))


def newton_method(g, guess):
    return fixed_point(newton_tranform(g), guess)


def sqrt4(x):
    return newton_method(lambda y: x - y * y, 1.0)


def fixed_point_of_transform(g, transform, guess):
    return fixed_point(transform(g), guess)


def sqrt5(x):
    return fixed_point_of_transform(lambda y: y * y - x, newton_tranform, 1.0)


def sqrt6(x):
    return fixed_point_of_transform(lambda y: x / y, average_damp, 1.0)


def cubic(a, b, c):
    return lambda x: x * x * x + a * x * x + b * x + c


def double(f):
    return lambda x: f(f(x))


def compose(f, g):
    return lambda x: f(g(x))


if __name__ == '__main__':
    print(compose(lambda x: x * x, lambda x: x + 1)(6))
    print((double(double(double)))(lambda x: x + 1)(5))
    # print(newton_method(cubic(3, 2, 1), 1.0))
    # print(sqrt6(5))
    # print(sqrt5(5))
    # print(sqrt4(5))
    # print(deriv(lambda x: x * x * x)(5))
    # print(cube_root(3))
    # print(sqrt3(3))
    # print(average_damp(lambda x: x * x)(10))
    # print(golden_rate1())
    # print(fixed_point(lambda x: math.sin(x) + math.cos(x), 1.0))
    # print(search(lambda x: x * x * x - 2 * x - 3, 1.0, 2.0))
    # print(sum_2(lambda x: x, lambda x: x + 1, 1, 2000))
    # print(pi(2100))
    # print(simpson(lambda x: x * x * x, 0, 1, 1000))
    # print(integral(lambda x: x * x * x, 0, 1, 0.001))
    # print(8 * sum_1(lambda x: 1 / (x * (x + 2)), lambda x: x + 4, 1, 5000))
    # print(sum_1(lambda x: x * x * x, lambda x: x + 1, 1, 10))
    # print(sum_1(lambda x: x, lambda x: x + 1, 1, 2000))
    # print(fib(1000000))
    # print(multi(7, 8))
    # print(expt(2, 10000))
    # triangles1(10)
    # print(calucalate_f1(200))
    # print(count_change(2000))
    # print(fib(100))
    # print(factorial1(20))
    # print(sqrt1(9, 0.00000001))
    # print(sqrt(9, 0.00000001, guess2))
    # print(sqrt(9, 0.00000001, guess3))
