def add_rat(x, y):
    n = numer(x) * denom(y) + numer(y) * denom(x)
    d = denom(x) * denom(y)
    return make_rat(n, d)


def sub_rat(x, y):
    n = numer(x) * denom(y) - numer(y) * denom(x)
    d = denom(x) * denom(y)
    return make_rat(n, d)


def mul_rat(x, y):
    n = numer(x) * numer(y)
    d = denom(x) * denom(y)
    return make_rat(n, d)


def div_rat(x, y):
    n = numer(x) * denom(y)
    d = denom(x) * numer(y)
    return make_rat(n, d)


def equal_rat(x, y):
    return numer(x) * denom(y) == denom(x) * numer(y)


def print_rat(x):
    print()
    display(numer(x))
    display("/")
    display(denom(x))


def display(x):
    print(x, end='')


def make_rat(n, d):
    return [n, d]


def numer(x):
    return car(x)


def denom(x):
    return cdr(x)


def car(x):
    return x[0]


def cdr(x):
    return x[1]


def one_half():
    return make_rat(1, 2)


def one_third():
    return make_rat(1, 3)


if __name__ == '__main__':
    print_rat(one_half())
    print_rat(one_third())
    print_rat(add_rat(one_half(), one_third()))
