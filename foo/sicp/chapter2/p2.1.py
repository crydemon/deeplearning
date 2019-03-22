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
