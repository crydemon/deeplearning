def cons(x, y):
    return lambda m: m(pow(2, x), pow(3, y))


def car(z):
    return z(lambda p, q: p)


def cdr(z):
    return z(lambda p, q: q)


def multi(z):
    return cdr(z) * car(z)


if __name__ == '__main__':
    print(multi(cons(3, 2)))
    print(car(cons(3, 2)))
    print(cdr(cons(3, 2)))
