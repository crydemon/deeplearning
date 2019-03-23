def cons(x, y):
    return lambda m: m(x, y)


def car(z):
    return z(lambda p, q: p)


def cdr(z):
    return z(lambda p, q: q)


if __name__ == '__main__':
    print(car(cons(1, 2)))
    print(cdr(cons(1, 2)))
