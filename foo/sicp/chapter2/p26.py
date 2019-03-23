def zero():
    return lambda f: lambda x: x


def add_1(n):
    return lambda f: lambda x: f(n(f)(x))


def one():
    return lambda f: lambda x: f(x)


def two():
    return lambda f: lambda x: f(f(x))



            if __name__ == '__main__':
    print(one()(lambda x: x + 1)(1))
