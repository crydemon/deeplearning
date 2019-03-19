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


if __name__ == '__main__':
    print(sqrt(9, 0.00000001, guess2))
    print(sqrt(9, 0.00000001, guess3))
