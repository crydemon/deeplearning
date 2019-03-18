def sqrt(x, precsion):
    cur_precsion = 1
    d = 1.0
    while cur_precsion >= precsion:
        q = x / d
        avg = (q + d) / 2
        cur_precsion = abs(d - q)
        d = avg
    return d


if __name__ == '__main__':
    print(sqrt(2, 0.00000001))
