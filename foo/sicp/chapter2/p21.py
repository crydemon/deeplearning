def map1(proc, items):
    if items is None:
        return None
    elif len(items) == 1:
        return [proc(items[0])]
    else:
        return [proc(items[0])] + map1(proc, items[1:])


def square_list(items):
    return map1(lambda x: x * x, items)


if __name__ == '__main__':
    print(square_list([1, 2, 3, 4]))
