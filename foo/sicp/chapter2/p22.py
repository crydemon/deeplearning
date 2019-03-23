# 抽象屏障

# 段的程序
# 问题域中的点
# 点的表示
# 作为有序数对

# 构造函数
# 选择函数


def make_segement(sp, ep):
    return [sp, ep]


def start_segment(seg):
    return seg[0]


def end_segment(seg):
    return seg[1]


def make_point(x, y):
    return [x, y]


def x_point(p):
    return p[0]


def y_point(p):
    return p[1]


def mid_point_segment(seg):
    s = start_segment(seg)
    e = end_segment(seg)
    return make_point(average(x_point(s), x_point(e)), average(y_point(s), y_point(e)))


def average(x, y):
    return (x + y) / 2


def print_point(p):
    print()
    print('(', end='')
    print(x_point(p), end='')
    print(',', end='')
    print(y_point(p), end='')
    print(')', end='')


if __name__ == '__main__':
    sp = make_point(1, 3)
    ep = make_point(4, 3)
    seg = make_segement(sp, ep)
    print_point(mid_point_segment(seg))
