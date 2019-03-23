import p22 as point


def perimeter_rectangle(r):
    l = length_of_rectangle(r)
    w = width_of_rectangle(r)
    return (l + w) * 2


def area_rectangle(r):
    length = length_of_rectangle(r)
    width = width_of_rectangle(r)
    return length * width


def make_rectangle(length_1, length_2, width_1, width_2):
    return [[length_1, length_2], [width_1, width_2]]


def length_of_rectangle(r):
    return r[0][0][1] - r[0][0][0]


def width_of_rectangle(r):
    return r[1][0][1] - r[1][0][0]


def length_1_rectangle(r):
    return r[0][0]


def length_2_rectangle(r):
    return r[0][1]


def width_1_rectangle(r):
    return r[1][0]


def width_2_rectangle(r):
    return r[1][1]


def print_rectangle(r):
    l1 = length_1_rectangle(r)
    l2 = length_2_rectangle(r)
    w1 = width_1_rectangle(r)
    w2 = width_2_rectangle(r)
    print()
    print("length 1:")
    point.print_point(point.start_segment(l1))
    point.print_point(point.end_segment(l1))
    print()
    print("length 2:")
    point.print_point(point.start_segment(l2))
    point.print_point(point.end_segment(l2))

    print()
    print("width 1:")
    point.print_point(point.start_segment(w1))
    point.print_point(point.end_segment(w1))
    print()
    print("width 2:")
    point.print_point(point.start_segment(w2))
    point.print_point(point.end_segment(w2))


if __name__ == '__main__':
    p1 = point.make_point(1, 4)
    p2 = point.make_point(4, 4)
    p3 = point.make_point(1, 2)
    p4 = point.make_point(4, 2)
    len_1 = point.make_segement(p1, p2)
    len_2 = point.make_segement(p3, p4)

    w_1 = point.make_segement(p4, p2)
    w_2 = point.make_segement(p3, p1)
    r = make_rectangle(len_1, len_2, w_1, w_2)
    print_rectangle(r)
