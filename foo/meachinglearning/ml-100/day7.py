"""
knn
k-d树是每个节点都为k维点的二叉树。
所有非叶子节点可以视作用一个超平面把空间分割成两个半空间。
节点左边的子树代表在超平面左边的点，节点右边的子树代表在超平面右边的点。
选择超平面的方法如下：每个节点都与k维中垂直于超平面的那一维有关。
因此，如果选择按照x轴划分，
所有x值小于指定值的节点都会出现在左子树，所有x值大于指定值的节点都会出现在右子树。
这样，超平面可以用该x值来确定，其法線为x轴的单位向量。
"""

from collections import namedtuple
from operator import itemgetter
from pprint import pformat


class Node(namedtuple('Node', 'location left_child right_child')):
    # 重构__repr__方法后，不管直接输出对象还是通过print打印的信息都按我们__repr__方法中定义的格式进行显示了
    # 你会发现，直接输出对象ts时并没有按我们__str__方法中定义的格式进行输出，而用print输出的信息却改变了
    # __repr__和__str__这两个方法都是用于显示的，__str__是面向用户的，而__repr__面向程序员。
    def __repr__(self):
        """Format a Python object into a pretty-printed representation."""
        return pformat(tuple(self))


def kd_tree(point_list, depth=0):
    try:
        k = len(point_list[0])  # assumes all points have the same dimension
    except IndexError as e:  # if not point_list:
        return None
    # Select axis based on depth so that axis cycles through all valid values
    axis = depth % k

    # Sort point list and choose median as pivot element
    # operator模块提供的itemgetter函数用于获取对象的哪些维的数据，参数为一些序号
    point_list.sort(key=itemgetter(axis))
    median = len(point_list) // 2  # choose median 向下取整

    # Create node and construct subtrees
    return Node(
        location=point_list[median],
        left_child=kd_tree(point_list[:median], depth + 1),
        right_child=kd_tree(point_list[median + 1:], depth + 1)
    )


def findmin(n, depth, cutting_dim, min):
    """
    1.2）寻找d维最小坐标值点
    a）若当前节点的切分维度是d
    因其右子树节点均大于等于当前节点在d维的坐标值，
    所以可以忽略其右子树，仅在其左子树进行搜索。若无左子树，当前节点即是最小坐标值节点。
    b）若当前节点的切分维度不是d
    需在其左子树与右子树分别进行递归搜索。
    :param n: node
    :param depth:
    :param cutting_dim:
    :param min:
    :return:
    """
    if min is None:
        min = n.location
    if n is None:
        return min
    current_cutting_dim = depth % len(min)
    if n.location[cutting_dim] < min[cutting_dim]:
        min = n.location
    if cutting_dim == current_cutting_dim:
        return findmin(n.left, depth + 1, cutting_dim, min)
    else:
        leftmin = findmin(n.left, depth + 1, cutting_dim, min)
        rightmin = findmin(n.right, depth + 1, cutting_dim, min)
        if leftmin[cutting_dim] > rightmin[cutting_dim]:
            return rightmin
        else:
            return leftmin


def insert(n, point, depth):
    """
    1.3）新增节点
    从根节点出发，若待插入节点在当前节点切分维度的坐标值小于当前节点在该维度的坐标值时，
    在其左子树插入；若大于等于当前节点在该维度的坐标值时，在其右子树插入。递归遍历，直至叶子节点。
    :param n:
    :param point:
    :param depth:
    :return:
    """
    if n is None:
        return Node(point)
    cutting_dim = depth % len(point)
    if point[cutting_dim] < n.location[cutting_dim]:
        if n.left is None:
            n.left = Node(point)
        else:
            insert(n.left, point, depth + 1)
    else:
        if n.right is None:
            n.right = Node(point)
        else:
            insert(n.right, point, depth + 1)


def delete(n, point, depth):
    """
    1.4）删除节点
最简单的方法是将待删节点的所有子节点组成一个新的集合，然后对其进行重新构建。将构建好的子树挂载到被删节点即可。此方法性能不佳，下面考虑优化后的算法。
假设待删节点T的切分维度为x，下面根据待删节点的几类不同情形进行考虑。
a）无子树
本身为叶子节点，直接删除。
b）有右子树
在T.right寻找x切分维度最小的节点p，然后替换被删节点T；递归处理删除节点p。
c）无右子树有左子树
在T.left寻找x切分维度最小的节点p，即p=findmin(T.left, cutting-dim=x)，然后用节点p替换被删节点T；将原T.left作为p.right；递归处理删除节点p。
（之所以未采用findmax(T.left, cutting-dim=x)节点来替换被删节点，是由于原被删节点的左子树节点存在x维度最大值相等的情形，这样就破坏了左子树在x分割维度的坐标需小于其根节点的定义）
    :param n:
    :param point:
    :param depth:
    :return:
    """
    cutting_dim = depth % len(point)
    if n.location == point:
        if n.right is not None:
            n.location = findmin(n.right, depth + 1, cutting_dim, None)
            delete(n.right, n.location, depth + 1)
        elif n.left is not None:
            n.location = findmin(n.left, depth + 1)
            delete(n.left, n.location, depth + 1)
            n.right = n.left
            n.left = None
        else:
            n = None
    else:
        if point[cutting_dim] < n.location[cutting_dim]:
            delete(n.left, point, depth + 1)
        else:
            delete(n.right, point, depth + 1)


def test():
    """Example usage"""
    point_list = [(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)]
    tree = kd_tree(point_list)
    print(tree)


def knn():
    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib.patches import Circle
    from sklearn.neighbors import KDTree
    np.random.seed(0)
    points = np.random.random((500, 2))
    tree = KDTree(points)
    point = points[3]
    # kNN
    dists, indices = tree.query([point], k=4)
    print(dists, indices)
    # query radius
    indices = tree.query_radius([point], r=0.2)
    print(indices)
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.add_patch(Circle(point, 0.2, color='r', fill=False))
    X, Y = [p[0] for p in points], [p[1] for p in points]
    plt.scatter(X, Y)
    plt.scatter([point[0]], [point[1]], c='r')
    plt.show()


if __name__ == '__main__':
    knn()
