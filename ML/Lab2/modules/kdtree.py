from __future__ import print_function

import math
import operator
from functools import wraps
from collections import deque

COMPARE_CHILD = {
    0: (operator.le, operator.sub),
    1: (operator.ge, operator.add),
}


def require_axis(f):

    @wraps(f)
    def _wrapper(self, *args, **kwargs):
        if None in (self.axis, self.sel_axis):
            raise ValueError('%(func_name) requires the node %(node)s '
                             'to have an axis and a sel_axis function' %
                             dict(func_name=f.__name__, node=repr(self)))

        return f(self, *args, **kwargs)

    return _wrapper


class KDNode:
    def __init__(self, data=None, parent=None, left=None, right=None,
                 axis=None, sel_axis=None, dimensions=None):
        self.data = data
        self.parent = parent
        self.left = left
        self.right = right
        self.axis = axis
        self.sel_axis = sel_axis
        self.dimensions = dimensions

    def is_leaf(self):
        return (not self.data) or (all(not bool(c) for c, p in self.children))

    def children(self):
        if self.left and self.left.data is not None:
            yield self.left, 0
        if self.right and self.right.data is not None:
            yield self.right, 1

    def height(self):
        min_height = int(bool(self))
        return max([min_height] + [c.height() + 1 for c, p in self.children()])

    def get_child_pos(self, child):
        for c, p in self.children:
            if child == c:
                return p

    @require_axis
    def add(self, point):
        current = self
        while True:
            check_dimensionality(point, dimensions=current.dimensions)

            if current.data is None:
                current.data = point
                return current
            if (point.get(current.axis, 0.) <
                    current.data.get(current.axis, 0.)):
                if current.left is None:
                    current.left = current.create_subnode(point)
                    return current.left
                else:
                    current = current.left
            else:
                if current.right is None:
                    current.right = current.create_subnode(point)
                    return current.right
                else:
                    current = current.right

    @require_axis
    def create_subnode(self, data):
        return self.__class__(data, parent=self,
                              axis=self.sel_axis(self.axis),
                              sel_axis=self.sel_axis,
                              dimensions=self.dimensions)

    def should_remove(self, point, node):
        if not self.data == point:
            return False

        return (node is None) or (node is self)

    @require_axis
    def remove(self, point, node=None):
        if not self:
            return


        if self.should_remove(point, node):
            return self._remove(point)

        if self.left and self.left.should_remove(point, node):
            self.left = self.left._remove(point)
        elif self.right and self.right.should_remove(point, node):
            self.right = self.right._remove(point)

        if point.get(self.axis, 0.) <= self.data.get(self.axis, 0.):
            if self.left:
                self.left = self.left.remove(point, node)
        if point.get(self.axis, 0.) >= self.data.get(self.axis, 0.):
            if self.right:
                self.right = self.right.remove(point, node)
        return self

    @require_axis
    def find_replacement(self):
        if self.right:
            child, parent = self.right.extreme_child(min, self.axis)
        else:
            child, parent = self.left.extreme_child(max, self.axis)

        return (child, parent if parent is not None else self)

    def extreme_child(self, sel_func, axis):

        max_key = lambda child_parent: child_parent[0].data.get(axis, 0.)


        me = [(self, None)] if self else []

        child_max = [c.extreme_child(sel_func, axis) for c, _ in self.children]
        # insert self for unknown parents
        child_max = [(c, p if p is not None else self) for c, p in child_max]

        candidates = me + child_max

        if not candidates:
            return None, None

        return sel_func(candidates, key=max_key)

    @require_axis
    def _remove(self, point):
        if self.is_leaf:
            self.data = None
            return self

        root, max_p = self.find_replacement()


        tmp_l, tmp_r = self.left, self.right
        self.left, self.right = root.left, root.right
        root.left = tmp_l if tmp_l is not root else self
        root.right = tmp_r if tmp_r is not root else self

        self.axis, root.axis = root.axis, self.axis


        if max_p is not self:
            pos = max_p.get_child_pos(root)
            max_p.set_child(pos, self)
            max_p.remove(point, self)
        else:
            root.remove(point, self)
        return root

    def axis_dist(self, point, axis):
        return math.pow(self.data.get(axis, 0.) - point.get(axis, 0.), 2)

    def dist(self, point):
        r = range(len(self.data))
        return sum([self.axis_dist(point, i) for i in r])

    def _search_node(self, point, k, results, examined, get_dist):
        examined.add(self)


        if not results:

            bestNode = None
            bestDist = float('inf')
        else:

            bestNode, bestDist = sorted(
                results.items(), key=lambda n_d: n_d[1], reverse=False)[0]

        nodesChanged = False
        nodeDist = get_dist(self)
        if nodeDist < bestDist:
            if len(results) == k and bestNode:
                maxNode, maxDist = sorted(
                    results.items(), key=lambda n: n[1], reverse=True)[0]
                results.pop(maxNode)

            results[self] = nodeDist
            nodesChanged = True
        elif nodeDist == bestDist:
            results[self] = nodeDist
            nodesChanged = True

        elif len(results) < k:
            results[self] = nodeDist
            nodesChanged = True


        if nodesChanged:
            bestNode, bestDist = sorted(
                results.items(), key=lambda n: n[1], reverse=False)[0]

        for child, pos in self.children():
            if child in examined:
                continue

            examined.add(child)
            compare, combine = COMPARE_CHILD[pos]

            nodePoint = self.data.get(self.axis, 0.)
            pointPlusDist = combine(point.get(self.axis, 0.), bestDist)
            lineIntersects = compare(pointPlusDist, nodePoint)

            if lineIntersects:
                child._search_node(point, k, results, examined, get_dist)

    def search_knn(self, point, k, dist=None):
        prev = None
        current = self

        if dist is None:
            get_dist = lambda n: n.dist(point)
        else:
            get_dist = lambda n: dist(n.data, point)


        while current:
            if (point.get(current.axis, 0.) <
                    current.data.get(current.axis, 0.)):

                prev = current
                current = current.left
            else:

                prev = current
                current = current.right

        if not prev:
            return []

        examined = set()
        results = {}


        current = prev
        while current:
            current._search_node(point, k, results, examined, get_dist)
            current = current.parent

        return sorted(results.items(), key=lambda a: a[1])

    def __nonzero__(self):
        return self.data is not None

    def __repr__(self):
        return "<%(cls)s - %(data)s>" % dict(cls=self.__class__.__name__,
                                             data=repr(self.data))

    __bool__ = __nonzero__

    def __eq__(self, other):
        if isinstance(other, tuple):
            return self.data == other
        else:
            return self.data == other.data

    def __hash__(self):
        return id(self)


def create(point_list, dimensions, axis=0, sel_axis=None, parent=None):
    if not point_list and not dimensions:
        raise ValueError('either point_list or dimensions must be provided')

    elif point_list:
        dimensions = check_dimensionality(point_list, dimensions)


    sel_axis = sel_axis or (lambda prev_axis: (prev_axis + 1) % dimensions)

    if not point_list:
        return KDNode(sel_axis=sel_axis, axis=axis, dimensions=dimensions)


    point_list.sort(key=lambda point: point.get(axis, 0.))
    median = len(point_list) // 2

    loc = point_list[median]
    root = KDNode(loc, parent, left=None, right=None,
                  axis=axis, sel_axis=sel_axis)
    root.left = create(point_list[:median],
                       dimensions, sel_axis(axis), parent=root)
    root.right = create(point_list[median + 1:],
                        dimensions, sel_axis(axis), parent=root)
    return root


def check_dimensionality(point_list, dimensions):

    dimensions = dimensions
    for p in point_list:
        if max(p.keys()) > dimensions:
            raise ValueError(
                'All Points in point_list must have the same dimensionality')

    return dimensions

