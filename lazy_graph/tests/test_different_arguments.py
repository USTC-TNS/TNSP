# -*- coding: utf-8 -*-
from lazy import Root, Node

calculate_count = 0


def add(a, b, c, d):
    global calculate_count
    calculate_count += 1
    return a + b + c + d


def test_different_arguments():
    a = Root(1)
    c = Root(3)
    m = Node(add, a, 2, c=c, d=4)
    assert m() == 10
