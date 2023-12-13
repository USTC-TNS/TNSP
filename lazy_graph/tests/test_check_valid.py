# -*- coding: utf-8 -*-
from lazy import Root, Node

calculate_count = 0


def add(a, b):
    global calculate_count
    calculate_count += 1
    return a + b


def test_check_valid():
    a = Root(1)
    b = Root(2)
    c = Node(add, a, b)
    assert bool(c) is False
    assert c() == 3
    assert bool(c) is True
