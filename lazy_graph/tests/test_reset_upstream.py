# -*- coding: utf-8 -*-
from lazy import Root, Node

calculate_count = 0


def add(a, b):
    global calculate_count
    calculate_count += 1
    return a + b


def test_reset_upstream():
    a = Root(1)
    b = Root(2)
    c = Node(add, a, b)
    assert calculate_count == 0
    assert c() == 3
    assert calculate_count == 1
    assert c() == 3
    assert calculate_count == 1
    a.reset(4)
    assert c() == 6
    assert calculate_count == 2
