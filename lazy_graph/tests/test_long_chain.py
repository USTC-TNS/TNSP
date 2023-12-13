# -*- coding: utf-8 -*-
from lazy import Root, Node

calculate_count = 0


def add1(a):
    global calculate_count
    calculate_count += 1
    return a + 1


def test_long_chain():
    a = Root(1)
    b = Node(add1, a)
    c = Node(add1, b)
    assert calculate_count == 0
    assert c() == 3
    assert calculate_count == 2
