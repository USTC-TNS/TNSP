# -*- coding: utf-8 -*-
import TAT


def test_basic_usage_0():
    a = TAT.Z2.S.Tensor(["i", "j"], [[(False, 2)], [(False, 1)]]).range_()
    b = a.split_edge({"i": [("k", [(False, 2)])], "j": []})


def test_basic_usage_1():
    a = TAT.Z2.S.Tensor(["i", "j"], [[(False, 2), (True, 2)], [(False, 1)]]).range_()
    b = a.split_edge({"i": [("k", [(False, 2), (True, 2)])], "j": []})
