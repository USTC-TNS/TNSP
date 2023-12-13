# -*- coding: utf-8 -*-
import TAT


def test_seed():
    TAT.random.seed(233)
    x = TAT.random.uniform_int(0, 100)()
    TAT.random.seed(233)
    y = TAT.random.uniform_int(0, 100)()
    assert x == y


def test_uniform_int():
    rg = TAT.random.uniform_int(0, 100)
    assert all(0 <= rg() <= 100 for _ in range(10000))


def test_uniform_real():
    rg = TAT.random.uniform_real(0, 100)
    assert all(0 <= rg() <= 100 for _ in range(10000))
