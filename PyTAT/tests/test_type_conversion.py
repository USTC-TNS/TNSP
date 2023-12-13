# -*- coding: utf-8 -*-
import TAT


def test_dummy():
    import numpy as np
    a = TAT.No.D.Tensor(["Left", "Right"], [3, 4]).range_(2)
    b = a.to("float64")
    assert np.shares_memory(a.storage, b.storage)
    c = a.to("float32")
    assert not np.shares_memory(a.storage, c.storage)


def test_inside_float():
    a = TAT.No.D.Tensor(["Left", "Right"], [3, 4]).range_(2)
    b = a.to("float32")
    assert all(a.storage == [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    assert all(b.storage == [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])


def test_inside_complex():
    a = TAT.No.Z.Tensor(["Left", "Right"], [3, 4]).range_(2)
    b = a.to("complex64")
    assert all(a.storage == [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    assert all(b.storage == [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])


def test_complex_to_float():
    a = TAT.No.Z.Tensor(["Left", "Right"], [3, 4]).range_(2)
    b = a.to("float32")
    assert all(a.storage == [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    assert all(b.storage == [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])


def test_float_to_complex():
    a = TAT.No.D.Tensor(["Left", "Right"], [3, 4]).range_(2)
    b = a.to("complex64")
    assert all(a.storage == [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    assert all(b.storage == [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
