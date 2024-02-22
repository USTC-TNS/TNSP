# -*- coding: utf-8 -*-
import TAT


def test_tensor_and_number():
    import numpy as np
    a = TAT.BoseZ2.Z.Tensor(["Left", "Right", "Phy"], [
        [(False, 2), (True, 2)],
        [(False, 2), (True, 2)],
        [(False, 2), (True, 2)],
    ]).range_()
    assert a.storage.size == 32
    s_a = np.arange(32, dtype=float)
    assert all(a.storage == s_a)
    b = a + 1.0
    s_b = s_a + 1.0
    assert all(b.storage == s_b)
    c = 1.0 + a
    s_c = 1.0 + s_a
    assert all(c.storage == s_c)
    d = a - 1.0
    s_d = s_a - 1.0
    assert all(d.storage == s_d)
    e = 1.0 - a
    s_e = 1.0 - s_a
    assert all(e.storage == s_e)
    f = a * 1.5
    s_f = s_a * 1.5
    assert all(f.storage == s_f)
    g = 1.5 * a
    s_g = 1.5 * s_a
    assert all(g.storage == s_g)
    h = a / 1.5
    s_h = s_a / 1.5
    assert all(h.storage == s_h)
    i = 1.5 / (a + 1)
    s_i = 1.5 / (s_a + 1)
    assert all(i.storage == s_i)


def test_tensor_and_tensor():
    import numpy as np
    a = TAT.No.D.Tensor(["Left", "Right"], [3, 4]).range_()
    b = TAT.No.D.Tensor(["Left", "Right"], [3, 4]).range_(0, 0.1)
    s_a = np.zeros(12)
    s_b = np.zeros(12)
    s_a[0] = s_b[0] = 0
    for i in range(1, 12):
        s_a[i] = s_a[i - 1] + 1
        s_b[i] = s_b[i - 1] + 0.1
    assert a.storage.size == s_a.size
    assert b.storage.size == s_b.size
    assert all(a.storage == s_a)
    assert all(b.storage == s_b)
    c = a + b
    s_c = s_a + s_b
    assert all(c.storage == s_c)
    d = a - b
    s_d = s_a - s_b
    assert all(d.storage == s_d)
    e = a * b
    s_e = s_a * s_b
    assert all(e.storage == s_e)
    f = a / (b + 1)
    s_f = s_a / (s_b + 1)
    assert all(f.storage == s_f)


def test_tensor_and_number_inplace():
    import numpy as np
    a = TAT.BoseZ2.Z.Tensor(["Left", "Right", "Phy"], [
        [(False, 2), (True, 2)],
        [(False, 2), (True, 2)],
        [(False, 2), (True, 2)],
    ]).range_()
    s_a = np.arange(32, dtype=float)
    assert all(a.storage == s_a)
    a += 1.5
    s_a += 1.5
    assert all(a.storage == s_a)
    a *= 0.9
    s_a *= 0.9
    assert all(a.storage == s_a)
    a -= 0.1
    s_a -= 0.1
    assert all(a.storage == s_a)
    a /= 2.0
    s_a /= 2.0
    assert all(a.storage == s_a)


def test_tensor_and_tensor_inplace():
    import numpy as np
    a = TAT.No.D.Tensor(["Left", "Right"], [3, 4]).range_()
    b = TAT.No.D.Tensor(["Left", "Right"], [3, 4]).range_(0, 0.1)
    s_a = np.zeros(12)
    s_b = np.zeros(12)
    s_a[0] = s_b[0] = 0
    for i in range(1, 12):
        s_a[i] = s_a[i - 1] + 1
        s_b[i] = s_b[i - 1] + 0.1
    assert a.storage.size == s_a.size
    assert b.storage.size == s_b.size
    assert all(a.storage == s_a)
    assert all(b.storage == s_b)
    a += b
    s_a += s_b
    assert all(a.storage == s_a)
    a *= b
    s_a *= s_b
    assert all(a.storage == s_a)
    a -= b
    s_a -= s_b
    assert all(a.storage == s_a)
    a /= b + 1
    s_a /= s_b + 1
    assert all(a.storage == s_a)
