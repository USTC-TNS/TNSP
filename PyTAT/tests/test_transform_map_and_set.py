import TAT


def test_transform():
    t = TAT.No.D.Tensor(["i", "j"], [2, 3]).range_()
    assert all(t.storage == [0, 1, 2, 3, 4, 5])
    t.transform_(lambda x: x + 1)
    assert all(t.storage == [1, 2, 3, 4, 5, 6])


def test_map():
    t = TAT.No.D.Tensor(["i", "j"], [2, 3]).range_()
    assert all(t.storage == [0, 1, 2, 3, 4, 5])
    z = t.map(lambda x: x + 1)
    assert all(z.storage == [1, 2, 3, 4, 5, 6])
    assert all(t.storage == [0, 1, 2, 3, 4, 5])


def test_copy():
    t = TAT.No.D.Tensor(["i", "j"], [2, 3]).range_()
    assert all(t.storage == [0, 1, 2, 3, 4, 5])
    s = t.copy()
    assert all(s.storage == [0, 1, 2, 3, 4, 5])


def test_set():
    s = iter([6, 2, 8, 3, 7, 1])
    t = TAT.No.D.Tensor(["i", "j"], [2, 3]).set_(lambda: next(s))
    assert all(t.storage == [6, 2, 8, 3, 7, 1])


def test_zero():
    t = TAT.No.D.Tensor(["i", "j"], [2, 3]).zero_()
    assert all(t.storage == [0, 0, 0, 0, 0, 0])


def test_range():
    t = TAT.No.D.Tensor(["i", "j"], [2, 3]).range_(3, 2)
    assert all(t.storage == [3, 5, 7, 9, 11, 13])
