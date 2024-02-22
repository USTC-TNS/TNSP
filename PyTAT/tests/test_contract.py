# -*- coding: utf-8 -*-
import TAT


def test_no_symmetry_example_0():
    a = TAT.No.D.Tensor(["A", "B"], [2, 2]).range_()
    b = TAT.No.D.Tensor(["C", "D"], [2, 2]).range_()

    c = a.contract(b, {("A", "C")})
    c_expect = TAT.No.D.Tensor(["B", "D"], [2, 2])
    c_expect.storage = [4, 6, 6, 10]
    assert (c - c_expect).norm_max() == 0

    d = a.contract(b, {("A", "D")})
    d_expect = TAT.No.D.Tensor(["B", "C"], [2, 2])
    d_expect.storage = [2, 6, 3, 11]
    assert (d - d_expect).norm_max() == 0

    e = a.contract(b, {("B", "C")})
    e_expect = TAT.No.D.Tensor(["A", "D"], [2, 2])
    e_expect.storage = [2, 3, 6, 11]
    assert (e - e_expect).norm_max() == 0

    f = a.contract(b, {("B", "D")})
    f_expect = TAT.No.D.Tensor(["A", "C"], [2, 2])
    f_expect.storage = [1, 3, 3, 13]
    assert (f - f_expect).norm_max() == 0


def test_no_symmetry_example_1():
    x = TAT.No.D.Tensor(["A", "B", "C", "D"], [1, 2, 3, 4]).range_()
    y = TAT.No.D.Tensor(["E", "F", "G", "H"], [3, 1, 2, 4]).range_()
    a = x.contract(y, {("B", "G"), ("D", "H")})
    b = TAT.No.D.Tensor(["A", "C", "E", "F"], [1, 3, 3, 1])
    b.storage = [316, 796, 1276, 428, 1164, 1900, 540, 1532, 2524]
    assert (a - b).norm_max() == 0


def test_u1_symmetry_example_0():
    Tensor = TAT.BoseU1.D.Tensor
    edge1 = [(-1, 2), (0, 2), (+1, 2)]
    edge2 = [(+1, 2), (0, 2), (-1, 2)]
    a = Tensor(["a", "b", "c", "d"], [edge1, edge2, edge1, edge2]).range_()
    b = Tensor(["e", "f", "g", "h"], [edge1, edge2, edge1, edge2]).range_()
    for plan in [
        {("a", "f"), ("b", "e")},
        {("a", "f"), ("b", "g")},
        {("a", "h"), ("b", "e")},
        {("a", "h"), ("b", "g")},
        {("a", "f"), ("c", "h")},
        {("a", "h"), ("c", "f")},
        {("a", "f"), ("d", "e")},
        {("a", "f"), ("d", "g")},
        {("a", "h"), ("d", "e")},
        {("a", "h"), ("d", "g")},
        {("c", "f"), ("b", "e")},
        {("c", "f"), ("b", "g")},
        {("c", "h"), ("b", "e")},
        {("c", "h"), ("b", "g")},
        {("b", "e"), ("d", "g")},
        {("b", "g"), ("d", "e")},
        {("c", "f"), ("d", "e")},
        {("c", "f"), ("d", "g")},
        {("c", "h"), ("d", "e")},
        {("c", "h"), ("d", "g")},
    ]:
        c = a.contract(b, plan).clear_symmetry()
        d = a.clear_symmetry().contract(b.clear_symmetry(), plan)
        assert (c - d).norm_max() == 0


def test_fermi_symmetry_example_0():
    FermiTensor = TAT.FermiU1.D.Tensor
    fermi_edge1 = [(-1, 2), (0, 2), (+1, 2)]
    fermi_edge2 = [(+1, 2), (0, 2), (-1, 2)]
    fermi_a = FermiTensor(["a", "b", "c", "d"], [
        (fermi_edge1, True),
        fermi_edge2,
        (fermi_edge1, True),
        (fermi_edge2, True),
    ]).range_()
    fermi_b = FermiTensor(["e", "f", "g", "h"], [
        fermi_edge1,
        fermi_edge2,
        (fermi_edge1, True),
        fermi_edge2,
    ]).range_()
    fermi_c = fermi_a.contract(fermi_b, {("d", "e"), ("c", "f")})
    fermi_d = fermi_b.contract(fermi_a, {("e", "d"), ("f", "c")})
    assert (fermi_c - fermi_d).norm_max() == 0

    U1Tensor = TAT.BoseU1.D.Tensor
    u1_edge1 = [(-1, 2), (0, 2), (+1, 2)]
    u1_edge2 = [(+1, 2), (0, 2), (-1, 2)]
    u1_a = U1Tensor(["a", "b", "c", "d"], [
        u1_edge1,
        u1_edge2,
        u1_edge1,
        u1_edge2,
    ]).range_()
    u1_b = U1Tensor(["e", "f", "g", "h"], [
        u1_edge1,
        u1_edge2,
        u1_edge1,
        u1_edge2,
    ]).range_()
    u1_c = u1_a.contract(u1_b, {("d", "e"), ("c", "f")})
    u1_d = u1_b.contract(u1_a, {("e", "d"), ("f", "c")})
    assert (u1_c - u1_d).norm_max() == 0

    assert all(fermi_a.storage == u1_a.storage)
    assert all(fermi_b.storage == u1_b.storage)
    assert all(fermi_c.storage == u1_c.storage)


def test_contract_with_split_and_merge():
    Tensor = TAT.FermiU1.D.Tensor
    edge1 = ([(-1, 2), (0, 2), (+1, 2)], False)
    edge2 = ([(+1, 2), (0, 2), (-1, 2)], True)
    a = Tensor(["a", "b", "c", "d"], [edge1, edge2, edge1, edge2]).range_()
    b = Tensor(["e", "f", "g", "h"], [edge1, edge2, edge1, edge2]).range_()
    c = a.contract(b, {("a", "f"), ("b", "g"), ("c", "h")})

    a_merged = a.merge_edge({"m": ["b", "a"]}, False)
    b_merged = b.merge_edge({"m": ["g", "f"]}, True)
    c_merged = a_merged.contract(b_merged, {("m", "m"), ("c", "h")})

    assert (c - c_merged).norm_max() == 0


def test_contract_with_reverse_0():
    a = TAT.FermiZ2.D.Tensor(
        ["i", "j"],
        [([(False, 2), (True, 2)], False), ([(False, 2), (True, 2)], True)],
    ).range_()
    b = TAT.FermiZ2.D.Tensor(
        ["i", "j"],
        [([(False, 2), (True, 2)], False), ([(False, 2), (True, 2)], True)],
    ).range_().transpose(["j", "i"])
    c = a.contract(b, {("j", "i")})

    a_reversed = a.reverse_edge({"j"}, False)
    b_reversed = b.reverse_edge({"i"}, True)
    c_reversed = a_reversed.contract(b_reversed, {("j", "i")})

    assert (c - c_reversed).norm_max() == 0


def test_contract_with_reverse_1():
    Tensor = TAT.FermiU1.D.Tensor
    edge1 = ([(-1, 2), (0, 2), (+1, 2)], False)
    edge2 = ([(+1, 2), (0, 2), (-1, 2)], True)
    a = Tensor(["a", "b", "c", "d"], [edge1, edge2, edge1, edge2]).range_()
    b = Tensor(["e", "f", "g", "h"], [edge1, edge2, edge1, edge2]).range_()
    c = a.contract(b, {("a", "f"), ("b", "g"), ("c", "h")})

    a_reversed = a.reverse_edge({"b", "a"}, False)
    b_reversed = b.reverse_edge({"g", "f"}, True)
    c_reversed = a_reversed.contract(b_reversed, {("a", "f"), ("b", "g"), ("c", "h")})

    assert (c - c_reversed).norm_max() == 0


def test_fuse():
    fuse_d = 3
    common_d = 4
    a = TAT.No.D.Tensor(["A", "B", "C"], [fuse_d, common_d, 5]).range_()
    b = TAT.No.D.Tensor(["A", "B", "D"], [fuse_d, common_d, 7]).range_()
    c = a.contract(b, {("B", "B")}, {"A"})
    for i in range(fuse_d):
        hat = TAT.No.D.Tensor(["A"], [fuse_d]).zero_()
        hat.storage[i] = 1
        a0 = a.contract(hat, {("A", "A")})
        b0 = b.contract(hat, {("A", "A")})
        c0 = c.contract(hat, {("A", "A")})
        assert (a0.contract(b0, {("B", "B")}) - c0).norm_max() == 0


def test_corner_no_symmetry_0k():
    a = TAT.No.D.Tensor(["A", "B"], [2, 0]).range_()
    b = TAT.No.D.Tensor(["C", "D"], [0, 2]).range_()
    c = a.contract(b, {("B", "C")})
    assert c.storage.size == 4
    assert c.norm_max() == 0


def test_corner_z2_symmetry_0k():
    a = TAT.BoseZ2.D.Tensor(["A", "B"], [[(False, 2)], [(False, 0)]]).range_()
    b = TAT.BoseZ2.D.Tensor(["C", "D"], [[(False, 0)], [(False, 2)]]).range_()
    c = a.contract(b, {("B", "C")})
    assert c.storage.size == 4
    assert c.norm_max() == 0


def test_corner_z2_symmetry_not_match_missing_left():
    a = TAT.BoseZ2.D.Tensor(["A", "B"], [[(True, 2)], [(False, 0)]]).range_()
    b = TAT.BoseZ2.D.Tensor(["C", "D"], [[(False, 0)], [(False, 2)]]).range_()
    c = a.contract(b, {("B", "C")})
    assert c.storage.size == 0


def test_corner_z2_symmetry_not_match_missing_right():
    a = TAT.BoseZ2.D.Tensor(["A", "B"], [[(False, 2)], [(False, 0)]]).range_()
    b = TAT.BoseZ2.D.Tensor(["C", "D"], [[(False, 0)], [(True, 2)]]).range_()
    c = a.contract(b, {("B", "C")})
    assert c.storage.size == 0


def test_corner_z2_symmetry_not_match_missing_middle():
    a = TAT.BoseZ2.D.Tensor(["A", "B"], [[(False, 2)], [(True, 0)]]).range_()
    b = TAT.BoseZ2.D.Tensor(["C", "D"], [[(True, 0)], [(False, 2)]]).range_()
    c = a.contract(b, {("B", "C")})
    assert c.storage.size == 4
    assert c.norm_max() == 0
