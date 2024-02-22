# -*- coding: utf-8 -*-
import TAT


def check_unitary(tensor, name, name_prime, fermi):
    pairs = set()
    for n in tensor.names:
        if n != name:
            pairs.add((n, n))
    conjugated = tensor.conjugate(True).edge_rename({name: name_prime})
    product = tensor.contract(conjugated, pairs)
    identity = product.same_shape().identity_({(name, name_prime)})
    if fermi:
        product.transform_(abs)
        identity.transform_(abs)
    assert (product - identity).norm_max() < 1e-6


def test_no_symmetry():
    a = TAT.No.D.Tensor(["A", "B", "C", "D"], [2, 3, 4, 5]).range_()
    u, s, v = a.svd({"C", "A"}, "E", "F", "U", "V")
    check_unitary(u, "E", "E'", False)
    check_unitary(v, "F", "F'", False)
    b = v.contract(s, {("F", "V")}).contract(u, {("U", "E")})
    assert (a - b).norm_max() < 1e-6


def test_no_symmetry_cut():
    a = TAT.No.D.Tensor(["A", "B", "C", "D"], [5, 4, 3, 2]).range_()
    u, s, v = a.svd({"C", "A"}, "E", "F", "U", "V", 2)
    check_unitary(u, "E", "E'", False)
    check_unitary(v, "F", "F'", False)
    b = v.contract(s, {("F", "V")}).contract(u, {("U", "E")})
    assert (a - b).norm_max() < 1e-6


def test_u1_symmetry():
    a = TAT.BoseU1.D.Tensor(
        ["A", "B", "C", "D"],
        [
            ([(-1, 1), (0, 1), (-2, 1)], True),
            ([(0, 1), (1, 2)], False),
            ([(0, 2), (1, 2)], False),
            ([(-2, 2), (-1, 1), (0, 2)], True),
        ],
    ).range_()
    u, s, v = a.svd({"C", "A"}, "E", "F", "U", "V")
    check_unitary(u, "E", "E'", False)
    check_unitary(v, "F", "F'", False)
    b = v.contract(s, {("F", "V")}).contract(u, {("U", "E")})
    assert (a - b).norm_max() < 1e-6


def test_u1_symmetry_cut():
    a = TAT.BoseU1.D.Tensor(
        ["A", "B", "C", "D"],
        [
            ([(-1, 1), (0, 1), (-2, 1)], True),
            ([(0, 1), (1, 2)], False),
            ([(0, 2), (1, 2)], False),
            ([(-2, 2), (-1, 1), (0, 2)], True),
        ],
    ).range_()
    u, s, v = a.svd({"C", "A"}, "E", "F", "U", "V", 7)
    check_unitary(u, "E", "E'", False)
    check_unitary(v, "F", "F'", False)
    b = v.contract(s, {("F", "V")}).contract(u, {("U", "E")})
    assert (a - b).norm_max() < 1e-6


def test_fermi_symmetry():
    a = TAT.FermiU1.D.Tensor(
        ["A", "B", "C", "D"],
        [
            ([(-1, 1), (0, 1), (-2, 1)], True),
            ([(0, 1), (1, 2)], False),
            ([(0, 2), (1, 2)], False),
            ([(-2, 2), (-1, 1), (0, 2)], True),
        ],
    ).range_()
    u, s, v = a.svd({"C", "A"}, "E", "F", "U", "V")
    check_unitary(u, "E", "E'", True)
    check_unitary(v, "F", "F'", True)
    b = v.contract(s, {("F", "V")}).contract(u, {("U", "E")})
    assert (a - b).norm_max() < 1e-6


def test_fermi_symmetry_cut():
    a = TAT.FermiU1.D.Tensor(
        ["A", "B", "C", "D"],
        [
            ([(-1, 1), (0, 1), (-2, 1)], True),
            ([(0, 1), (1, 2)], False),
            ([(0, 2), (1, 2)], False),
            ([(-2, 2), (-1, 1), (0, 2)], True),
        ],
    ).range_()
    u, s, v = a.svd({"B", "D"}, "E", "F", "U", "V", 8)
    check_unitary(u, "E", "E'", True)
    check_unitary(v, "F", "F'", True)
    b = v.contract(s, {("F", "V")}).contract(u, {("U", "E")})
    assert (a - b).norm_max() < 1e-6


def test_no_symmetry_cut_too_small():
    a = TAT.No.D.Tensor(["A", "B"], [2, 2]).zero_()
    a[{"A": 0, "B": 0}] = 1
    u, s, v = a.svd({"B"}, "E", "F", "U", "V", 8)
    check_unitary(u, "E", "E'", False)
    check_unitary(v, "F", "F'", False)
    b = v.contract(s, {("F", "V")}).contract(u, {("U", "E")})
    assert (a - b).norm_max() < 1e-6
    assert s.storage.size == 1


def test_fermi_symmetry_cut_too_small():
    a = TAT.FermiU1.D.Tensor(
        ["A", "B"],
        [
            [(0, 1), (+1, 1)],
            [(-1, 1), (0, 1)],
        ],
    ).range_(0, 1)
    u, s, v = a.svd({"B"}, "E", "F", "U", "V", 8)
    check_unitary(u, "E", "E'", True)
    check_unitary(v, "F", "F'", True)
    b = v.contract(s, {("F", "V")}).contract(u, {("U", "E")})
    assert (a - b).norm_max() < 1e-6
    assert s.storage.size == 1
