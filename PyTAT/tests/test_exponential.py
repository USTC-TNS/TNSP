# -*- coding: utf-8 -*-
import TAT


def reference_exponential(tensor, pairs, step):
    result = tensor.same_shape().identity_(pairs)
    power = result
    for i in range(1, step):
        power = power.contract(tensor, pairs) / i
        result += power
    return result


def test_no_symmetry():
    A = TAT.No.D.Tensor(["i", "j"], [3, 3]).range_()
    pairs = {("i", "j")}
    expA = A.exponential(pairs, 10)
    expA_r = reference_exponential(A, pairs, 100)
    assert (expA - expA_r).norm_max() < 1e-8


def test_u1_symmetry():
    A = TAT.BoseU1.D.Tensor(
        ["i", "j", "k", "l"],
        [
            [(-1, 2), (0, 2), (+1, 2)],
            [(+1, 2), (0, 2), (-1, 2)],
            [(+1, 2), (0, 2), (-1, 2)],
            [(-1, 2), (0, 2), (+1, 2)],
        ],
    ).range_()
    A /= A.norm_max()
    pairs = {("i", "k"), ("l", "j")}
    expA = A.exponential(pairs, 10)
    expA_r = reference_exponential(A, pairs, 100)
    assert (expA - expA_r).norm_max() < 1e-8


def test_fermi_symmetry():
    A = TAT.FermiU1.D.Tensor(
        ["i", "j", "k", "l"],
        [
            ([(-1, 2), (0, 2), (+1, 2)], True),
            ([(+1, 2), (0, 2), (-1, 2)], True),
            ([(+1, 2), (0, 2), (-1, 2)], False),
            ([(-1, 2), (0, 2), (+1, 2)], False),
        ],
    ).range_()
    A /= A.norm_max()
    pairs = {("i", "k"), ("l", "j")}
    expA = A.exponential(pairs, 10)
    expA_r = reference_exponential(A, pairs, 100)
    assert (expA - expA_r).norm_max() < 1e-8
