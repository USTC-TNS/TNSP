# -*- coding: utf-8 -*-
import TAT


def test_no_symmetry_basic_0():
    a = TAT.No.D.Tensor(["Left", "Right"], [2, 3]).range_()
    b = a.transpose(["Right", "Left"])
    assert all(a.storage == [0, 1, 2, 3, 4, 5])
    assert all(b.storage == [0, 3, 1, 4, 2, 5])


def test_no_symmetry_basic_1():
    a = TAT.No.D.Tensor(["i", "j", "k"], [2, 3, 4]).range_()
    for result_edge in [
        ["i", "j", "k"],
        ["i", "k", "j"],
        ["j", "k", "i"],
        ["j", "i", "k"],
        ["k", "i", "j"],
        ["k", "j", "i"],
    ]:
        b = a.transpose(result_edge)
        for i in range(2):
            for j in range(3):
                for k in range(4):
                    assert a[{"i": i, "j": j, "k": k}] == b[{"i": i, "j": j, "k": k}]


def test_no_symmetry_high_dimension():
    a = TAT.No.D.Tensor(["i", "j", "k", "l", "m", "n"], [2, 2, 2, 2, 2, 2]).range_()
    b = a.transpose(["l", "j", "i", "n", "k", "m"])
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    for m in range(2):
                        for n in range(2):
                            assert a[{
                                "i": i,
                                "j": j,
                                "k": k,
                                "l": l,
                                "m": m,
                                "n": n
                            }] == b[{
                                "i": i,
                                "j": j,
                                "k": k,
                                "l": l,
                                "m": m,
                                "n": n
                            }]


def test_z2_symmetry_high_dimension():
    edge = TAT.Z2.Edge([(False, 2), (True, 2)])
    a = TAT.Z2.D.Tensor(["i", "j", "k", "l", "m", "n"], [edge, edge, edge, edge, edge, edge]).range_()
    b = a.transpose(["l", "j", "i", "n", "k", "m"])
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    for m in range(2):
                        for n in range(2):
                            p_i = i & 2 != 0
                            p_j = j & 2 != 0
                            p_k = k & 2 != 0
                            p_l = l & 2 != 0
                            p_m = m & 2 != 0
                            p_n = n & 2 != 0
                            if p_i ^ p_j ^ p_k ^ p_l ^ p_m ^ p_n:
                                continue
                            assert a[{
                                "i": i,
                                "j": j,
                                "k": k,
                                "l": l,
                                "m": m,
                                "n": n
                            }] == b[{
                                "i": i,
                                "j": j,
                                "k": k,
                                "l": l,
                                "m": m,
                                "n": n
                            }]


def test_parity_symmetry_high_dimension():
    edge = TAT.Parity.Edge([(False, 2), (True, 2)])
    a = TAT.Parity.D.Tensor(["i", "j", "k", "l", "m", "n"], [edge, edge, edge, edge, edge, edge]).range_()
    b = a.transpose(["l", "j", "i", "n", "k", "m"])
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    for m in range(2):
                        for n in range(2):
                            p_i = i & 2 != 0
                            p_j = j & 2 != 0
                            p_k = k & 2 != 0
                            p_l = l & 2 != 0
                            p_m = m & 2 != 0
                            p_n = n & 2 != 0
                            if p_i ^ p_j ^ p_k ^ p_l ^ p_m ^ p_n:
                                continue
                            parity = (p_l and (p_i ^ p_j ^ p_k)) ^ (p_j and p_i) ^ (p_n and (p_k ^ p_m))
                            if parity:
                                assert -a[{
                                    "i": i,
                                    "j": j,
                                    "k": k,
                                    "l": l,
                                    "m": m,
                                    "n": n
                                }] == b[{
                                    "i": i,
                                    "j": j,
                                    "k": k,
                                    "l": l,
                                    "m": m,
                                    "n": n
                                }]
                            else:
                                assert a[{
                                    "i": i,
                                    "j": j,
                                    "k": k,
                                    "l": l,
                                    "m": m,
                                    "n": n
                                }] == b[{
                                    "i": i,
                                    "j": j,
                                    "k": k,
                                    "l": l,
                                    "m": m,
                                    "n": n
                                }]
