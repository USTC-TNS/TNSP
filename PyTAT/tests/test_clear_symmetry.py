# -*- coding: utf-8 -*-
import TAT


def test_bose_mode():
    a = TAT.U1.S.Tensor(["i", "j"], [
        [(0, 2), (1, 3), (2, 5)],
        [(0, 5), (-1, 4), (-2, 2)],
    ]).range_()
    b = a.clear_symmetry()
    for sym in range(3):
        dim_i = a.edges[0].dimension_by_symmetry(+sym)
        dim_j = a.edges[1].dimension_by_symmetry(-sym)
        for i in range(dim_i):
            for j in range(dim_j):
                index_i = a.edges[0].index_by_point((+sym, i))
                index_j = a.edges[1].index_by_point((-sym, j))
                assert a[{"i": index_i, "j": index_j}] == b[{"i": index_i, "j": index_j}]


def test_fermi_mode():
    a = TAT.Fermi.S.Tensor(["i", "j"], [
        [(0, 2), (1, 3), (2, 5)],
        [(0, 5), (-1, 4), (-2, 2)],
    ]).range_()
    b = a.clear_symmetry()
    for sym in range(3):
        dim_i = a.edges[0].dimension_by_symmetry(+sym)
        dim_j = a.edges[1].dimension_by_symmetry(-sym)
        for i in range(dim_i):
            for j in range(dim_j):
                p_sym = sym % 2 != 0
                p_i = i
                for s, d in a.edges[0].segments:
                    # print(s, +sym, s == +sym, s.parity, p_sym, s.parity == p_sym)
                    if s == +sym:
                        break
                    if s.parity == p_sym:
                        p_i += d
                p_j = j
                for s, d in a.edges[1].segments:
                    if s == -sym:
                        break
                    if s.parity == p_sym:
                        p_j += d
                index_i = a.edges[0].index_by_point((+sym, i))
                index_j = a.edges[1].index_by_point((-sym, j))
                assert a[{"i": (+sym, i), "j": (-sym, j)}] == b[{"i": (p_sym, p_i), "j": (p_sym, p_j)}]
