# -*- coding: utf-8 -*-
import TAT


def trace_two(tensor, pairs, fuses={}):
    traced_tensor_0 = tensor.trace(pairs, fuses)

    double_names = set()
    names = []
    edges = []
    for n0, n1 in pairs:
        double_names.add((n0, n0))
        double_names.add((n1, n1))
        names.append(n0)
        names.append(n1)
        edges.append(tensor.edge_by_name(n0).conjugate())
        edges.append(tensor.edge_by_name(n1).conjugate())
    identity = tensor.__class__(names, edges).identity_(pairs)
    if fuses:
        for out, [in_0, in_1] in fuses.items():
            dimension = tensor.edge_by_name(in_0).dimension
            tee = tensor.__class__([out, in_0, in_1], [dimension, dimension, dimension]).zero_()
            for i in range(dimension):
                tee[{out: i, in_0: i, in_1: i}] = 1
            identity = identity.contract(tee, set())
            double_names.add((in_0, in_0))
            double_names.add((in_1, in_1))
    traced_tensor_1 = tensor.contract(identity, double_names)

    assert (traced_tensor_0 - traced_tensor_1).norm_max() == 0
    return traced_tensor_0, traced_tensor_1


def test_no_symmetry():
    trace_two(TAT.No.D.Tensor(
        ["A", "B", "C", "D", "E"],
        [2, 3, 2, 3, 4],
    ).range_(), {("A", "C"), ("B", "D")})
    trace_two(TAT.No.D.Tensor(
        ["A", "B", "C"],
        [2, 3, 2],
    ).range_(), {("A", "C")})
    a = TAT.No.D.Tensor(["A", "B", "C"], [4, 3, 5]).range_()
    b = TAT.No.D.Tensor(["D", "E", "F"], [5, 4, 6]).range_()
    trace_two(a.contract(b, set()), {("A", "E"), ("C", "D")})


def test_u1_symmetry():
    a = TAT.U1.D.Tensor(
        ["A", "B", "C", "D"],
        [
            ([(-1, 1), (0, 1), (-2, 1)], True),
            ([(0, 1), (1, 2)], False),
            ([(0, 2), (1, 2)], False),
            ([(0, 2), (-1, 1), (-2, 2)], True),
        ],
    ).range_()
    b = TAT.U1.D.Tensor(
        ["E", "F", "G", "H"],
        [
            ([(0, 2), (1, 1)], False),
            ([(-2, 1), (-1, 1), (0, 2)], True),
            ([(0, 1), (-1, 2)], True),
            ([(0, 2), (1, 1), (2, 2)], False),
        ],
    ).range_()
    c = a.contract(b, set())
    d = trace_two(c, {("B", "G")})
    e = trace_two(d[0], {("H", "D")})
    f = trace_two(c, {("G", "B"), ("D", "H")})
    assert (e[0] - f[0]).norm_max() == 0


def test_fermi_symmetry():
    a = TAT.Fermi.D.Tensor(
        ["A", "B", "C", "D"],
        [
            ([(-1, 1), (0, 1), (-2, 1)], True),
            ([(0, 1), (1, 2)], False),
            ([(0, 2), (1, 2)], False),
            ([(-2, 2), (-1, 1), (0, 2)], True),
        ],
    ).range_()
    b = TAT.Fermi.D.Tensor(
        ["E", "F", "G", "H"],
        [
            ([(0, 2), (1, 1)], False),
            ([(-2, 1), (-1, 1), (0, 2)], True),
            ([(0, 1), (-1, 2)], True),
            ([(2, 2), (1, 1), (0, 2)], False),
        ],
    ).range_()
    c = a.contract(b, set())
    d = trace_two(c, {("B", "G")})
    e = trace_two(d[0], {("H", "D")})
    f = trace_two(c, {("G", "B"), ("D", "H")})
    assert (e[0] - f[0]).norm_max() == 0


def test_fuse():
    a = TAT.No.D.Tensor(["A", "B", "C", "D"], [4, 4, 4, 4]).range_()
    b = TAT.No.D.Tensor(["E", "F", "G", "H"], [4, 4, 4, 4]).range_()
    c = a.contract(b, set())
    d = trace_two(c, {("B", "G")}, {"X": ("C", "F")})
    e = trace_two(d[0], set(), {"Y": ("A", "H")})
    f = trace_two(c, {("B", "G")}, {"X": ("F", "C"), "Y": ("A", "H")})
    assert (e[0] - f[0]).norm_max() == 0
