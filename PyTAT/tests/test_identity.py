import TAT

arrange_pairs_indices = [
    [0, 1, 2, 3, 4, 5],
    [0, 1, 2, 4, 5, 3],
    [0, 1, 2, 5, 3, 4],
    [0, 2, 1, 3, 4, 5],
    [0, 2, 1, 4, 5, 3],
    [0, 2, 1, 5, 3, 4],
    [0, 3, 1, 2, 4, 5],
    [0, 3, 1, 4, 5, 2],
    [0, 3, 1, 5, 2, 4],
    [0, 4, 1, 2, 3, 5],
    [0, 4, 1, 3, 5, 2],
    [0, 4, 1, 5, 2, 3],
    [0, 5, 1, 2, 3, 4],
    [0, 5, 1, 3, 4, 2],
    [0, 5, 1, 4, 2, 3],
]

order_lists = [
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1],
]


def test_no_symmetry_0():
    a = TAT.No.float.Tensor(["i", "j"], [4, 4]).identity({("i", "j")})
    assert (a - a.contract(a, {("i", "j")})).norm_max() == 0
    assert (a - a.contract(a, {("j", "i")})).norm_max() == 0


def test_no_symmetry_1():
    a = TAT.No.float.Tensor(["i", "j"], [4, 4]).identity({("j", "i")})
    assert (a - a.contract(a, {("i", "j")})).norm_max() == 0
    assert (a - a.contract(a, {("j", "i")})).norm_max() == 0


def test_no_symmetry_2():
    half_rank = 3
    for pairs_index in arrange_pairs_indices:
        a = TAT.No.S.Tensor(["1", "2", "3", "4", "5", "6"], [4, 4, 4, 4, 4, 4]).range()
        pairs = set()
        for i in range(half_rank):
            p0 = pairs_index[i * 2 + 0]
            p1 = pairs_index[i * 2 + 1]
            pairs.add((a.names[p0], a.names[p1]))
        a.identity(pairs)
        assert (a - a.contract(a, pairs)).norm_max() == 0


def test_z2_symmetry_0():
    half_rank = 3
    edge = [(False, 2), (True, 2)]
    for pairs_index in arrange_pairs_indices:
        a = TAT.Z2.S.Tensor(["1", "2", "3", "4", "5", "6"], [edge, edge, edge, edge, edge, edge]).range()
        pairs = set()
        for i in range(half_rank):
            p0 = pairs_index[i * 2 + 0]
            p1 = pairs_index[i * 2 + 1]
            pairs.add((a.names[p0], a.names[p1]))
        a.identity(pairs)
        assert (a - a.contract(a, pairs)).norm_max() == 0


def test_u1_symmetry_0():
    half_rank = 3
    edge0 = [(-1, 1), (0, 1), (+1, 1)]
    edge1 = [(+1, 1), (0, 1), (-1, 1)]
    for pairs_index in arrange_pairs_indices:
        names = ["1", "2", "3", "4", "5", "6"]
        edges = [None, None, None, None, None, None]
        pairs = set()
        for i in range(half_rank):
            p0 = pairs_index[i * 2 + 0]
            p1 = pairs_index[i * 2 + 1]
            pairs.add((names[p0], names[p1]))
            edges[p0] = edge0
            edges[p1] = edge1
        a = TAT.U1.S.Tensor(names, edges).range()
        a.identity(pairs)
        assert (a - a.contract(a, pairs)).norm_max() == 0


def test_fermi_symmetry_0():
    half_rank = 3
    edge0 = ([(-1, 1), (0, 1), (+1, 1)], False)
    edge1 = ([(+1, 1), (0, 1), (-1, 1)], True)
    for order in order_lists:
        for pairs_index in arrange_pairs_indices:
            names = ["1", "2", "3", "4", "5", "6"]
            edges = [None, None, None, None, None, None]
            pairs = set()
            for i in range(half_rank):
                p0 = pairs_index[i * 2 + 0]
                p1 = pairs_index[i * 2 + 1]
                pairs.add((names[p0], names[p1]))
                if order[i] != 0:
                    edges[p0] = edge0
                    edges[p1] = edge1
                else:
                    edges[p0] = edge1
                    edges[p1] = edge0
            a = TAT.Fermi.S.Tensor(names, edges).range()
            a.identity(pairs)
            assert (a - a.contract(a, pairs)).norm_max() == 0
