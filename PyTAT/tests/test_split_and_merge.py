import TAT


def test_no_symmetry_basic():
    a = TAT.No.D.Tensor(["Left", "Right"], [2, 3]).range_()

    b = a.merge_edge({"Merged": ["Left", "Right"]})
    assert all(b.storage == a.storage)
    b_a = b.split_edge({"Merged": [("Left", 2), ("Right", 3)]})
    assert (b_a - a).norm_max() == 0

    c = a.merge_edge({"Merged": ["Right", "Left"]})
    assert all(c.storage == a.transpose(["Right", "Left"]).storage)
    c_a = c.split_edge({"Merged": [("Right", 3), ("Left", 2)]})
    assert (c_a - a).norm_max() == 0


def test_no_symmetry_high_dimension():
    a = TAT.No.D.Tensor(["1", "2", "3", "4", "5", "6", "7", "8"], [2, 2, 2, 2, 2, 2, 2, 2]).range_()
    for i in range(8):
        for j in range(i, 8):
            names = a.names[i:j]
            plans = [(name, 2) for name in names]
            b = a.merge_edge({"m": names})
            assert all(b.storage == a.storage)
            c = b.split_edge({"m": plans})
            assert (c - a).norm_max() == 0


def test_u1_symmetry_basic():
    a = TAT.U1.D.Tensor(["i", "j"], [[-1, 0, +1], [-1, 0, +1]]).range_()
    d = TAT.U1.D.Tensor(["m"], [[(-2, 1), (-1, 2), (0, 3), (+1, 2), (+2, 1)]]).range_()

    b = a.merge_edge({"m": ["i", "j"]})
    assert (d - b).norm_max() == 0
    c = b.split_edge({"m": [("i", [-1, 0, +1]), ("j", [-1, 0, +1])]})
    assert (c - a).norm_max() == 0


def test_u1_symmetry_high_dimension():
    edge = [(-1, 2), (0, 2), (+1, 2)]
    a = TAT.U1.D.Tensor(["1", "2", "3", "4", "5"], [edge, edge, edge, edge, edge]).range_()
    for i in range(5):
        for j in range(i, 5):
            names = a.names[i:j]
            plans = [(name, edge) for name in names]
            b = a.merge_edge({"m": names})
            c = b.split_edge({"m": plans})
            assert (c - a).norm_max() == 0


def test_fermi_symmetry_high_dimension():
    edge = [(-1, 2), (0, 2), (+1, 2)]
    a = TAT.Fermi.D.Tensor(["1", "2", "3", "4", "5"], [edge, edge, edge, edge, edge]).range_()
    for i in range(5):
        for j in range(i, 5):
            for p in [False, True]:
                names = a.names[i:j]
                plans = [(name, edge) for name in names]
                b = a.merge_edge({"m": names}, p)
                c = b.split_edge({"m": plans}, p)
                assert (c - a).norm_max() == 0


def test_fermi_symmetry_high_dimension_compare_u1():
    edge = [(-1, 1), (0, 1), (+1, 1)]
    a_u1 = TAT.U1.D.Tensor(["1", "2", "3", "4", "5"], [edge, edge, edge, edge, edge]).range_()
    a_f = TAT.Fermi.D.Tensor(["1", "2", "3", "4", "5"], [edge, edge, edge, edge, edge]).range_()
    for i in range(5):
        for j in range(i, 5):
            for p in [False, True]:
                names = a_u1.names[i:j]
                plans = [(name, edge) for name in names]
                b_u1 = a_u1.merge_edge({"m": names})
                b_f = a_f.merge_edge({"m": names}, p)
                if p:
                    s = [0, 0, 0, 0, 0]
                    for s[0] in [-1, 0, +1]:
                        for s[1] in [-1, 0, +1]:
                            for s[2] in [-1, 0, +1]:
                                for s[3] in [-1, 0, +1]:
                                    for s[4] in [-1, 0, +1]:
                                        if sum(s) != 0:
                                            continue
                                        item = a_u1[{
                                            "1": (s[0], 0),
                                            "2": (s[1], 0),
                                            "3": (s[2], 0),
                                            "4": (s[3], 0),
                                            "5": (s[4], 0),
                                        }]
                                        assert item in b_u1.storage
                                        count = sum(s[x] != 0 for x in range(i, j))
                                        parity = count & 2
                                        if parity:
                                            assert -item in b_f.storage
                                        else:
                                            assert item in b_f.storage
                else:
                    assert all(b_u1.storage == b_f.storage)
