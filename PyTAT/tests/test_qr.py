import TAT


def check_unitary(tensor, name, name_prime, fermi):
    pairs = set()
    for n in tensor.names:
        if n != name:
            pairs.add((n, n))
    conjugated = tensor.conjugate(True).edge_rename({name: name_prime})
    product = tensor.contract(conjugated, pairs)
    identity = product.same_shape().identity({(name, name_prime)})
    if fermi:
        product.transform(abs)
        identity.transform(abs)
    assert (product - identity).norm_max() < 1e-6


def test_no_symmetry_0():
    a = TAT.No.D.Tensor(["A", "B"], [5, 10]).range()
    q, r = a.qr('r', {"A"}, "newQ", "newR")
    q_2, r_2 = a.qr('q', {"B"}, "newQ", "newR")
    assert (q - q_2).norm_max() == 0
    assert (r - r_2).norm_max() == 0
    assert q.names == ["newQ", "B"]
    assert r.names == ["A", "newR"]
    assert q.edges("newQ").dimension == 5
    assert r.edges("newR").dimension == 5
    check_unitary(q, "newQ", "newQ'", False)
    assert (q.contract(r, {("newQ", "newR")}) - a).norm_max() < 1e-6


def test_no_symmetry_1():
    a = TAT.No.Z.Tensor(["A", "B"], [10, 5]).range(-21 - 29j, 2 + 3j)
    q, r = a.qr('r', {"A"}, "newQ", "newR")
    q_2, r_2 = a.qr('q', {"B"}, "newQ", "newR")
    assert (q - q_2).norm_max() == 0
    assert (r - r_2).norm_max() == 0
    assert q.names == ["newQ", "B"]
    assert r.names == ["A", "newR"]
    assert q.edges("newQ").dimension == 5
    assert r.edges("newR").dimension == 5
    check_unitary(q, "newQ", "newQ'", False)
    assert (q.contract(r, {("newQ", "newR")}) - a).norm_max() < 1e-6


def test_fermi_symmetry_0():
    a = TAT.Fermi.D.Tensor(
        ["A", "B"],
        [
            ([(-1, 2), (0, 1), (+1, 2)], False),
            ([(-1, 4), (0, 3), (+1, 3)], True),
        ],
    ).range()
    q, r = a.qr('r', {"A"}, "newQ", "newR")
    q_2, r_2 = a.qr('q', {"B"}, "newQ", "newR")
    assert (q - q_2).norm_max() == 0
    assert (r - r_2).norm_max() == 0
    assert q.names == ["newQ", "B"]
    assert r.names == ["A", "newR"]
    assert q.edges("newQ").dimension == 5
    assert r.edges("newR").dimension == 5
    check_unitary(q, "newQ", "newQ'", True)
    assert (q.contract(r, {("newQ", "newR")}) - a).norm_max() < 1e-6


def test_fermi_symmetry_1():
    a = TAT.Fermi.Z.Tensor(
        ["A", "B"],
        [
            ([(-1, 2), (0, 1), (+1, 2)], True),
            ([(-1, 4), (0, 3), (+1, 3)], False),
        ],
    ).range()
    q, r = a.qr('r', {"A"}, "newQ", "newR")
    q_2, r_2 = a.qr('q', {"B"}, "newQ", "newR")
    assert (q - q_2).norm_max() == 0
    assert (r - r_2).norm_max() == 0
    assert q.names == ["newQ", "B"]
    assert r.names == ["A", "newR"]
    assert q.edges("newQ").dimension == 5
    assert r.edges("newR").dimension == 5
    check_unitary(q, "newQ", "newQ'", True)
    assert (q.contract(r, {("newQ", "newR")}) - a).norm_max() < 1e-6


def test_fermi_symmetry_edge_mismatch():
    a = TAT.Fermi.Z.Tensor(
        ["A", "B"],
        [
            ([(-1, 2), (0, 2), (+2, 2)], True),
            ([(-1, 4), (0, 3), (+2, 3)], False),
        ],
    ).range()
    q, r = a.qr('r', {"B"}, "newQ", "newR")
    q_2, r_2 = a.qr('q', {"A"}, "newQ", "newR")
    assert (q - q_2).norm_max() == 0
    assert (r - r_2).norm_max() == 0
    assert q.names == ["A", "newQ"]
    assert r.names == ["newR", "B"]
    assert q.edges("newQ").dimension == 2
    assert r.edges("newR").dimension == 2
    check_unitary(q, "newQ", "newQ'", True)
    assert (q.contract(r, {("newQ", "newR")}) - a).norm_max() < 1e-6
