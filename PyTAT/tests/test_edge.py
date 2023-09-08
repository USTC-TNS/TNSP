import TAT


def test_create_trivial():
    assert TAT.No.Edge(2) == TAT.No.Edge([(TAT.No.Symmetry(), 2)])
    assert TAT.Fermi.Edge(2) == TAT.Fermi.Edge([(TAT.Fermi.Symmetry(0), 2)])
    assert TAT.Fermi.Edge(2).arrow == False


def test_create_symmetry_list():
    assert TAT.U1.Edge([1, 2, 3]) == TAT.U1.Edge([(1, 1), (2, 1), (3, 1)])
    assert TAT.Fermi.Edge([1, 2, 3]) == TAT.Fermi.Edge([(1, 1), (2, 1), (3, 1)], False)
    assert TAT.Fermi.Edge([1, 2, 3], True) == TAT.Fermi.Edge([(1, 1), (2, 1), (3, 1)], True)
    assert TAT.Fermi.Edge([1, 2, 3], True) != TAT.Fermi.Edge([(1, 1), (2, 1), (3, 1)])
    assert TAT.Fermi.Edge(([1, 2, 3], True)) == TAT.Fermi.Edge([(1, 1), (2, 1), (3, 1)], True)


def test_segments():
    assert TAT.No.Edge(2).segments == [(TAT.No.Symmetry(), 2)]
    assert TAT.Z2.Edge(2).segments == [(TAT.Z2.Symmetry(0), 2)]
    assert TAT.U1.Edge([1, 2, 3]).segments == [(sym, 1) for sym in [1, 2, 3]]
    assert TAT.U1.Edge([1, 2, 3]).segments == [((sym,), 1) for sym in [1, 2, 3]]
    e = TAT.U1.Edge([1, 2, 3])
    assert TAT.U1.Edge(e.segments).segments == [(sym, 1) for sym in [1, 2, 3]]
    assert TAT.U1.Edge([(1, 2), (2, 2), (3, 2)]).segments == [(sym, 2) for sym in [1, 2, 3]]
    assert TAT.Fermi.Edge([(1, 2), (2, 2), (3, 2)]).segments == [(sym, 2) for sym in [1, 2, 3]]
    assert TAT.Fermi.Edge([(1, 2), (2, 2), (3, 2)], True).segments == [(sym, 2) for sym in [1, 2, 3]]


def test_arrow_when_construct():
    assert TAT.Fermi.Edge([(1, 2), (2, 2), (3, 2)], False), arrow() == False
    assert TAT.Fermi.Edge([(1, 2), (2, 2), (3, 2)], True), arrow() == True
    assert TAT.U1.Edge([(1, 2), (2, 2), (3, 2)], False), arrow() == False
    assert TAT.U1.Edge([(1, 2), (2, 2), (3, 2)], True), arrow() == False


def test_compare_arrow():
    assert TAT.U1.Edge([1, 2, 3], False) == TAT.U1.Edge([1, 2, 3], True)
    assert TAT.Fermi.Edge([1, 2, 3], False) != TAT.Fermi.Edge([1, 2, 3], True)


def test_compare_segments():
    assert TAT.U1.Edge([(1, 2), (2, 2), (3, 2)]) == TAT.U1.Edge([(1, 2), (2, 2), (3, 2)])
    assert TAT.U1.Edge([(1, 2), (2, 2), (3, 2)]) != TAT.U1.Edge([(1, 2), (2, 2), (3, 3)])
    assert TAT.U1.Edge([(1, 2), (2, 2), (3, 2)]) != TAT.U1.Edge([(1, 2), (2, 2), (4, 2)])


def test_conjugate():
    e1 = TAT.Fermi.Edge([(+1, 2), (+2, 2), (+3, 2)], True)
    e2 = TAT.Fermi.Edge([(-1, 2), (-2, 2), (-3, 2)], False)
    assert e1.conjugated() == e2


def test_segments_size():
    e1 = TAT.U1.Edge([(1, 2), (2, 2), (3, 2)])
    assert len(e1.segments) == e1.segments_size
    e2 = TAT.Parity.Edge([(False, 2), (True, 2)], True)
    assert len(e2.segments) == e2.segments_size


def test_total_dimension():
    e1 = TAT.U1.Edge([(1, 2), (2, 2), (3, 2)])
    assert e1.dimension == 6
    e2 = TAT.Parity.Edge([(False, 2), (True, 2)], True)
    assert e2.dimension == 4


def test_segment_query():
    e1 = TAT.Fermi.Edge([(1, 2), (2, 2), (3, 4)], True)
    assert e1.coord_by_point((2, 1)) == (1, 1)
    assert e1.point_by_coord((1, 1)) == (2, 1)
    assert e1.coord_by_index(3) == (1, 1)
    assert e1.index_by_coord((1, 1)) == 3
    assert e1.point_by_index(3) == (2, 1)
    assert e1.index_by_point((2, 1)) == 3

    assert e1.dimension_by_symmetry(1) == 2
    assert e1.dimension_by_symmetry(2) == 2
    assert e1.dimension_by_symmetry(3) == 4
    assert e1.position_by_symmetry(1) == 0
    assert e1.position_by_symmetry(2) == 1
    assert e1.position_by_symmetry(3) == 2
