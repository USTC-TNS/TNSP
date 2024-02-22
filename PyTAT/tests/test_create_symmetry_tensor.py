# -*- coding: utf-8 -*-
import TAT


def test_basic_usage():
    # 1 1 0 : 3*1*3
    # 1 0 1 : 3*2*2
    # 0 1 1 : 1*1*2
    # 0 0 0 : 1*2*3
    a = TAT.BoseZ2.D.Tensor(
        ["Left", "Right", "Up"],
        [[(True, 3), (False, 1)], [(True, 1), (False, 2)], [(True, 2), (False, 3)]],
    ).range_()
    assert a.names == ["Left", "Right", "Up"]
    assert a.rank == 3
    assert a.storage.size == 1 * 2 * 3 + 1 * 1 * 2 + 3 * 2 * 2 + 3 * 1 * 3
    assert a.edges[0] == a.edge_by_name("Left")
    assert a.edges[1] == a.edge_by_name("Right")
    assert a.edges[2] == a.edge_by_name("Up")

    assert a.blocks[[("Left", True), ("Right", False), ("Up", True)]].shape == (3, 2, 2)
    assert a.blocks[[("Left", False), ("Right", True), ("Up", True)]].shape == (1, 1, 2)

    assert a[{"Left": (True, 2), "Right": (False, 0), "Up": (True, 1)}] == 3 * 1 * 3 + 9
    assert a[{"Left": 2, "Right": 1, "Up": 1}] == 3 * 1 * 3 + 9
    assert a[{"Left": (False, 0), "Right": (False, 1), "Up": (False, 2)}] == 3 * 1 * 3 + 3 * 2 * 2 + 1 * 1 * 2 + 5
    assert a[{"Left": 3, "Right": 2, "Up": 4}] == 3 * 1 * 3 + 3 * 2 * 2 + 1 * 1 * 2 + 5


def test_when_0rank():
    a = TAT.BoseU1.D.Tensor([], []).range_(2333)
    assert a.names == []
    assert a.rank == 0
    assert (a.storage == [2333]).all()

    assert a.blocks[()].shape == ()

    assert a[{}] == 2333


def test_when_0size():
    a = TAT.BoseU1.D.Tensor(
        ["Left", "Right", "Up"],
        [[
            (0, 0),
        ], [(-1, 1), (0, 2), (1, 3)], [(-1, 2), (0, 3), (1, 1)]],
    ).zero_()
    assert a.names == ["Left", "Right", "Up"]
    assert a.rank == 3
    assert a.storage.size == 0
    assert a.edges[0] == a.edge_by_name("Left")
    assert a.edges[1] == a.edge_by_name("Right")
    assert a.edges[2] == a.edge_by_name("Up")

    assert a.blocks[("Left", 0), ("Right", +1), ("Up", -1)].shape == (0, 3, 2)


def test_when_0block():
    a = TAT.BoseU1.D.Tensor(
        ["Left", "Right", "Up"],
        [[], [(-1, 1), (0, 2), (1, 3)], [(-1, 2), (0, 3), (1, 1)]],
    ).zero_()
    assert a.names == ["Left", "Right", "Up"]
    assert a.rank == 3
    assert a.storage.size == 0
    assert a.edges[0] == a.edge_by_name("Left")
    assert a.edges[1] == a.edge_by_name("Right")
    assert a.edges[2] == a.edge_by_name("Up")


def test_conversion_scalar():
    a = TAT.BoseU1.D.Tensor(2333, ["i", "j"], [-2, +2])
    assert a.names == ["i", "j"]
    assert a.rank == 2
    assert (a.storage == [2333]).all()
    assert a.edges[0] == a.edge_by_name("i")
    assert a.edges[1] == a.edge_by_name("j")

    assert a.blocks[("i", -2), ("j", +2)].shape == (1, 1)
    assert a.blocks[("i", (-2,)), ("j", (+2,))].shape == (1, 1)
    assert a.blocks[("j", TAT.BoseU1.Symmetry(+2)), ("i", TAT.BoseU1.Symmetry(-2))].shape == (1, 1)

    assert a[{"i": (-2, 0), "j": (+2, 0)}] == 2333
    assert a[{"i": ((-2,), 0), "j": ((+2,), 0)}] == 2333
    assert float(a) == 2333


def test_conversion_scalar_empty():
    a = TAT.BoseU1.D.Tensor(["i"], [[
        (+2, 333),
    ]]).range_(2333)
    assert a.rank == 1
    assert float(a) == 0
