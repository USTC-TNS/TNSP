import TAT


def test_basic_usage():
    a = TAT.No.Z.Tensor(["Left", "Right"], [3, 4]).range()
    assert a.names == ["Left", "Right"]
    assert a.rank == 2
    assert (a.storage == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]).all()
    assert a.edges(0) == a.edges("Left")
    assert a.edges(1) == a.edges("Right")

    assert a.blocks["Left", "Right"].shape == (3, 4)
    assert a.blocks["Right", "Left"].shape == (4, 3)
    assert a.blocks[("Left", ()), ("Right", ())].shape == (3, 4)
    assert a.blocks[("Right", ()), ("Left", ())].shape == (4, 3)
    assert a.blocks[("Left", TAT.No.Symmetry()), ("Right", TAT.No.Symmetry())].shape == (3, 4)
    assert a.blocks[("Right", TAT.No.Symmetry()), ("Left", TAT.No.Symmetry())].shape == (4, 3)

    assert a[{"Left": 1, "Right": 2}] == a[{"Right": 2, "Left": 1}] == 6
    assert a[{"Left": ((), 1), "Right": ((), 2)}] == a[{"Right": ((), 2), "Left": ((), 1)}] == 6


def test_when_0rank():
    a = TAT.No.Z.Tensor([], []).range()
    assert a.names == []
    assert a.rank == 0
    assert (a.storage == [0]).all()

    assert a.blocks[()].shape == ()

    assert a[{}] == 0


def test_when_0size():
    a = TAT.No.Z.Tensor(["Left", "Right"], [0, 4]).range()
    assert a.names == ["Left", "Right"]
    assert a.rank == 2
    assert a.storage.size == 0
    assert a.edges(0) == a.edges("Left")
    assert a.edges(1) == a.edges("Right")

    assert a.blocks["Left", "Right"].shape == (0, 4)
    assert a.blocks[("Left", ()), ("Right", ())].shape == (0, 4)
    assert a.blocks[("Right", TAT.No.Symmetry()), ("Left", TAT.No.Symmetry())].shape == (4, 0)


def test_conversion_scalar():
    a = TAT.No.Z.Tensor(2333, ["i", "j"])
    assert a.names == ["i", "j"]
    assert a.rank == 2
    assert (a.storage == [2333]).all()
    assert a.edges(0) == a.edges("i")
    assert a.edges(1) == a.edges("j")

    assert a.blocks["i", "j"].shape == (1, 1)
    assert a.blocks[("i", ()), ("j", ())].shape == (1, 1)
    assert a.blocks[("j", TAT.No.Symmetry()), ("i", TAT.No.Symmetry())].shape == (1, 1)

    assert a[{"i": 0, "j": 0}] == 2333
    assert a[{"i": ((), 0), "j": ((), 0)}] == 2333
    assert complex(a) == 2333
