import TAT


class FakeEdge:
    __slots__ = ["direction"]

    def __init__(self, direction):
        self.direction = direction

    def __getitem__(self, x):
        return (list(x), self.direction)


Fedge = FakeEdge(False)
Tedge = FakeEdge(True)


def test_basic_usage():
    # 1 1 0 : 3*1*3
    # 1 0 1 : 3*2*2
    # 0 1 1 : 1*1*2
    # 0 0 0 : 1*2*3
    a = TAT.Parity.D.Tensor(["Left", "Right", "Up"], [
        Tedge[(True, 3), (False, 1)],
        Fedge[(True, 1), (False, 2)],
        Tedge[(True, 2), (False, 3)],
    ]).range_()
    assert a.names == ["Left", "Right", "Up"]
    assert a.edges("Left") == a.edges(0)
    assert a.edges("Right") == a.edges(1)
    assert a.edges("Up") == a.edges(2)
    assert a.edges(0).arrow == True
    assert a.edges(1).arrow == False
    assert a.edges(2).arrow == True

    assert a.blocks[("Left", True), ("Right", False), ("Up", True)].shape == (3, 2, 2)
    assert a.blocks[("Left", False), ("Up", True), ("Right", True)].shape == (1, 2, 1)

    assert a[{"Left": (True, 2), "Right": (False, 0), "Up": (True, 1)}] == 3 * 1 * 3 + 9
    assert a[{"Left": (False, 0), "Right": (False, 1), "Up": (False, 2)}] == 3 * 1 * 3 + 3 * 2 * 2 + 1 * 1 * 2 + 5


def test_when_0rank():
    a = TAT.Parity.D.Tensor([], []).range_(2333)
    assert a.names == []
    assert all(a.storage == [2333])

    assert a.blocks[()].shape == ()

    assert a[{}] == 2333
    assert float(a) == 2333


def test_when_0size():
    a = TAT.Fermi.D.Tensor(["Left", "Right", "Up"], [
        Fedge[((0, 0),)],
        Tedge[(-1, 1), (0, 2), (1, 3)],
        Tedge[(-1, 2), (0, 3), (1, 1)],
    ]).zero_()
    assert a.names == ["Left", "Right", "Up"]
    assert a.storage.size == 0
    assert a.edges("Left") == a.edges(0)
    assert a.edges("Right") == a.edges(1)
    assert a.edges("Up") == a.edges(2)
    assert a.edges(0).arrow == False
    assert a.edges(1).arrow == True
    assert a.edges(2).arrow == True

    assert a.blocks[("Left", 0), ("Right", +1), ("Up", -1)].shape == (0, 3, 2)
    assert a.blocks[("Left", 0), ("Up", -1), ("Right", +1)].shape == (0, 2, 3)


def test_when_0block():
    a = TAT.Fermi.D.Tensor(["Left", "Right", "Up"], [
        Fedge[()],
        Fedge[(-1, 1), (0, 2), (1, 3)],
        Tedge[(-1, 2), (0, 3), (1, 1)],
    ]).zero_()
    assert a.names == ["Left", "Right", "Up"]
    assert a.storage.size == 0
    assert a.edges("Left") == a.edges(0)
    assert a.edges("Right") == a.edges(1)
    assert a.edges("Up") == a.edges(2)
    assert a.edges(0).arrow == False
    assert a.edges(1).arrow == False
    assert a.edges(2).arrow == True


def test_conversion_scalar():
    a = TAT.Fermi.D.Tensor(2333, ["i", "j"], [-2, +2], [True, False])
    assert a.names == ["i", "j"]
    assert (a.storage == [2333]).all()
    assert a.edges("i") == a.edges(0)
    assert a.edges("j") == a.edges(1)
    assert a.edges(0).arrow == True
    assert a.edges(1).arrow == False

    assert a.blocks[("i", -2), ("j", +2)].shape == (1, 1)
    assert a.blocks[("j", +2), ("i", -2)].shape == (1, 1)

    assert float(a) == 2333
    assert a[{"i": 0, "j": 0}] == 2333


def test_conversion_scalar_empty():
    a = TAT.Fermi.Z.Tensor(["i"], [[
        (-2, 333),
    ]]).range_(2333)
    assert complex(a) == 0
