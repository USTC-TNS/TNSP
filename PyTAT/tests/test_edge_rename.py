import TAT


def test_basic_rename():
    import numpy as np
    t1 = TAT.Z2.D.Tensor(["Left", "Right", "Phy"], [
        [(False, 1), (True, 2)],
        [(False, 3), (True, 4)],
        [(False, 5), (True, 6)],
    ]).range_()
    t2 = t1.edge_rename({"Left": "Up"})
    assert t1.names == ["Left", "Right", "Phy"]
    assert t2.names == ["Up", "Right", "Phy"]
    assert np.shares_memory(t1.storage, t2.storage)
