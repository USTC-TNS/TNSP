import TAT


def test_basic_usage():
    t = TAT.U1.D.Tensor(["Left", "Right", "Up"], [
        [(-1, 3), (0, 1), (1, 2)],
        [(-1, 1), (0, 2), (1, 3)],
        [(-1, 2), (0, 3), (1, 1)],
    ]).range_(7).to("complex128")
    assert t.storage.size == 60
    assert t.norm_max() == 66
    assert t.norm_num() == 60
    assert t.norm_sum() == (7 + 66) * 60 / 2
    assert t.norm_2() == ((66 * (66 + 1) * (2 * 66 + 1) - 6 * (6 + 1) * (2 * 6 + 1)) / 6.)**(1 / 2.)
