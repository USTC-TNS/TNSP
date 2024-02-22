# -*- coding: utf-8 -*-
import TAT


def test_create_symmetry():
    assert TAT.No.Symmetry() == TAT.No.Symmetry(())

    assert TAT.BoseZ2.Symmetry() == TAT.BoseZ2.Symmetry(False) == TAT.BoseZ2.Symmetry((False,))
    assert TAT.BoseZ2.Symmetry(True).z2 == True

    assert TAT.BoseU1.Symmetry() == TAT.BoseU1.Symmetry(0) == TAT.BoseU1.Symmetry((0,))
    assert TAT.BoseU1.Symmetry(2).u1 == 2

    assert TAT.FermiU1.Symmetry() == TAT.FermiU1.Symmetry(0) == TAT.FermiU1.Symmetry((0,))
    assert TAT.FermiU1.Symmetry(2).fermi == 2

    assert TAT.FermiU1BoseZ2.Symmetry() == TAT.FermiU1BoseZ2.Symmetry(0, False) == TAT.FermiU1BoseZ2.Symmetry(
        (0, False))
    assert TAT.FermiU1BoseZ2.Symmetry(2, True).fermi == 2
    assert TAT.FermiU1BoseZ2.Symmetry(2, True).z2 == True

    assert TAT.FermiU1BoseU1.Symmetry() == TAT.FermiU1BoseU1.Symmetry(0, 0) == TAT.FermiU1BoseU1.Symmetry((0, 0))
    assert TAT.FermiU1BoseU1.Symmetry(2, 3).fermi == 2
    assert TAT.FermiU1BoseU1.Symmetry(2, 3).u1 == 3

    assert TAT.FermiZ2.Symmetry() == TAT.FermiZ2.Symmetry(False) == TAT.FermiZ2.Symmetry((False,))
    assert TAT.FermiZ2.Symmetry(True).parity == True

    assert TAT.FermiU1FermiU1.Symmetry() == TAT.FermiU1FermiU1.Symmetry(0, 0) == TAT.FermiU1FermiU1.Symmetry((0, 0))
    assert TAT.FermiU1FermiU1.Symmetry(2, 3).fermi_0 == 2
    assert TAT.FermiU1FermiU1.Symmetry(2, 3).fermi_1 == 3


def test_compare():
    assert TAT.FermiU1BoseZ2.Symmetry(4, False) < TAT.FermiU1BoseZ2.Symmetry(5, False)
    assert TAT.FermiU1BoseZ2.Symmetry(5, True) >= TAT.FermiU1BoseZ2.Symmetry(4, True)
    assert TAT.FermiU1BoseZ2.Symmetry(4, False) < TAT.FermiU1BoseZ2.Symmetry(4, True)
    assert TAT.FermiU1BoseZ2.Symmetry(4, False) <= TAT.FermiU1BoseZ2.Symmetry(5, True)
    assert TAT.FermiU1BoseZ2.Symmetry(5, False) > TAT.FermiU1BoseZ2.Symmetry(4, True)


def test_arithmetic():
    assert -TAT.FermiU1BoseZ2.Symmetry(4, False) == TAT.FermiU1BoseZ2.Symmetry(-4, False)
    assert -TAT.FermiU1BoseZ2.Symmetry(4, True) == TAT.FermiU1BoseZ2.Symmetry(-4, True)
    assert TAT.FermiU1BoseZ2.Symmetry(2, True) + TAT.FermiU1BoseZ2.Symmetry(-1, True) == TAT.FermiU1BoseZ2.Symmetry(
        1, False)
    assert TAT.FermiU1BoseZ2.Symmetry(2, False) - TAT.FermiU1BoseZ2.Symmetry(-1, True) == TAT.FermiU1BoseZ2.Symmetry(
        3, True)


def test_parity():
    assert TAT.BoseU1.Symmetry(233).parity == False
    assert TAT.FermiU1.Symmetry(233).parity == True
    assert TAT.FermiU1BoseU1.Symmetry(1, 2).parity == True
    assert TAT.FermiU1BoseZ2.Symmetry(2, True).parity == False
    assert TAT.FermiU1FermiU1.Symmetry(2, 3).parity == True
    assert TAT.FermiU1FermiU1.Symmetry(3, 3).parity == False


def test_hash():
    assert hash(TAT.FermiU1.Symmetry(4)) == hash(TAT.BoseU1.Symmetry(4))
    assert hash(TAT.FermiU1.Symmetry(4)) != hash(TAT.FermiU1.Symmetry(5))


def test_io():
    import pickle

    e = TAT.FermiU1BoseZ2.Symmetry(233, True)
    assert e == pickle.loads(pickle.dumps(e))
