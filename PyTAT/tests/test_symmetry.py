import TAT


def test_create_symmetry():
    assert TAT.No.Symmetry() == TAT.No.Symmetry(())

    assert TAT.Z2.Symmetry() == TAT.Z2.Symmetry(False) == TAT.Z2.Symmetry((False,))
    assert TAT.Z2.Symmetry(True).z2 == True

    assert TAT.U1.Symmetry() == TAT.U1.Symmetry(0) == TAT.U1.Symmetry((0,))
    assert TAT.U1.Symmetry(2).u1 == 2

    assert TAT.Fermi.Symmetry() == TAT.Fermi.Symmetry(0) == TAT.Fermi.Symmetry((0,))
    assert TAT.Fermi.Symmetry(2).fermi == 2

    assert TAT.FermiZ2.Symmetry() == TAT.FermiZ2.Symmetry(0, False) == TAT.FermiZ2.Symmetry((0, False))
    assert TAT.FermiZ2.Symmetry(2, True).fermi == 2
    assert TAT.FermiZ2.Symmetry(2, True).z2 == True

    assert TAT.FermiU1.Symmetry() == TAT.FermiU1.Symmetry(0, 0) == TAT.FermiU1.Symmetry((0, 0))
    assert TAT.FermiU1.Symmetry(2, 3).fermi == 2
    assert TAT.FermiU1.Symmetry(2, 3).u1 == 3

    assert TAT.Parity.Symmetry() == TAT.Parity.Symmetry(False) == TAT.Parity.Symmetry((False,))
    assert TAT.Parity.Symmetry(True).parity == True

    assert TAT.FermiFermi.Symmetry() == TAT.FermiFermi.Symmetry(0, 0) == TAT.FermiFermi.Symmetry((0, 0))
    assert TAT.FermiFermi.Symmetry(2, 3).fermi_0 == 2
    assert TAT.FermiFermi.Symmetry(2, 3).fermi_1 == 3


def test_compare():
    assert TAT.FermiZ2.Symmetry(4, False) < TAT.FermiZ2.Symmetry(5, False)
    assert TAT.FermiZ2.Symmetry(5, True) >= TAT.FermiZ2.Symmetry(4, True)
    assert TAT.FermiZ2.Symmetry(4, False) < TAT.FermiZ2.Symmetry(4, True)
    assert TAT.FermiZ2.Symmetry(4, False) <= TAT.FermiZ2.Symmetry(5, True)
    assert TAT.FermiZ2.Symmetry(5, False) > TAT.FermiZ2.Symmetry(4, True)


def test_arithmetic():
    assert -TAT.FermiZ2.Symmetry(4, False) == TAT.FermiZ2.Symmetry(-4, False)
    assert -TAT.FermiZ2.Symmetry(4, True) == TAT.FermiZ2.Symmetry(-4, True)
    assert TAT.FermiZ2.Symmetry(2, True) + TAT.FermiZ2.Symmetry(-1, True) == TAT.FermiZ2.Symmetry(1, False)
    assert TAT.FermiZ2.Symmetry(2, False) - TAT.FermiZ2.Symmetry(-1, True) == TAT.FermiZ2.Symmetry(3, True)


def test_parity():
    assert TAT.U1.Symmetry(233).parity == False
    assert TAT.Fermi.Symmetry(233).parity == True
    assert TAT.FermiU1.Symmetry(1, 2).parity == True
    assert TAT.FermiZ2.Symmetry(2, True).parity == False
    assert TAT.FermiFermi.Symmetry(2, 3).parity == True
    assert TAT.FermiFermi.Symmetry(3, 3).parity == False


def test_hash():
    assert hash(TAT.Fermi.Symmetry(4)) == hash(TAT.U1.Symmetry(4))
    assert hash(TAT.Fermi.Symmetry(4)) != hash(TAT.Fermi.Symmetry(5))


def test_io():
    import pickle

    e = TAT.FermiZ2.Symmetry(233, True)
    assert e == pickle.loads(pickle.dumps(e))
