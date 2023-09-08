import TAT


def test_normal_tensor():
    assert TAT.No == TAT.Normal


def test_scalar():
    assert TAT.Fermi.S == TAT.Fermi.float32
    assert TAT.Z2.D == TAT.Z2.float == TAT.Z2.float64
    assert TAT.Parity.C == TAT.Parity.complex64
    assert TAT.FermiU1.Z == TAT.FermiU1.complex == TAT.FermiU1.complex128
