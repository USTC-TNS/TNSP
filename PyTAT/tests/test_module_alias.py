# -*- coding: utf-8 -*-
import TAT


def test_normal_tensor():
    assert TAT.No == TAT.Normal


def test_scalar():
    assert TAT.FermiU1.S == TAT.FermiU1.float32
    assert TAT.BoseZ2.D == TAT.BoseZ2.float == TAT.BoseZ2.float64
    assert TAT.FermiZ2.C == TAT.FermiZ2.complex64
    assert TAT.FermiU1BoseU1.Z == TAT.FermiU1BoseU1.complex == TAT.FermiU1BoseU1.complex128
