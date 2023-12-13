# -*- coding: utf-8 -*-
import TAT


def test_no_symmetry_float():
    A = TAT.No.D.Tensor(["i", "j"], [2, 3]).range_()
    A_c = A.conjugate()
    assert all(A.storage.conj() == A_c.storage)


def test_no_symmetry_complex():
    A = TAT.No.Z.Tensor(["i", "j"], [2, 3]).range_(1 + 5j, 1 + 7j)
    A_c = A.conjugate()
    assert all(A.storage.conj() == A_c.storage)


def test_u1_symmetry_float():
    A = TAT.U1.D.Tensor(
        ["i", "j"],
        [
            [(-1, 2), (0, 2), (+1, 2)],
            [(-1, 2), (0, 2), (+1, 2)],
        ],
    ).range_(-8, +1)
    A_c = A.conjugate()
    B = A.contract(A_c, {("i", "i"), ("j", "j")})
    assert float(B) > 0
    assert (A_c.conjugate() - A).norm_max() == 0


def test_u1_symmetry_complex():
    A = TAT.U1.Z.Tensor(
        ["i", "j"],
        [
            [(-1, 2), (0, 2), (+1, 2)],
            [(-1, 2), (0, 2), (+1, 2)],
        ],
    ).range_(-8 - 20j, +1 + 7j)
    A_c = A.conjugate()
    B = A.contract(A_c, {("i", "i"), ("j", "j")})
    assert complex(B).real > 0
    assert complex(B).imag == 0
    assert (A_c.conjugate() - A).norm_max() == 0


def test_Fermi_symmetry_float():
    A = TAT.Fermi.D.Tensor(
        ["i", "j"],
        [
            [(-1, 2), (0, 2), (+1, 2)],
            [(-1, 2), (0, 2), (+1, 2)],
        ],
    ).range_(-8, +1)
    A_c = A.conjugate()
    B = A.contract(A_c, {("i", "i"), ("j", "j")})
    assert float(B) > 0
    assert (A_c.conjugate() - A).norm_max() == 0


def test_Fermi_symmetry_float_bidirection_arrow():
    A = TAT.Fermi.D.Tensor(
        ["i", "j"],
        [
            ([(-1, 2), (0, 2), (+1, 2)], False),
            ([(-1, 2), (0, 2), (+1, 2)], True),
        ],
    ).range_(-8, +1)
    A_c = A.conjugate()
    B = A.contract(A_c, {("i", "i"), ("j", "j")})
    assert float(B) < 0  # The A * Ac may not be positive
    assert (A_c.conjugate() - A).norm_max() == 0


def test_Fermi_symmetry_float_bidirection_arrow_fixed():
    A = TAT.Fermi.D.Tensor(
        ["i", "j"],
        [
            ([(-1, 2), (0, 2), (+1, 2)], False),
            ([(-1, 2), (0, 2), (+1, 2)], True),
        ],
    ).range_(-8, +1)
    A_c = A.conjugate(True)
    B = A.contract(A_c, {("i", "i"), ("j", "j")})
    assert float(B) > 0
    assert (A_c.conjugate(True) - A).norm_max() == 0


def test_fermi_symmetry_complex():
    A = TAT.Fermi.Z.Tensor(
        ["i", "j"],
        [
            [(-1, 2), (0, 2), (+1, 2)],
            [(-1, 2), (0, 2), (+1, 2)],
        ],
    ).range_(-8 - 20j, +1 + 7j)
    A_c = A.conjugate()
    B = A.contract(A_c, {("i", "i"), ("j", "j")})
    assert complex(B).real > 0
    assert complex(B).imag == 0
    assert (A_c.conjugate() - A).norm_max() == 0


def test_fermi_symmetry_complex_bidirection_arrow():
    A = TAT.Fermi.Z.Tensor(
        ["i", "j"],
        [
            ([(-1, 2), (0, 2), (+1, 2)], False),
            ([(-1, 2), (0, 2), (+1, 2)], True),
        ],
    ).range_(-8 - 20j, +1 + 7j)
    A_c = A.conjugate()
    B = A.contract(A_c, {("i", "i"), ("j", "j")})
    assert complex(B).real < 0
    assert complex(B).imag == 0
    assert (A_c.conjugate() - A).norm_max() == 0


def test_fermi_symmetry_complex_bidirection_arrow_fixed():
    A = TAT.Fermi.Z.Tensor(
        ["i", "j"],
        [
            ([(-1, 2), (0, 2), (+1, 2)], False),
            ([(-1, 2), (0, 2), (+1, 2)], True),
        ],
    ).range_(-8 - 20j, +1 + 7j)
    A_c = A.conjugate(True)
    B = A.contract(A_c, {("i", "i"), ("j", "j")})
    assert complex(B).real > 0
    assert complex(B).imag == 0
    assert (A_c.conjugate(True) - A).norm_max() == 0


def test_fermi_symmetry_contract_with_conjugate():
    A = TAT.Fermi.Z.Tensor(
        ["i", "j"],
        [
            ([(-1, 2), (0, 2), (+1, 2)], False),
            ([(+1, 2), (0, 2), (-1, 2)], True),
        ],
    ).range_(-8 - 20j, +1 + 7j)
    B = TAT.Fermi.Z.Tensor(
        ["i", "j"],
        [
            ([(-1, 2), (0, 2), (+1, 2)], False),
            ([(+1, 2), (0, 2), (-1, 2)], True),
        ],
    ).range_(-7 - 29j, +5 + 3j)
    C = A.contract(B, {("i", "j")})
    A_c = A.conjugate()
    B_c = B.conjugate()
    C_c = A_c.contract(B_c, {("i", "j")})
    assert (C.conjugate() - C_c).norm_max() == 0


def test_fermi_symmetry_contract_with_conjugate_arrow_fix_wrong():
    A = TAT.Fermi.Z.Tensor(
        ["i", "j"],
        [
            ([(-1, 2), (0, 2), (+1, 2)], False),
            ([(+1, 2), (0, 2), (-1, 2)], True),
        ],
    ).range_(-8 - 20j, +1 + 7j)
    B = TAT.Fermi.Z.Tensor(
        ["i", "j"],
        [
            ([(-1, 2), (0, 2), (+1, 2)], False),
            ([(+1, 2), (0, 2), (-1, 2)], True),
        ],
    ).range_(-7 - 29j, +5 + 3j)
    C = A.contract(B, {("i", "j")})
    A_c = A.conjugate(True)
    B_c = B.conjugate(True)
    C_c = A_c.contract(B_c, {("i", "j")})
    assert (C.conjugate(True) - C_c).norm_max() > 1e-3
