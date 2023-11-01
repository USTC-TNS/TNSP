"Test compat"

import torch
import tat
from tat import compat as TAT

# pylint: disable=missing-function-docstring
# pylint: disable=invalid-name
# pylint: disable=singleton-comparison

# It is strange, but pylint complains function args too many. So add it here
# pylint: disable=too-many-function-args


def test_edge_from_dimension() -> None:
    assert TAT.No.Edge(4) == tat.Edge(dimension=4)
    assert TAT.Fermi.Edge(4) == tat.Edge(fermion=[True],
                                         symmetry=[torch.tensor([0, 0, 0, 0], dtype=torch.int)],
                                         arrow=False)
    assert TAT.Z2.Edge(4) == tat.Edge(symmetry=[torch.tensor([False, False, False, False])])


def test_edge_from_segments() -> None:
    assert TAT.Z2.Edge([
        (False, 2),
        (True, 3),
    ]) == tat.Edge(symmetry=[torch.tensor([False, False, True, True, True])])
    assert TAT.Fermi.Edge([
        (-1, 1),
        (0, 2),
        (+1, 3),
    ], True) == tat.Edge(
        symmetry=[torch.tensor([-1, 0, 0, +1, +1, +1], dtype=torch.int)],
        arrow=True,
        fermion=[True],
    )
    assert TAT.FermiFermi.Edge([
        ((-1, -2), 1),
        ((0, +1), 2),
        ((+1, 0), 3),
    ], True) == tat.Edge(
        symmetry=[
            torch.tensor([-1, 0, 0, +1, +1, +1], dtype=torch.int),
            torch.tensor([-2, +1, +1, 0, 0, 0], dtype=torch.int),
        ],
        arrow=True,
        fermion=[True, True],
    )


def test_edge_from_segments_without_dimension() -> None:
    assert TAT.Z2.Edge([False, True]) == tat.Edge(symmetry=[torch.tensor([False, True])])
    assert TAT.Fermi.Edge([-1, 0, +1], True) == tat.Edge(
        symmetry=[torch.tensor([-1, 0, +1], dtype=torch.int)],
        arrow=True,
        fermion=[True],
    )
    assert TAT.FermiFermi.Edge([
        (-1, -2),
        (0, +1),
        (+1, 0),
    ], True) == tat.Edge(
        symmetry=[torch.tensor([-1, 0, +1], dtype=torch.int),
                  torch.tensor([-2, +1, 0], dtype=torch.int)],
        arrow=True,
        fermion=[True, True],
    )


def test_edge_from_tuple() -> None:
    assert TAT.FermiFermi.Edge(([
        ((-1, -2), 1),
        ((0, +1), 2),
        ((+1, 0), 3),
    ], True)) == tat.Edge(
        symmetry=[
            torch.tensor([-1, 0, 0, +1, +1, +1], dtype=torch.int),
            torch.tensor([-2, +1, +1, 0, 0, 0], dtype=torch.int),
        ],
        arrow=True,
        fermion=[True, True],
    )
    assert TAT.FermiFermi.Edge(([
        (-1, -2),
        (0, +1),
        (+1, 0),
    ], True)) == tat.Edge(
        symmetry=[torch.tensor([-1, 0, +1], dtype=torch.int),
                  torch.tensor([-2, +1, 0], dtype=torch.int)],
        arrow=True,
        fermion=[True, True],
    )


def test_tensor() -> None:
    a = TAT.FermiZ2.D.Tensor(["i", "j"], [
        [(-1, False), (-1, True), (0, True), (0, False)],
        [(+1, True), (+1, False), (0, False), (0, True)],
    ])
    b = tat.Tensor(
        [
            "i",
            "j",
        ],
        [
            tat.Edge(
                fermion=[True, False],
                symmetry=[
                    torch.tensor([-1, -1, 0, 0], dtype=torch.int),
                    torch.tensor([False, True, True, False]),
                ],
                arrow=False,
            ),
            tat.Edge(
                fermion=[True, False],
                symmetry=[
                    torch.tensor([+1, +1, 0, 0], dtype=torch.int),
                    torch.tensor([True, False, False, True]),
                ],
                arrow=False,
            ),
        ],
    )
    assert a.same_shape_with(b, allow_transpose=False)
