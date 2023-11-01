"Test edge"

import torch
from tat import Edge

# pylint: disable=missing-function-docstring
# pylint: disable=invalid-name
# pylint: disable=singleton-comparison


def test_create_edge_and_basic() -> None:
    a = Edge(dimension=5)
    assert a.arrow == False
    assert a.dimension == 5
    b = Edge(symmetry=[torch.tensor([False, False, True, True])])
    assert b.arrow == False
    assert b.dimension == 4
    c = Edge(fermion=[False, True], symmetry=[torch.tensor([False, True]), torch.tensor([False, True])], arrow=True)
    assert c.arrow == True
    assert c.dimension == 2


def test_edge_conjugate_and_equal() -> None:
    a = Edge(fermion=[False, True], symmetry=[torch.tensor([False, True]), torch.tensor([0, 1])], arrow=True)
    b = Edge(fermion=[False, True], symmetry=[torch.tensor([False, True]), torch.tensor([0, -1])], arrow=False)
    assert a.conjugate() == b
    assert a != 2


def test_repr() -> None:
    a = Edge(fermion=[False, True], symmetry=[torch.tensor([False, True]), torch.tensor([0, 1])], arrow=True)
    repr_a = "Edge(dimension=2, arrow=True, fermion=(False,True), symmetry=([False,True],[0,1]))"
    assert repr_a == repr(a)
    b = Edge(symmetry=[torch.tensor([False, True]), torch.tensor([0, 1])])
    repr_b = "Edge(dimension=2, symmetry=([False,True],[0,1]))"
    assert repr_b == repr(b)
    c = Edge(dimension=4)
    repr_c = "Edge(dimension=4)"
    assert repr_c == repr(c)
