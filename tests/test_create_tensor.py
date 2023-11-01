"Test create tensor"

import torch
import tat

# pylint: disable=missing-function-docstring
# pylint: disable=invalid-name
# pylint: disable=singleton-comparison


def test_create_tensor() -> None:
    a = tat.Tensor(
        [
            "i",
            "j",
        ],
        [
            tat.Edge(symmetry=[torch.tensor([False, False, True])], fermion=[True], arrow=True),
            tat.Edge(symmetry=[torch.tensor([False, False, False, True, True])], fermion=[True], arrow=False),
        ],
    )
    assert a.rank == 2
    assert a.names == ["i", "j"]
    assert a.edges[0] == tat.Edge(symmetry=[torch.tensor([False, False, True])], fermion=[True], arrow=True)
    assert a.edges[1] == tat.Edge(symmetry=[torch.tensor([False, False, False, True, True])],
                                  fermion=[True],
                                  arrow=False)
    assert a.edges[0] == a.edge_by_name("i")
    assert a.edges[1] == a.edge_by_name("j")


def test_tensor_get_set_item() -> None:
    a = tat.Tensor(
        [
            "i",
            "j",
        ],
        [
            tat.Edge(symmetry=[torch.tensor([False, False, True])], fermion=[True], arrow=True),
            tat.Edge(symmetry=[torch.tensor([False, False, False, True, True])], fermion=[True], arrow=False),
        ],
    )
    a[{"i": 0, "j": 0}] = 1
    assert a[0, 0] == 1
    a["i":2, "j":3] = 2  # type: ignore[misc]
    assert a[{"i": 2, "j": 3}] == 2
    try:
        a[2, 0] = 3
        assert False
    except IndexError:
        pass
    assert a["i":2, "j":0] == 0  # type: ignore[misc]

    b = tat.Tensor(
        [
            "i",
            "j",
        ],
        [
            tat.Edge(symmetry=[torch.tensor([0, 0, -1])], fermion=[False]),
            tat.Edge(symmetry=[torch.tensor([0, 0, 0, +1, +1])], fermion=[False]),
        ],
    )
    b[{"i": 0, "j": 0}] = 1
    assert b[0, 0] == 1
    b["i":2, "j":3] = 2  # type: ignore[misc]
    assert b[{"i": 2, "j": 3}] == 2
    try:
        b[2, 0] = 3
        assert False
    except IndexError:
        pass
    assert b["i":2, "j":0] == 0  # type: ignore[misc]


def test_create_randn_tensor() -> None:
    a = tat.Tensor(
        ["i", "j"],
        [
            tat.Edge(symmetry=[torch.tensor([False, True])]),
            tat.Edge(symmetry=[torch.tensor([False, True])]),
        ],
        dtype=torch.float16,
    ).randn_()
    assert a.dtype == torch.float16
    assert a[0, 0] != 0
    assert a[1, 1] != 0
    assert a[0, 1] == 0
    assert a[1, 0] == 0

    b = tat.Tensor(
        ["i", "j"],
        [
            tat.Edge(symmetry=[torch.tensor([False, False]), torch.tensor([0, -1])]),
            tat.Edge(symmetry=[torch.tensor([False, False]), torch.tensor([0, +1])]),
        ],
        dtype=torch.float16,
    ).randn_()
    assert b.dtype == torch.float16
    assert b[0, 0] != 0
    assert b[1, 1] != 0
    assert b[0, 1] == 0
    assert b[1, 0] == 0
