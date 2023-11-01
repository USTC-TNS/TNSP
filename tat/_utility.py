"""
Some internal utilities used by tat.
"""

import torch

# pylint: disable=missing-function-docstring
# pylint: disable=no-else-return


def unsqueeze(tensor: torch.Tensor, index: int, rank: int) -> torch.Tensor:
    return tensor.reshape([-1 if i == index else 1 for i in range(rank)])


def neg_symmetry(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype is torch.bool:
        return tensor
    else:
        return -tensor


def add_symmetry(tensor_1: torch.Tensor, tensor_2: torch.Tensor) -> torch.Tensor:
    if tensor_1.dtype is torch.bool:
        return tensor_1 ^ tensor_2
    else:
        return tensor_1 + tensor_2


def zero_symmetry(tensor: torch.Tensor) -> torch.Tensor:
    return tensor == torch.zeros([], dtype=tensor.dtype)


def parity(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype is torch.bool:
        return tensor
    else:
        return tensor % 2 != 0
