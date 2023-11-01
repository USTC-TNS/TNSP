"Test SVD"

import torch
from tat._svd import svd

# pylint: disable=missing-function-docstring
# pylint: disable=invalid-name


def svd_func(A: torch.Tensor) -> torch.Tensor:
    U, S, V = svd(A, 1e-10)
    return U @ torch.diag(S).to(dtype=A.dtype) @ V


def check_svd(A: torch.Tensor) -> None:
    m, n = A.size()
    U, S, V = svd(A, 1e-10)
    assert torch.allclose(U @ torch.diag(S.to(dtype=A.dtype)) @ V, A)
    assert torch.allclose(U.H @ U, torch.eye(min(m, n), dtype=A.dtype, device=A.device))
    assert torch.allclose(V @ V.H, torch.eye(min(m, n), dtype=A.dtype, device=A.device))
    grad_check = torch.autograd.gradcheck(
        svd_func,
        A,
        eps=1e-8,
        atol=1e-4,
        nondet_tol=1e-10,
    )
    assert grad_check


def test_svd_real() -> None:
    check_svd(torch.randn(7, 5, dtype=torch.float64, requires_grad=True))
    check_svd(torch.randn(5, 5, dtype=torch.float64, requires_grad=True))
    check_svd(torch.randn(5, 7, dtype=torch.float64, requires_grad=True))


def test_svd_complex() -> None:
    check_svd(torch.randn(7, 5, dtype=torch.complex128, requires_grad=True))
    check_svd(torch.randn(5, 5, dtype=torch.complex128, requires_grad=True))
    check_svd(torch.randn(5, 7, dtype=torch.complex128, requires_grad=True))
