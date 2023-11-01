"Test QR"

import torch
from tat._qr import givens_qr, householder_qr

# pylint: disable=missing-function-docstring
# pylint: disable=invalid-name


def check_givens(A: torch.Tensor) -> None:
    m, n = A.size()
    Q, R = givens_qr(A)
    assert torch.allclose(A, Q @ R)
    assert torch.allclose(Q.H @ Q, torch.eye(min(m, n), dtype=A.dtype, device=A.device))
    grad_check = torch.autograd.gradcheck(
        givens_qr,
        A,
        eps=1e-8,
        atol=1e-4,
    )
    assert grad_check


def test_qr_real_givens() -> None:
    check_givens(torch.randn(7, 5, dtype=torch.float64, requires_grad=True))
    check_givens(torch.randn(5, 5, dtype=torch.float64, requires_grad=True))
    check_givens(torch.randn(5, 7, dtype=torch.float64, requires_grad=True))


def test_qr_complex_givens() -> None:
    check_givens(torch.randn(7, 5, dtype=torch.complex128, requires_grad=True))
    check_givens(torch.randn(5, 5, dtype=torch.complex128, requires_grad=True))
    check_givens(torch.randn(5, 7, dtype=torch.complex128, requires_grad=True))


def check_householder(A: torch.Tensor) -> None:
    m, n = A.size()
    Q, R = householder_qr(A)
    assert torch.allclose(A, Q @ R)
    assert torch.allclose(Q.H @ Q, torch.eye(min(m, n), dtype=A.dtype, device=A.device))
    grad_check = torch.autograd.gradcheck(
        householder_qr,
        A,
        eps=1e-8,
        atol=1e-4,
    )
    assert grad_check


def test_qr_real_householder() -> None:
    check_householder(torch.randn(7, 5, dtype=torch.float64, requires_grad=True))
    check_householder(torch.randn(5, 5, dtype=torch.float64, requires_grad=True))
    check_householder(torch.randn(5, 7, dtype=torch.float64, requires_grad=True))


def test_qr_complex_householder() -> None:
    check_householder(torch.randn(7, 5, dtype=torch.complex128, requires_grad=True))
    check_householder(torch.randn(5, 5, dtype=torch.complex128, requires_grad=True))
    check_householder(torch.randn(5, 7, dtype=torch.complex128, requires_grad=True))
