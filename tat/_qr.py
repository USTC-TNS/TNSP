"""
This module implements QR decomposition based on Givens rotation and Householder reflection.
"""

import typing
import torch

# pylint: disable=invalid-name


def _syminvadj(X: torch.Tensor) -> torch.Tensor:
    ret = X + X.H
    ret.diagonal().real[:] *= 1 / 2
    return ret


def _triliminvadjskew(X: torch.Tensor) -> torch.Tensor:
    ret = torch.tril(X - X.H)
    if torch.is_complex(X):
        ret.diagonal().imag[:] *= 1 / 2
    return ret


def _qr_backward(
    Q: torch.Tensor,
    R: torch.Tensor,
    Q_grad: typing.Optional[torch.Tensor],
    R_grad: typing.Optional[torch.Tensor],
) -> typing.Optional[torch.Tensor]:
    # see https://arxiv.org/pdf/2009.10071.pdf section 4.3 and 4.5
    # see pytorch torch/csrc/autograd/FunctionsManual.cpp:linalg_qr_backward
    m = Q.size(0)
    n = R.size(1)

    if Q_grad is not None:
        if R_grad is not None:
            MH = R_grad @ R.H - Q.H @ Q_grad
        else:
            MH = -Q.H @ Q_grad
    else:
        if R_grad is not None:
            MH = R_grad @ R.H
        else:
            return None

    # pylint: disable=no-else-return
    if m >= n:
        # Deep and square matrix
        b = Q @ _syminvadj(torch.triu(MH))
        if Q_grad is not None:
            b = b + Q_grad
        return torch.linalg.solve_triangular(R.H, b, upper=False, left=False)
    else:
        # Wide matrix
        b = Q @ (_triliminvadjskew(-MH))
        result = torch.linalg.solve_triangular(R[:, :m].H, b, upper=False, left=False)
        result = torch.cat((result, torch.zeros([m, n - m], dtype=result.dtype, device=result.device)), dim=1)
        if R_grad is not None:
            result = result + Q @ R_grad
        return result


class CommonQR(torch.autograd.Function):
    """
    Implement the autograd function for QR.
    """

    # pylint: disable=abstract-method

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: typing.Any,
        Q_grad: typing.Optional[torch.Tensor],
        R_grad: typing.Optional[torch.Tensor],
    ) -> typing.Optional[torch.Tensor]:
        # pylint: disable=arguments-differ
        Q, R = ctx.saved_tensors
        return _qr_backward(Q, R, Q_grad, R_grad)


def _normalize_diagonal(a: torch.Tensor) -> torch.Tensor:
    r = torch.sqrt(a.conj() * a)
    return torch.where(
        r == torch.zeros([], dtype=a.dtype, device=a.device),
        torch.ones([], dtype=a.dtype, device=a.device),
        a / r,
    )


def _givens_parameter(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r = torch.sqrt(a.conj() * a + b.conj() * b)
    return torch.where(
        b == torch.zeros([], dtype=a.dtype, device=a.device),
        torch.ones([], dtype=a.dtype, device=a.device),
        a / r,
    ), torch.where(
        b == torch.zeros([], dtype=a.dtype, device=a.device),
        torch.zeros([], dtype=a.dtype, device=a.device),
        b / r,
    )


def _givens_qr(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = A.shape
    k = min(m, n)
    Q = torch.eye(m, dtype=A.dtype, device=A.device)
    R = A.clone(memory_format=torch.contiguous_format)

    # Parallel strategy
    # Every row rotated to the nearest row above
    for g in range(m - 1, 0, -1):
        # rotate R[g, 0], R[g+2, 1], R[g+4, 2], ...
        for i, col in zip(range(g, m, 2), range(n)):
            j = i - 1
            # Rotate inside column col
            # Rotate from row i to row j
            c, s = _givens_parameter(R[j, col], R[i, col])
            Q[i], Q[j] = -s * Q[j] + c * Q[i], c.conj() * Q[j] + s.conj() * Q[i]
            R[i], R[j] = -s * R[j] + c * R[i], c.conj() * R[j] + s.conj() * R[i]
    for g in range(1, k):
        # rotate R[g+1, g], R[g+1+2, g+1], R[g+1+4, g+2], ...
        for i, col in zip(range(g + 1, m, 2), range(g, n)):
            j = i - 1
            # Rotate inside column col
            # Rotate from row i to row j
            c, s = _givens_parameter(R[j, col], R[i, col])
            Q[i], Q[j] = -s * Q[j] + c * Q[i], c.conj() * Q[j] + s.conj() * Q[i]
            R[i], R[j] = -s * R[j] + c * R[i], c.conj() * R[j] + s.conj() * R[i]

    # for j in range(n):
    #     for i in range(j + 1, m):
    #         col = j
    #         # Rotate inside column col
    #         # Rotate from row i to row j
    #         c, s = _givens_parameter(R[j, col], R[i, col])
    #         Q[i], Q[j] = -s * Q[j] + c * Q[i], c.conj() * Q[j] + s.conj() * Q[i]
    #         R[i], R[j] = -s * R[j] + c * R[i], c.conj() * R[j] + s.conj() * R[i]

    # Make diagonal positive
    c = _normalize_diagonal(R.diagonal()).conj()
    Q[:k] *= torch.unsqueeze(c, 1)
    R[:k] *= torch.unsqueeze(c, 1)

    Q, R = Q[:k].H, R[:k]
    return Q, R


class GivensQR(CommonQR):
    """
    Compute the reduced QR decomposition using Givens rotation.
    """

    # pylint: disable=abstract-method

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        A: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # pylint: disable=arguments-differ
        Q, R = _givens_qr(A)
        ctx.save_for_backward(Q, R)
        return Q, R


def _normalize_delta(a: torch.Tensor) -> torch.Tensor:
    norm = a.norm()
    return torch.where(
        norm == torch.zeros([], dtype=a.dtype, device=a.device),
        torch.zeros([], dtype=a.dtype, device=a.device),
        a / norm,
    )


def _reflect_target(x: torch.Tensor) -> torch.Tensor:
    return torch.norm(x) * _normalize_diagonal(x[0])


def _householder_qr(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = A.shape
    k = min(m, n)
    Q = torch.eye(m, dtype=A.dtype, device=A.device)
    R = A.clone(memory_format=torch.contiguous_format)

    for i in range(k):
        x = R[i:, i]
        v = torch.zeros_like(x)
        # For complex matrix, it require <v|x> = <x|v>, i.e. v[0] and x[0] have opposite argument.
        v[0] = _reflect_target(x)
        # Reflect x to v
        delta = _normalize_delta(v - x)
        # H = 1 - 2 |Delta><Delta|
        R[i:] -= 2 * torch.outer(delta, delta.conj() @ R[i:])
        Q[i:] -= 2 * torch.outer(delta, delta.conj() @ Q[i:])

    # Make diagonal positive
    c = _normalize_diagonal(R.diagonal()).conj()
    Q[:k] *= torch.unsqueeze(c, 1)
    R[:k] *= torch.unsqueeze(c, 1)

    Q, R = Q[:k].H, R[:k]
    return Q, R


class HouseholderQR(CommonQR):
    """
    Compute the reduced QR decomposition using Householder reflection.
    """

    # pylint: disable=abstract-method

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        A: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # pylint: disable=arguments-differ
        Q, R = _householder_qr(A)
        ctx.save_for_backward(Q, R)
        return Q, R


givens_qr = GivensQR.apply
householder_qr = HouseholderQR.apply
