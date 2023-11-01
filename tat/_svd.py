"""
This module implements SVD decomposition without Householder reflection.
"""

import typing
import torch
from ._qr import _normalize_diagonal, _givens_parameter

# pylint: disable=invalid-name


def _svd(A: torch.Tensor, error: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements
    # pylint: disable=too-many-nested-blocks

    # see https://web.stanford.edu/class/cme335/lecture6.pdf
    m, n = A.shape
    trans = False
    if m < n:
        trans = True
        A = A.transpose(0, 1)
        m, n = n, m
    U = torch.eye(m, dtype=A.dtype, device=A.device)
    V = torch.eye(n, dtype=A.dtype, device=A.device)

    # Make bidiagonal matrix
    B = A.clone(memory_format=torch.contiguous_format)
    for i in range(n):
        # (i:, i)
        for j in range(m - 1, i, -1):
            col = i
            # Rotate inside col i
            # Rotate from row j to j-1
            c, s = _givens_parameter(B[j - 1, col], B[j, col])
            U[j], U[j - 1] = -s * U[j - 1] + c * U[j], c.conj() * U[j - 1] + s.conj() * U[j]
            B[j], B[j - 1] = -s * B[j - 1] + c * B[j], c.conj() * B[j - 1] + s.conj() * B[j]
        # x = B[i:, i]
        # v = torch.zeros_like(x)
        # v[0] = _reflect_target(x)
        # delta = _normalize_delta(v - x)
        # B[i:, :] -= 2 * torch.outer(delta, delta.conj() @ B[i:, :])
        # U[i:, :] -= 2 * torch.outer(delta, delta.conj() @ U[i:, :])

        # (i, i+1:)/H
        if i == n - 1:
            break
        for j in range(n - 1, i + 1, -1):
            row = i
            # Rotate inside row i
            # Rotate from col j to j-1
            c, s = _givens_parameter(B[row, j - 1], B[row, j])
            V[j], V[j - 1] = -s * V[j - 1] + c * V[j], c.conj() * V[j - 1] + s.conj() * V[j]
            B[:, j], B[:, j - 1] = -s * B[:, j - 1] + c * B[:, j], c.conj() * B[:, j - 1] + s.conj() * B[:, j]
        # x = B[i, i + 1:]
        # v = torch.zeros_like(x)
        # v[0] = _reflect_target(x)
        # delta = _normalize_delta(v - x)
        # B[:, i + 1:] -= 2 * torch.outer(B[:, i + 1:] @ delta.conj(), delta)
        # V[i + 1:, :] -= 2 * torch.outer(delta, delta.conj() @ V[i + 1:, :])
    B = B[:n]
    U = U[:n]
    # print(B)
    # error_decomp = torch.max(torch.abs(U.H @ B @ V.H.T - A)).item()
    # assert error_decomp < 1e-4

    # QR iteration with implicit Q
    S = torch.diagonal(B).clone(memory_format=torch.contiguous_format)
    F = torch.diagonal(B, offset=1).clone(memory_format=torch.contiguous_format)
    F.resize_(S.size(0))
    F[-1] = 0
    X = F[-1]
    stack: list[tuple[int, int]] = [(0, n - 1)]
    while stack:
        # B.zero_()
        # B.diagonal()[:] = S
        # B.diagonal(offset = 1)[:] = F[:-1]
        # error_decomp = torch.max(torch.abs(U.H @ B @ V.H.T - A)).item()
        # assert error_decomp < 1e-4

        low = stack[-1][0]
        high = stack[-1][1]

        if low == high:
            stack.pop()
            continue

        max_diagonal = torch.abs(S[low])
        for b in range(low, high + 1):
            Sb = torch.abs(S[b])
            if Sb < max_diagonal:
                max_diagonal = Sb
            # Check if S[b] is zero
            if Sb < error:
                # pylint: disable=no-else-continue
                if b == low:
                    X = F[b].clone()
                    F[b] = 0
                    for i in range(b + 1, high + 1):
                        c, s = _givens_parameter(S[i], X)
                        U[b], U[i] = -s * U[i] + c * U[b], c.conj() * U[i] + s.conj() * U[b]

                        S[i] = c.conj() * S[i] + s.conj() * X
                        if i != high:
                            X, F[i] = -s * F[i] + c * X, c.conj() * F[i] + s.conj() * X
                    stack.pop()
                    stack.append((b + 1, high))
                    stack.append((low, b))
                    continue
                else:
                    X = F[b - 1].clone()
                    F[b - 1] = 0
                    for i in range(b - 1, low - 1, -1):
                        c, s = _givens_parameter(S[i], X)
                        V[b], V[i] = -s * V[i] + c * V[b], c.conj() * V[i] + s.conj() * V[b]

                        S[i] = c.conj() * S[i] + s.conj() * X
                        if i != low:
                            X, F[i - 1] = -s * F[i - 1] + c * X, c.conj() * F[i - 1] + s.conj() * X
                    stack.pop()
                    stack.append((b, high))
                    stack.append((low, b - 1))
                    continue

        b = int(torch.argmin(torch.abs(F[low:high]))) + low
        if torch.abs(F[b]) < max_diagonal * error:
            F[b] = 0
            stack.pop()
            stack.append((b + 1, high))
            stack.append((low, b))
            continue

        tdn = (S[b + 1].conj() * S[b + 1] + F[b].conj() * F[b]).real
        tdn_1 = (S[b].conj() * S[b] + F[b - 1].conj() * F[b - 1]).real
        tsn_1 = F[b].conj() * S[b]
        d = (tdn_1 - tdn) / 2
        mu = tdn + d - torch.sign(d) * torch.sqrt(d**2 + tsn_1.conj() * tsn_1)
        for i in range(low, high):
            if i == low:
                c, s = _givens_parameter(S[low].conj() * S[low] - mu, S[low].conj() * F[low])
            else:
                c, s = _givens_parameter(F[i - 1], X)
            V[i + 1], V[i] = -s * V[i] + c * V[i + 1], c.conj() * V[i] + s.conj() * V[i + 1]
            if i != low:
                F[i - 1] = c.conj() * F[i - 1] + s.conj() * X
            F[i], S[i] = -s * S[i] + c * F[i], c.conj() * S[i] + s.conj() * F[i]
            S[i + 1], X = c * S[i + 1], s.conj() * S[i + 1]

            c, s = _givens_parameter(S[i], X)
            U[i + 1], U[i] = -s * U[i] + c * U[i + 1], c.conj() * U[i] + s.conj() * U[i + 1]

            S[i] = c.conj() * S[i] + s.conj() * X
            S[i + 1], F[i] = -s * F[i] + c * S[i + 1], c.conj() * F[i] + s.conj() * S[i + 1]
            if i != high - 1:
                F[i + 1], X = c * F[i + 1], s.conj() * F[i + 1]

    # Make diagonal positive
    c = _normalize_diagonal(S).conj()
    V *= c.unsqueeze(1)  # U is larger than V
    S *= c
    S = S.real

    # Sort
    S, order = torch.sort(S, descending=True)
    U = U[order]
    V = V[order]

    # pylint: disable=no-else-return
    if trans:
        return V.H, S, U.H.T
    else:
        return U.H, S, V.H.T


def _skew(A: torch.Tensor) -> torch.Tensor:
    return A - A.H


def _svd_backward(
    U: torch.Tensor,
    S: torch.Tensor,
    Vh: torch.Tensor,
    gU: typing.Optional[torch.Tensor],
    gS: typing.Optional[torch.Tensor],
    gVh: typing.Optional[torch.Tensor],
) -> typing.Optional[torch.Tensor]:
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-arguments

    # See pytorch torch/csrc/autograd/FunctionsManual.cpp:svd_backward
    if gS is None and gU is None and gVh is None:
        return None

    m = U.size(0)
    n = Vh.size(1)

    if gU is None and gVh is None:
        assert gS is not None
        # pylint: disable=no-else-return
        if m >= n:
            return U @ (gS.unsqueeze(1) * Vh)
        else:
            return (U * gS.unsqueeze(0)) @ Vh

    is_complex = torch.is_complex(U)

    UhgU = _skew(U.H @ gU) if gU is not None else None
    VhgV = _skew(Vh @ gVh.H) if gVh is not None else None

    S2 = S * S
    E = S2.unsqueeze(0) - S2.unsqueeze(1)
    E.diagonal()[:] = 1

    if gU is not None:
        if gVh is not None:
            assert UhgU is not None
            assert VhgV is not None
            gA = (UhgU * S.unsqueeze(0) + S.unsqueeze(1) * VhgV) / E
        else:
            assert UhgU is not None
            gA = (UhgU / E) * S.unsqueeze(0)
    else:
        assert VhgV is not None
        gA = S.unsqueeze(1) * (VhgV / E)

    if gS is not None:
        gA = gA + torch.diag(gS)

    if is_complex and gU is not None and gVh is not None:
        assert UhgU is not None
        gA = gA + torch.diag(UhgU.diagonal() / (2 * S))

    if m > n and gU is not None:
        gA = U @ gA
        gUSinv = gU / S.unsqueeze(0)
        gA = gA + gUSinv - U @ (U.H @ gUSinv)
        gA = gA @ Vh
    elif m < n and gVh is not None:
        gA = gA @ Vh
        SinvgVh = gVh / S.unsqueeze(1)
        gA = gA + SinvgVh - (SinvgVh @ Vh.H) @ Vh
        gA = U @ gA
    elif m >= n:
        gA = U @ (gA @ Vh)
    else:
        gA = (U @ gA) @ Vh

    return gA


class SVD(torch.autograd.Function):
    """
    Compute SVD decomposition without Householder reflection.
    """

    # pylint: disable=abstract-method

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        A: torch.Tensor,
        error: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # pylint: disable=arguments-differ
        U, S, V = _svd(A, error)
        ctx.save_for_backward(U, S, V)
        return U, S, V

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: typing.Any,
        U_grad: typing.Optional[torch.Tensor],
        S_grad: typing.Optional[torch.Tensor],
        V_grad: typing.Optional[torch.Tensor],
    ) -> tuple[typing.Optional[torch.Tensor], None]:
        # pylint: disable=arguments-differ
        U, S, V = ctx.saved_tensors
        return _svd_backward(U, S, V, U_grad, S_grad, V_grad), None


svd = SVD.apply
