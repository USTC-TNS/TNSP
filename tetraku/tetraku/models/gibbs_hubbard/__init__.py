#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2023-2024 Hao Zhang<zh970205@mail.ustc.edu.cn>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

import TAT
import tetragono as tet
from tetragono.common_tensor.tensor_toolkit import half_reverse


def abstract_state(L1, L2, t, U, mu, side=1):
    """
    Create density matrix of Hubbard model state.

    Parameters
    ----------
    L1, L2 : int
        The lattice size.
    t, U : float
        Hubbard model parameters.
    mu : float
        The chemical potential.
    side : 1 | 2, default=1
        The Hamiltonian should apply to single side or both side of density matrix.
    """
    if side not in [1, 2]:
        raise RuntimeError("side should be either 1 or 2")
    state = tet.AbstractState(TAT.FermiZ2.D.Tensor, L1, L2)
    state.total_symmetry = 0
    for l1 in range(L1):
        for l2 in range(L2):
            state.physics_edges[(l1, l2, 0)] = [(False, 2), (True, 2)]
            state.physics_edges[(l1, l2, 1)] = [(False, 2), (True, 2)]
    NN = tet.common_tensor.Parity_Hubbard.NN.to(float)
    N0 = tet.common_tensor.Parity_Hubbard.N0.to(float)
    N1 = tet.common_tensor.Parity_Hubbard.N1.to(float)
    CSCS = tet.common_tensor.Parity_Hubbard.CSCS.to(float)
    single_site = U * NN - mu * (N0 + N1)
    tCC = -t * CSCS
    single_site_double_side = [single_site, half_reverse(single_site.conjugate())]
    tCC_double_side = [tCC, half_reverse(tCC.conjugate())]
    for layer in range(side):
        # The hamiltonian in second layer is conjugate and half reverse of the first layer.
        for l1 in range(L1):
            for l2 in range(L2):
                state.hamiltonians[
                    (l1, l2, layer),
                ] = single_site_double_side[layer]
                if l1 != 0:
                    state.hamiltonians[(l1 - 1, l2, layer), (l1, l2, layer)] = tCC_double_side[layer]
                if l2 != 0:
                    state.hamiltonians[(l1, l2 - 1, layer), (l1, l2, layer)] = tCC_double_side[layer]
    return state


def abstract_lattice(L1, L2, t, U, mu, side=1, D=1):
    """
    Create density matrix of Hubbard model lattice.

    Parameters
    ----------
    L1, L2 : int
        The lattice size.
    t, U : float
        Hubbard model parameters.
    mu : float
        The chemical potential.
    side : 1 | 2, default=1
        The Hamiltonian should apply to single side or both side of density matrix.
    D : int, default=1
        The initial virtual dimension in PEPS network.
    """
    state = tet.AbstractLattice(abstract_state(L1, L2, t, U, mu, side=side))
    if D == 1:
        edge = [(False, 1)]
    else:
        D1 = D // 2
        D2 = D - D1
        edge = [(False, D1), (True, D2)]
    state.virtual_bond["R"] = state.virtual_bond["D"] = edge
    return state
