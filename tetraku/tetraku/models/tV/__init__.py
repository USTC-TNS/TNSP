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
from tetragono.common_tensor.tensor_toolkit import rename_io, kronecker_product


def abstract_state(L1, L2, T, t, V):
    """
    Create tV model state.

    Parameters
    ----------
    L1, L2 : int
        The lattice size.
    T : int
        The half particle number.
    t, V : float
        tV model parameters.
    """
    state = tet.AbstractState(TAT.FermiU1.D.Tensor, L1, L2)
    state.total_symmetry = T
    state.physics_edges[...] = [(0, 1), (1, 1)]
    CC = tet.common_tensor.Fermi.CC.to(float)
    N = tet.common_tensor.Fermi.N.to(float)
    NN = kronecker_product(rename_io(N, [0]), rename_io(N, [1]))
    H = -t * CC + V * NN
    state.hamiltonians["vertical_bond"] = H
    state.hamiltonians["horizontal_bond"] = H
    return state


def abstract_lattice(L1, L2, D, T, t, V):
    """
    Create tV model lattice.

    Parameters
    ----------
    L1, L2 : int
        The lattice size.
    D : int
        The cut dimension.
    T : int
        The half particle number.
    t, V : float
        tV model parameters.
    """
    state = tet.AbstractLattice(abstract_state(L1, L2, T, t, V))
    tt = T / state.L1
    for l1 in range(state.L1 - 1):
        Q = int(T * (state.L1 - l1 - 1) / state.L1)
        state.virtual_bond[l1, 0, "D"] = [(Q - 1, D), (Q, D), (Q + 1, D)]
    for l1 in range(state.L1 - 1):
        for l2 in range(1, state.L2):
            state.virtual_bond[l1, l2, "D"] = [(0, D)]
    for l1 in range(state.L1):
        for l2 in range(state.L2 - 1):
            Q = int(tt * (state.L2 - l2 - 1) / state.L2)
            state.virtual_bond[l1, l2, "R"] = [(Q - 1, D), (Q, D), (Q + 1, D)]

    return state
