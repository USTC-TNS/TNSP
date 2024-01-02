#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2021-2024 Hao Zhang<zh970205@mail.ustc.edu.cn>
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


def abstract_state(L1, L2, J, side=1):
    """
    Create density matrix of a heisenberg state.

    Parameters
    ----------
    L1, L2 : int
        The lattice size.
    J : float
        The heisenberg parameter.
    side : 1 | 2, default=1
        The Hamiltonian should apply to single side or both side of density matrix.
    """
    if side not in [1, 2]:
        raise RuntimeError("side should be either 1 or 2")
    state = tet.AbstractState(TAT.No.D.Tensor, L1, L2)
    for l1 in range(L1):
        for l2 in range(L2):
            for layer in (0, 1):
                state.physics_edges[(l1, l2, layer)] = 2
    SS = tet.common_tensor.No.SS.to(float)
    JSS = -J * SS
    for layer in range(side):
        # Hamiltonian for the second layer should be transposed
        # (transpose but not conjugate, or conjugate but not transpose),
        # But the hamiltonian is real, so nothing to do here
        for l1 in range(L1):
            for l2 in range(L2):
                if l1 != 0:
                    state.hamiltonians[(l1 - 1, l2, layer), (l1, l2, layer)] = JSS
                if l2 != 0:
                    state.hamiltonians[(l1, l2 - 1, layer), (l1, l2, layer)] = JSS
    return state


def abstract_lattice(L1, L2, D, J, side=1):
    """
    Create density matrix of a heisenberg lattice.

    Parameters
    ----------
    L1, L2 : int
        The lattice size.
    D : int
        The cut dimension.
    J : float
        The heisenberg parameter.
    side : 1 | 2, default=1
        The Hamiltonian should apply to single side or both side of density matrix.
    """
    state = tet.AbstractLattice(abstract_state(L1, L2, J, side=side))
    state.virtual_bond["R"] = D
    state.virtual_bond["D"] = D
    return state
