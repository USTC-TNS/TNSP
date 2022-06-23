#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2021-2022 Hao Zhang<zh970205@mail.ustc.edu.cn>
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


def abstract_state(L1, L2, J):
    """
    Create density matrix of a heisenberg state.

    Parameters
    ----------
    L1, L2 : int
        The lattice size.
    J : float
        The heisenberg parameter.
    """
    state = tet.AbstractState(TAT.No.D.Tensor, L1, L2)
    for l1 in range(L1):
        for l2 in range(L2):
            for layer in (0, 1):
                state.physics_edges[(l1, l2, layer)] = 2
    SS = tet.common_tensor.No.SS.to(float)
    JSS = -J * SS
    for l1 in range(L1):
        for l2 in range(L2):
            if l1 != 0:
                state.hamiltonians[(l1 - 1, l2, 0), (l1, l2, 0)] = JSS
            if l2 != 0:
                state.hamiltonians[(l1, l2 - 1, 0), (l1, l2, 0)] = JSS
    return state


def abstract_lattice(L1, L2, D, J):
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
    """
    state = tet.AbstractLattice(abstract_state(L1, L2, J))
    state.virtual_bond["R"] = D
    state.virtual_bond["D"] = D
    return state
