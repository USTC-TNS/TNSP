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


def create(L1, L2, l1, l2, D, J):
    """
    Create cluster heisenberg lattice.

    Parameters
    ----------
    L1, L2 : int
        The lattice size.
    l1, l2 : int
        The cluster size.
    D : int
        The cut dimension.
    J : float
        The heisenberg parameter.
    """
    edge_pool = [[[] for i2 in range(l2)] for i1 in range(l1)]
    for I1 in range(L1):
        for I2 in range(L2):
            edge_pool[l1 * I1 // L1][l2 * I2 // L2].append((I1, I2))
    edge_map = {}
    state = tet.AbstractState(TAT.No.D.Tensor, l1, l2)
    for i1 in range(l1):
        for i2 in range(l2):
            for i, j in enumerate(edge_pool[i1][i2]):
                edge_map[j] = (i1, i2, i)
                state.physics_edges[(i1, i2, i)] = 2
    SS = tet.common_variable.No.SS.to(float)
    H = -J * SS
    for I1 in range(L1):
        for I2 in range(L2):
            if I1 != 0:
                state.hamiltonians[edge_map[I1 - 1, I2], edge_map[I1, I2]] = H
            if I2 != 0:
                state.hamiltonians[edge_map[I1, I2 - 1], edge_map[I1, I2]] = H
    state = tet.AbstractLattice(state)
    state.virtual_bond["R"] = D
    state.virtual_bond["D"] = D
    state = tet.SimpleUpdateLattice(state)
    return state
