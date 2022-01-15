#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2020-2022 Hao Zhang<zh970205@mail.ustc.edu.cn>
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


def create(L1, L2, D, Jx, Jy, Jz):
    state = tet.AbstractState(TAT.No.Z.Tensor, L1, L2)
    for l1 in range(L1):
        for l2 in range(L2):
            if not (l1, l2) == (0, 0):
                state.physics_edges[l1, l2, 0] = 2
            if not (l1, l2) == (L1 - 1, L2 - 1):
                state.physics_edges[l1, l2, 1] = 2
    SxSx = tet.common_variable.No.SxSx.to(complex)
    SySy = tet.common_variable.No.SySy.to(complex)
    SzSz = tet.common_variable.No.SzSz.to(complex)
    for l1 in range(L1):
        for l2 in range(L2):
            if not ((l1, l2) == (0, 0) or (l1, l2) == (L1 - 1, L2 - 1)):
                state.hamiltonians[(l1, l2, 0), (l1, l2, 1)] = SxSx * Jx
            if l1 != 0:
                state.hamiltonians[(l1 - 1, l2, 1), (l1, l2, 0)] = SySy * Jy
            if l2 != 0:
                state.hamiltonians[(l1, l2 - 1, 1), (l1, l2, 0)] = SzSz * Jz
    state = tet.AbstractLattice(state)
    state.virtual_bond["R"] = D
    state.virtual_bond["D"] = D
    state = tet.SimpleUpdateLattice(state)
    return state
