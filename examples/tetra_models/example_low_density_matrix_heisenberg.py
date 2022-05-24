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


def create(L1, L2, J):
    state = tet.AbstractState(TAT.No.D.Tensor, L1, L2)
    for layer in (0, 1):
        for l1 in range(L1):
            for l2 in range(L2):
                state.physics_edges[(l1, l2, layer)] = 2
    SS = tet.common_tensor.No.SS.to(float)
    JSS = -J * SS
    for layer in (0, 1):
        for l1 in range(L1):
            for l2 in range(L2):
                if l1 != 0:
                    state.hamiltonians[(l1 - 1, l2, layer), (l1, l2, layer)] = JSS
                if l2 != 0:
                    state.hamiltonians[(l1, l2 - 1, layer), (l1, l2, layer)] = JSS
    state = tet.AbstractLattice(state)
    state.virtual_bond["R"] = 1
    state.virtual_bond["D"] = 1
    return state


TAT.random.seed(2333)

L1 = 4
L2 = 4
abstract_lattice = create(L1=L1, L2=L2, J=-1)
su_lattice = tet.SimpleUpdateLattice(abstract_lattice)
for l1 in range(L1):
    for l2 in range(L2):
        su_lattice[l1, l2].zero()
        su_lattice[l1, l2].storage = [1, 0, 0, 1]

su_lattice.update(100, 0.01, new_dimension=5)
