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


def abstract_state(L1, L2, Jx, Jy, Jz):
    """
    Create kitaev model state.
    see https://arxiv.org/pdf/cond-mat/0506438

    Parameters
    ----------
    L1, L2 : int
        The lattice size.
    Jx, Jy, Jz : float
        Rydberg atom parameters.
    """
    state = tet.AbstractState(TAT.No.D.Tensor, L1, L2)
    for l1 in range(L1):
        for l2 in range(L2):
            if (l1, l2) != (0, 0):
                state.physics_edges[l1, l2, 0] = 2
            if (l1, l2) != (L1 - 1, L2 - 1):
                state.physics_edges[l1, l2, 1] = 2
    pauli_x_pauli_x = tet.common_tensor.No.pauli_x_pauli_x.to(float)
    pauli_y_pauli_y = tet.common_tensor.No.pauli_y_pauli_y.to(float)
    pauli_z_pauli_z = tet.common_tensor.No.pauli_z_pauli_z.to(float)
    Hx = -Jx * pauli_x_pauli_x
    Hy = -Jy * pauli_y_pauli_y
    Hz = -Jz * pauli_z_pauli_z
    for l1 in range(L1):
        for l2 in range(L2):
            if (l1, l2) not in ((0, 0), (L1 - 1, L2 - 1)):
                state.hamiltonians[(l1, l2, 0), (l1, l2, 1)] = Hx
            if l1 != 0:
                state.hamiltonians[(l1 - 1, l2, 1), (l1, l2, 0)] = Hy
            if l2 != 0:
                state.hamiltonians[(l1, l2 - 1, 1), (l1, l2, 0)] = Hz
    return state


def abstract_lattice(L1, L2, D, Jx, Jy, Jz):
    """
    Create kitaev model lattice.
    see https://arxiv.org/pdf/cond-mat/0506438

    Parameters
    ----------
    L1, L2 : int
        The lattice size.
    D : int
        The cut dimension.
    Jx, Jy, Jz : float
        Rydberg atom parameters.
    """
    state = tet.AbstractLattice(abstract_state(L1, L2, Jx, Jy, Jz))
    state.virtual_bond["R"] = D
    state.virtual_bond["D"] = D
    return state
