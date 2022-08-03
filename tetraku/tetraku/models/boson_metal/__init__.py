#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2022 Chao Wang<1023649157@qq.com>
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


def abstract_state(L1, L2, J, K, mu):
    """
    Create boson metal state.

    Parameters
    ----------
    L1, L2 : int
        The lattice size.
    J, K, mu : float
        boson metal parameters.
    """
    Tensor = TAT.No.D.Tensor
    state = tet.AbstractState(Tensor, L1, L2)
    for l1 in range(L1):
        for l2 in range(L2):
            state.physics_edges[l1, l2, 0] = 2
    pauli_x_pauli_x = tet.common_tensor.No.pauli_x_pauli_x.to(float)
    pauli_y_pauli_y = tet.common_tensor.No.pauli_y_pauli_y.to(float)
    pauli_z = tet.common_tensor.No.pauli_z.to(float)
    identity = tet.common_tensor.No.identity.to(float)

    plaq = Tensor(["I0", "O0", "I1", "O1", "I2", "O2", "I3", "O3"], [2, 2, 2, 2, 2, 2, 2, 2]).zero()
    plaq[{"I0": 0, "I1": 1, "I2": 0, "I3": 1, "O0": 1, "O1": 0, "O2": 1, "O3": 0}] = 1
    plaq[{"I0": 1, "I1": 0, "I2": 1, "I3": 0, "O0": 0, "O1": 1, "O2": 0, "O3": 1}] = 1
    plaq = plaq.to(float)

    hop_term = (pauli_x_pauli_x + pauli_y_pauli_y) * 0.5 * J
    n_term = (pauli_z + identity) * 0.5 * (-mu)
    p_term = plaq * K

    for l1 in range(L1):
        for l2 in range(L2):
            state.hamiltonians[(l1, l2, 0),] = n_term
            if l1 != 0:
                state.hamiltonians[(l1 - 1, l2, 0), (l1, l2, 0)] = hop_term
            if l2 != 0:
                state.hamiltonians[(l1, l2 - 1, 0), (l1, l2, 0)] = hop_term
            if l1 != 0 and l2 != 0:
                state.hamiltonians[(l1 - 1, l2 - 1, 0), (l1 - 1, l2, 0), (l1, l2, 0), (l1, l2 - 1, 0)] = p_term

    return state


def abstract_lattice(L1, L2, D, J, K, mu):
    """
    Create boson metal lattice.

    Parameters
    ----------
    L1, L2 : int
        The lattice size.
    D : int
        The cut dimension.
    J, K, mu : float
        boson metal parameters.
    """
    state = tet.AbstractLattice(abstract_state(L1, L2, J, K, mu))
    state.virtual_bond["R"] = D
    state.virtual_bond["D"] = D
    return state
