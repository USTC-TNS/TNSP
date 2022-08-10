#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2022 Hao Zhang<zh970205@mail.ustc.edu.cn>
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


def abstract_state(L1, L2, delta, omega, U):
    """
    Create kagome rydberg state.

    Parameters
    ----------
    L1, L2 : int
        The lattice size.
    delta, omega, U : float
        The rydberg atom array parameter.
    """
    state = tet.AbstractState(TAT.No.D.Tensor, L1, L2)
    #   ---0----1--
    #   \    /
    #    2  3
    #     \/
    #     /\
    #    4  5
    #   /    \
    for l1 in range(L1):
        for l2 in range(L2):
            if l2 != L2 - 1:
                state.physics_edges[l1, l2, 0] = 2
                state.physics_edges[l1, l2, 1] = 2
            if l1 != L1 - 1:
                state.physics_edges[l1, l2, 2] = 2
            if l1 != L1 - 1 and l2 != L2 - 1:
                state.physics_edges[l1, l2, 3] = 2
            if l1 != L1 - 1 and l2 != 0:
                state.physics_edges[l1, l2, 4] = 2
            if l1 != L1 - 1:
                state.physics_edges[l1, l2, 5] = 2

    sigma = tet.common_tensor.No.pauli_x.to(float)
    n = (tet.common_tensor.No.identity.to(float) - tet.common_tensor.No.pauli_z.to(float)) / 2
    H = -omega * sigma / 2 - delta * n

    for l1 in range(L1):
        for l2 in range(L2):
            if l2 != L2 - 1:
                state.hamiltonians[(l1, l2, 0),] = H
                state.hamiltonians[(l1, l2, 1),] = H
            if l1 != L1 - 1:
                state.hamiltonians[(l1, l2, 2),] = H
            if l1 != L1 - 1 and l2 != L2 - 1:
                state.hamiltonians[(l1, l2, 3),] = H
            if l1 != L1 - 1 and l2 != 0:
                state.hamiltonians[(l1, l2, 4),] = H
            if l1 != L1 - 1:
                state.hamiltonians[(l1, l2, 5),] = H

    nn = kronecker_product(rename_io(n, [0]), rename_io(n, [1]))
    Unn = U * nn
    for l1 in range(L1):
        for l2 in range(L2):
            site_list = []
            if l1 != 0:
                site_list.append((l1 - 1, l2, 5))
            if l2 != 0:
                site_list.append((l1, l2 - 1, 1))
            if l1 != L1 - 1:
                site_list.append((l1, l2, 2))
            if l2 != L2 - 1:
                site_list.append((l1, l2, 0))
            for i in range(len(site_list)):
                for j in range(i + 1, len(site_list)):
                    state.hamiltonians[site_list[i], site_list[j]] = Unn

            site_list = []
            if l1 != L1 - 1:
                site_list.append((l1, l2, 2))
            if l1 != L1 - 1 and l2 != L2 - 1:
                site_list.append((l1, l2, 3))
            if l1 != L1 - 1 and l2 != 0:
                site_list.append((l1, l2, 4))
            if l1 != L1 - 1:
                site_list.append((l1, l2, 5))
            for i in range(len(site_list)):
                for j in range(i + 1, len(site_list)):
                    state.hamiltonians[site_list[i], site_list[j]] = Unn

            site_list = []
            if l1 != 0 and l2 != L2 - 1:
                site_list.append((l1 - 1, l2 + 1, 4))
            if l2 != L2 - 1:
                site_list.append((l1, l2, 0))
            if l2 != L2 - 1:
                site_list.append((l1, l2, 1))
            if l1 != L1 - 1 and l2 != L2 - 1:
                site_list.append((l1, l2, 3))
            for i in range(len(site_list)):
                for j in range(i + 1, len(site_list)):
                    state.hamiltonians[site_list[i], site_list[j]] = Unn

    return state


def abstract_lattice(L1, L2, D, delta, omega, U):
    """
    Create kagome rydberg lattice.

    Parameters
    ----------
    L1, L2 : int
        The lattice size.
    D : int
        The cut dimension.
    delta, omega, U : float
        The rydberg atom array parameter.
    """
    state = tet.AbstractLattice(abstract_state(L1, L2, delta, omega, U))
    state.virtual_bond["R"] = D
    state.virtual_bond["D"] = D
    return state
