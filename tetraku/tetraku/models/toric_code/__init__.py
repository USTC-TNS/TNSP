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


def abstract_state(L1, L2, Jx, Jz):
    """
    Create toric code model state.

    Parameters
    ----------
    L1, L2 : int
        The lattice size.
    Jx, Jz : float
        The toric code parameter.
    """
    state = tet.AbstractState(TAT.No.D.Tensor, L1, L2)
    state.physics_edges[...] = 2
    sigma_x = tet.common_tensor.No.pauli_x.to(float)
    sigma_z = tet.common_tensor.No.pauli_z.to(float)
    sigma_xxxx = kronecker_product(
        rename_io(sigma_x, [0]),
        rename_io(sigma_x, [1]),
        rename_io(sigma_x, [2]),
        rename_io(sigma_x, [3]),
    )
    sigma_zzzz = kronecker_product(
        rename_io(sigma_z, [0]),
        rename_io(sigma_z, [1]),
        rename_io(sigma_z, [2]),
        rename_io(sigma_z, [3]),
    )
    Jsigma_xxxx = -Jx * sigma_xxxx
    Jsigma_zzzz = -Jz * sigma_zzzz
    for l1 in range(state.L1 - 1):
        for l2 in range(state.L2 - 1):
            if (l1 + l2) % 2 == 0:
                state.hamiltonians[(l1, l2, 0), (l1, l2 + 1, 0), (l1 + 1, l2, 0), (l1 + 1, l2 + 1, 0)] = Jsigma_xxxx
            else:
                state.hamiltonians[(l1, l2, 0), (l1, l2 + 1, 0), (l1 + 1, l2, 0), (l1 + 1, l2 + 1, 0)] = Jsigma_zzzz
    return state


def abstract_lattice(L1, L2, D, Jx, Jz):
    """
    Create toric code model lattice.

    Parameters
    ----------
    L1, L2 : int
        The lattice size.
    D : int
        The cut dimension.
    Jx, Jz : float
        The toric code parameter.
    """
    state = tet.AbstractLattice(abstract_state(L1, L2, Jx, Jz))
    state.virtual_bond["R"] = D
    state.virtual_bond["D"] = D
    return state
