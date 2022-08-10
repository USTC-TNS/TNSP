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
from tetragono.common_tensor.tensor_toolkit import rename_io, kronecker_product


def abstract_state(L1, L2, delta, omega, radius):
    """
    Create square Rydberg state.
    see https://arxiv.org/pdf/2112.10790.pdf

    Parameters
    ----------
    L1, L2 : int
        The lattice size.
    delta, omega, radius : float
        Rydberg atom parameters.
    """
    Tensor = TAT.No.D.Tensor
    state = tet.AbstractState(Tensor, L1, L2)
    state.physics_edges[...] = 2

    # hamiltonian
    sigma = tet.common_tensor.No.pauli_x.to(float)
    n = (tet.common_tensor.No.identity.to(float) - tet.common_tensor.No.pauli_z.to(float)) / 2
    single_body_hamiltonian = omega * sigma / 2 - delta * n
    for l1 in range(L1):
        for l2 in range(L2):
            state.hamiltonians[(l1, l2, 0),] = single_body_hamiltonian
    nn = kronecker_product(rename_io(n, [0]), rename_io(n, [1]))
    for al1 in range(L1):
        for al2 in range(L2):
            for bl1 in range(L1):
                for bl2 in range(L2):
                    if (al1, al2, 0) >= (bl1, bl2, 0):
                        continue
                    dl1 = abs(al1 - bl1)
                    dl2 = abs(al2 - bl2)
                    distance = (dl1**2 + dl2**2)**0.5
                    param = (radius / distance)**6
                    state.hamiltonians[(al1, al2, 0), (bl1, bl2, 0)] = omega * param * nn

    return state


def abstract_lattice(L1, L2, D, delta, omega, radius):
    """
    Create square Rydberg lattice.
    see https://arxiv.org/pdf/2112.10790.pdf

    Parameters
    ----------
    L1, L2 : int
        The lattice size.
    D : int
        The cut dimension.
    delta, omega, radius : float
        Rydberg atom parameters.
    """
    state = tet.AbstractLattice(abstract_state(L1, L2, delta, omega, radius))
    state.virtual_bond["R"] = D
    state.virtual_bond["D"] = D
    return state
