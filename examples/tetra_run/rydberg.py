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


def create(L1, L2, D, delta, omega, radius):
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
    Tensor = TAT.No.D.Tensor
    state = tet.AbstractState(Tensor, L1, L2)
    state.physics_edges[...] = 2

    # hamiltonian
    sigma = Tensor(["I0", "O0"], [2, 2])
    sigma.blocks[sigma.names] = [[0, 1], [1, 0]]
    n = Tensor(["I0", "O0"], [2, 2])
    n.blocks[n.names] = [[0, 0], [0, 1]]
    single_body_hamiltonian = sigma * omega / 2 - delta * n
    for l1 in range(L1):
        for l2 in range(L2):
            state.hamiltonians[(l1, l2, 0),] = single_body_hamiltonian
    nn = n.edge_rename({"I0": "I1", "O0": "O1"}).contract(n, set())
    for al1 in range(L1):
        for al2 in range(L2):
            for bl1 in range(L1):
                for bl2 in range(L2):
                    if (al1, al2, 0) >= (bl1, bl2, 0):
                        continue
                    distance = ((al1 - bl1)**2 + (al2 - bl2)**2)**0.5
                    param = omega * ((radius / distance)**6)
                    state.hamiltonians[(al1, al2, 0), (bl1, bl2, 0)] = param * nn

    state = tet.AbstractLattice(state)
    state.virtual_bond["R"] = D
    state.virtual_bond["D"] = D
    state = tet.SimpleUpdateLattice(state)
    return state
