#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2023-2024 Hao Zhang<zh970205@mail.ustc.edu.cn>
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


def abstract_state(L1, L2, t, V):
    """
    Create tV model state.

    Parameters
    ----------
    L1, L2 : int
        The lattice size.
    t, V : float
        tV model parameters.
    """
    state = tet.AbstractState(TAT.FermiZ2.D.Tensor, L1, L2)
    state.total_symmetry = False
    state.physics_edges[...] = [(False, 1), (True, 1)]
    CC = tet.common_tensor.Parity.CC.to(float)
    N = tet.common_tensor.Parity.N.to(float)
    NN = kronecker_product(rename_io(N, [0]), rename_io(N, [1]))
    H = -t * CC + V * NN
    state.hamiltonians["vertical_bond"] = H
    state.hamiltonians["horizontal_bond"] = H
    return state


def abstract_lattice(L1, L2, D, t, V):
    """
    Create tV model lattice.

    Parameters
    ----------
    L1, L2 : int
        The lattice size.
    D : int
        The cut dimension.
    t, V : float
        tV model parameters.
    """
    state = tet.AbstractLattice(abstract_state(L1, L2, t, V))
    for l1, l2 in state.sites():
        if l1 != 0:
            state.virtual_bond[l1, l2, "U"] = [(False, D), (True, D)]
        if l2 != 0:
            state.virtual_bond[l1, l2, "L"] = [(False, D), (True, D)]

    return state
