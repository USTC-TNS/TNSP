#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2021-2024 Hao Zhang<zh970205@mail.ustc.edu.cn>
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


def abstract_state(L1, L2, t, U, mu):
    """
    Create Hubbard model state.

    Parameters
    ----------
    L1, L2 : int
        The lattice size.
    t, U : float
        Hubbard model parameters.
    mu: float
        The chemical potential.
    """
    state = tet.AbstractState(TAT.FermiZ2.D.Tensor, L1, L2)
    state.total_symmetry = 0
    state.physics_edges[...] = [(False, 2), (True, 2)]
    NN = tet.common_tensor.Parity_Hubbard.NN.to(float)
    N0 = tet.common_tensor.Parity_Hubbard.N0.to(float)
    N1 = tet.common_tensor.Parity_Hubbard.N1.to(float)
    CSCS = tet.common_tensor.Parity_Hubbard.CSCS.to(float)
    state.hamiltonians["vertical_bond"] = -t * CSCS
    state.hamiltonians["horizontal_bond"] = -t * CSCS
    state.hamiltonians["single_site"] = U * NN - mu * (N0 + N1)
    return state


def abstract_lattice(L1, L2, D, t, U, mu):
    """
    Create Hubbard model lattice.

    Parameters
    ----------
    L1, L2 : int
        The lattice size.
    D : int
        The cut dimension.
    t, U : float
        Hubbard model parameters.
    mu: float
        The chemical potential.
    """
    state = tet.AbstractLattice(abstract_state(L1, L2, t, U, mu))
    D1 = D // 2
    D2 = D - D1
    state.virtual_bond["R"] = state.virtual_bond["D"] = [(False, D1), (True, D2)]

    return state
