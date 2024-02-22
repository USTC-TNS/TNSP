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


def abstract_state(L1, L2, t, Delta, mu):
    """
    Create BCS d-wave model state.

    Parameters
    ----------
    L1, L2 : int
        The lattice size.
    t, Delta : float
        BCS model parameters.
    mu : float
        chemical potential.
    """
    state = tet.AbstractState(TAT.FermiZ2.D.Tensor, L1, L2)
    state.physics_edges[...] = [(False, 2), (True, 2)]

    CSCS = tet.common_tensor.Parity_Hubbard.CSCS.to(float)

    N0 = tet.common_tensor.Parity_Hubbard.N0.to(float)
    N1 = tet.common_tensor.Parity_Hubbard.N1.to(float)
    N = N0 + N1

    state.hamiltonians["vertical_bond"] = -t * CSCS + Delta * tet.common_tensor.Parity_Hubbard.singlet.to(float)
    state.hamiltonians["horizontal_bond"] = -t * CSCS - Delta * tet.common_tensor.Parity_Hubbard.singlet.to(float)
    state.hamiltonians["single_site"] = -mu * N
    return state


def abstract_lattice(L1, L2, D, t, Delta, mu):
    """
    Create BCS model lattice.

    Parameters
    ----------
    L1, L2 : int
        The lattice size.
    D : int
        The cut dimension.
    t, Delta : float
        BCS model parameters.
    mu : float
        chemical potential.
    """
    state = tet.AbstractLattice(abstract_state(L1, L2, t, Delta, mu))
    D0 = D // 2
    D1 = D - D0
    state.virtual_bond["R"] = state.virtual_bond["D"] = [(False, D0), (True, D1)]

    return state
