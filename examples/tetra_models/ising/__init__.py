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


def abstract_state(L1, L2, J, g):
    """
    Create Ising model state.

    Parameters
    ----------
    L1, L2 : int
        The lattice size.
    J, g : float
        The ising parameter.
    """
    state = tet.AbstractState(TAT.No.D.Tensor, L1, L2)
    state.physics_edges[...] = 2
    sigma_x = tet.common_tensor.No.pauli_x.to(float)
    sigma_z_sigma_z = tet.common_tensor.No.pauli_z_pauli_z.to(float)
    state.hamiltonians["vertical_bond"] = -J * sigma_z_sigma_z
    state.hamiltonians["horizontal_bond"] = -J * sigma_z_sigma_z
    state.hamiltonians["single_site"] = -J * g * sigma_x
    return state


def abstract_lattice(L1, L2, D, J, g):
    """
    Create Ising model lattice.

    Parameters
    ----------
    L1, L2 : int
        The lattice size.
    D : int
        The cut dimension.
    J, g : float
        The ising parameter.
    """
    state = tet.AbstractLattice(abstract_state(L1, L2, J, g))
    state.virtual_bond["R"] = D
    state.virtual_bond["D"] = D
    return state
