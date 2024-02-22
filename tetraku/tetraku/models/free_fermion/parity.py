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


def abstract_state(L1, L2):
    """
    Create free fermion(no spin) state.

    Parameters
    ----------
    L1, L2 : int
        The lattice size.
    """
    state = tet.AbstractState(TAT.FermiZ2.D.Tensor, L1, L2)
    state.physics_edges[...] = tet.common_tensor.Fermi.EF
    CC = tet.common_tensor.Parity.CC.to(float)
    state.hamiltonians["vertical_bond"] = CC
    state.hamiltonians["horizontal_bond"] = CC
    return state


def abstract_lattice(L1, L2, D):
    """
    Create free fermion(no spin) lattice.

    Parameters
    ----------
    L1, L2 : int
        The lattice size.
    D : int
        The cut dimension.
    """
    state = tet.AbstractLattice(abstract_state(L1, L2))
    if D == 1:
        edge = [(False, 1)]
    else:
        D1 = D // 2
        D2 = D - D1
        edge = [(False, D1), (True, D2)]
    state.virtual_bond["R"] = state.virtual_bond["D"] = edge

    return state
