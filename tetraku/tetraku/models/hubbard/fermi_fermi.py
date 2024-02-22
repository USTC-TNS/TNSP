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


def abstract_state(L1, L2, T, t, U):
    """
    Create Hubbard model state.

    Parameters
    ----------
    L1, L2 : int
        The lattice size.
    T : int
        The half particle number.
    t, U : float
        Hubbard model parameters.
    """
    state = tet.AbstractState(TAT.FermiU1FermiU1.D.Tensor, L1, L2)
    half_T = T // 2
    if half_T * 2 != T:
        raise RuntimeError("T must be even number")
    state.total_symmetry = (half_T, half_T)
    state.physics_edges[...] = [((0, 0), 1), ((0, 1), 1), ((1, 0), 1), ((1, 1), 1)]
    NN = tet.common_tensor.FermiFermi_Hubbard.NN.to(float)
    CSCS = tet.common_tensor.FermiFermi_Hubbard.CSCS.to(float)
    state.hamiltonians["vertical_bond"] = -t * CSCS
    state.hamiltonians["horizontal_bond"] = -t * CSCS
    state.hamiltonians["single_site"] = U * NN
    return state
