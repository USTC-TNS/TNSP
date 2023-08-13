#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2022-2023 Chao Wang<1023649157@qq.com>
# Copyright (C) 2023 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
from tetragono.common_tensor.Parity import *
from tetragono.common_tensor.tensor_toolkit import rename_io, kronecker_product, half_reverse


def hopping_hamiltonians(state):
    # Two part, normal Hamiltonian and hopping between subspace
    hamiltonians = {}

    CM2 = rename_io(CM, [0]).contract(rename_io(CM, [1]), {("T", "T")})
    CP2 = rename_io(CP, [0]).contract(rename_io(CP, [1]), {("T", "T")})
    CCCC = (C0C1 + C1C0 + CM2 + CP2).merge_edge({"I0": ["I0", "I1"], "O0": ["O0", "O1"]})
    between_subspace = kronecker_product(rename_io(CCCC, [0]), rename_io(CCCC, [1]))

    CSCS = tet.common_tensor.Parity_Hubbard.CSCS.to(float)
    CSCS_double_side = [CSCS, half_reverse(CSCS.conjugate())]

    for l1, l2 in state.sites():
        hamiltonians[(l1, l2, 0), (l1, l2, 1)] = between_subspace
        for layer in range(2):
            # The hamiltonian in second layer is conjugate and half reverse of the first layer.
            if l1 != 0:
                hamiltonians[(l1 - 1, l2, layer), (l1, l2, layer)] = CSCS_double_side[layer]
            if l2 != 0:
                hamiltonians[(l1, l2 - 1, layer), (l1, l2, layer)] = CSCS_double_side[layer]
    return hamiltonians
