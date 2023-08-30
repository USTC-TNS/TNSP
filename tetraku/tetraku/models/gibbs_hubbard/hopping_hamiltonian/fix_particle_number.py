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
    # This hopping hamiltonian restrict the u1*u1 symmetry in each layer.
    # so it does NOT allow hopping from |2 particle><2 particle| to |1 particle><1 particle|,
    # or hopping from |2><2| to |0><0|. It even also restrict the spin z for each layer.

    hamiltonians = {}

    CSCS = tet.common_tensor.Parity_Hubbard.CSCS.to(float)
    CSCS_double_side = [CSCS, half_reverse(CSCS.conjugate())]

    for l1, l2 in state.sites():
        for layer in range(2):
            # The hamiltonian in second layer is conjugate and half reverse of the first layer.
            if l1 != 0:
                hamiltonians[(l1 - 1, l2, layer), (l1, l2, layer)] = CSCS_double_side[layer]
            if l2 != 0:
                hamiltonians[(l1, l2 - 1, layer), (l1, l2, layer)] = CSCS_double_side[layer]
    return hamiltonians