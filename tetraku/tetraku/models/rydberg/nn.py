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

import tetragono as tet
from tetragono.common_tensor.tensor_toolkit import rename_io, kronecker_product


def measurement(state):
    n = (tet.common_tensor.No.identity.to(float) - tet.common_tensor.No.pauli_z.to(float)) / 2
    nn = kronecker_product(rename_io(n, [0]), rename_io(n, [1]))
    return {((al1, al2, 0), (bl1, bl2, 0)): nn for al1 in range(state.L1) for al2 in range(state.L2)
            for bl1 in range(state.L1) for bl2 in range(state.L2) if (al1, al2) != (bl1, bl2)}


def save_result(state, result, step):
    with open("nn.log", "a", encoding="utf-8") as file:
        print(result, file=file)
