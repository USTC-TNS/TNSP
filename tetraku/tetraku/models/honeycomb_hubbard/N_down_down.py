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
from tetragono.common_tensor.tensor_toolkit import kronecker_product, rename_io


def measurement(state):
    N_up = tet.common_tensor.Fermi_Hubbard.CUCU.to(float)
    N_down = tet.common_tensor.Fermi_Hubbard.CDCD.to(float)
    NN = kronecker_product(rename_io(N_down, [0]), rename_io(N_down, [1]))

    result = {
        ((al1, al2, aorbit), (bl1, bl2, borbit)): NN for al1 in range(state.L1) for al2 in range(state.L2)
        for aorbit in range(0 if (al1, al2) != (0, 0) else 1, 2 if (al1, al2) != (state.L1 - 1, state.L2 - 1) else 1)
        for bl1 in range(state.L1) for bl2 in range(state.L2)
        for borbit in range(0 if (bl1, bl2) != (0, 0) else 1, 2 if (bl1, bl2) != (state.L1 - 1, state.L2 - 1) else 1)
        if (al1, al2, aorbit) != (bl1, bl2, borbit)
    }
    return result


def save_result(state, result, step):
    with open("N_down_down.log", "a", encoding="utf-8") as file:
        print(result, file=file)
