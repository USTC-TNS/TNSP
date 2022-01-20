#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2020-2022 Hao Zhang<zh970205@mail.ustc.edu.cn>
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


def measurement(state):
    Sz = state.Tensor(["I0", "O0"], [2, 2])
    Sz.blocks[Sz.names] = [[+1 / 2, 0], [0, -1 / 2]]
    result = {((l1, l2, orbit),): Sz for l1 in range(state.L1) for l2 in range(state.L2)
              for orbit in range(0 if (l1, l2) != (0, 0) else 1, 2 if (l1, l2) != (state.L1 - 1, state.L2 - 1) else 1)}
    print(result)
    return result


def save_result(state, result):
    to_print = [
        result[(l1, l2, orbit),][0] for l1 in range(state.L1) for l2 in range(state.L2)
        for orbit in range(0 if (l1, l2) != (0, 0) else 1, 2 if (l1, l2) != (state.L1 - 1, state.L2 - 1) else 1)
    ]
    tet.common_variable.showln("Sz", *to_print)
