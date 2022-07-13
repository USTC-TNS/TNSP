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


def measurement(state):
    Sz = tet.common_tensor.No.Sz.to(float)
    result = {((l1, l2, orbit),): Sz for l1 in range(state.L1) for l2 in range(state.L2)
              for orbit in range(0 if (l1, l2) != (0, 0) else 1, 2 if (l1, l2) != (state.L1 - 1, state.L2 - 1) else 1)}
    return result


def save_result(state, result, step):
    to_print = [
        result[(l1, l2, orbit),][0] for l1 in range(state.L1) for l2 in range(state.L2)
        for orbit in range(0 if (l1, l2) != (0, 0) else 1, 2 if (l1, l2) != (state.L1 - 1, state.L2 - 1) else 1)
    ]
    with open("Sz.log", "a", encoding="utf-8") as file:
        print(*to_print, file=file)
