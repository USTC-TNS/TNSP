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
    N_up = tet.common_tensor.Fermi_Hubbard.CUCU.to(float)
    result = {((l1, l2, orbit),): N_up for l1 in range(state.L1) for l2 in range(state.L2)
              for orbit in range(0 if (l1, l2) != (0, 0) else 1, 2 if (l1, l2) != (state.L1 - 1, state.L2 - 1) else 1)}
    return result


def save_result(state, result, step):
    n = [
        result[(l1, l2, orbit),][0] for l1 in range(state.L1) for l2 in range(state.L2)
        for orbit in range(0 if (l1, l2) != (0, 0) else 1, 2 if (l1, l2) != (state.L1 - 1, state.L2 - 1) else 1)
    ]
    with open("N_up.log", "a", encoding="utf-8") as file:
        print(*n, file=file)
    n_error = [
        result[(l1, l2, orbit),][1] for l1 in range(state.L1) for l2 in range(state.L2)
        for orbit in range(0 if (l1, l2) != (0, 0) else 1, 2 if (l1, l2) != (state.L1 - 1, state.L2 - 1) else 1)
    ]
    with open("N_up_error.log", "a", encoding="utf-8") as file:
        print(*n_error, file=file)
