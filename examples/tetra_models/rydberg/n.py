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
    n = (tet.common_tensor.No.identity.to(float) - tet.common_tensor.No.pauli_z.to(float)) / 2
    return {((l1, l2, 0),): n for l1 in range(state.L1) for l2 in range(state.L2)}


def save_result(state, result, step):
    n = [result[(l1, l2, 0),][0] for l1 in range(state.L1) for l2 in range(state.L2)]
    with open("n.log", "a", encoding="utf-8") as file:
        print(*n, file=file)
    n_error = [result[(l1, l2, 0),][1] for l1 in range(state.L1) for l2 in range(state.L2)]
    with open("n_error.log", "a", encoding="utf-8") as file:
        print(*n_error, file=file)
