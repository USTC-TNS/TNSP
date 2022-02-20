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

import TAT
import tetragono as tet


def measurement(state):
    n = state.Tensor(["I0", "O0"], [2, 2])
    n.blocks[n.names] = [[0, 0], [0, 1]]
    nn = n.edge_rename({"I0": "I1", "O0": "O1"}).contract(n, set())
    return {((al1, al2, 0), (bl1, bl2, 0)): nn for al1 in range(state.L1) for al2 in range(state.L2)
            for bl1 in range(state.L1) for bl2 in range(state.L2) if (al1, al2) != (bl1, bl2)}


def save_result(state, result, step):
    with open("nn.log", "a") as file:
        print(result, file=file)
