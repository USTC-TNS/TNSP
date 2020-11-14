#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

import numpy as np
from TAT.Tensor import DNo as Tensor

max_random = 8

for _ in range(1000):
    rank_A = np.random.randint(3, max_random)
    rank_trace = np.random.randint(1, rank_A // 2 + 1)
    pair_leg = np.random.choice(range(rank_A), [rank_trace, 2], False)

    name_list = [f"A.{i}" for i in range(rank_A)]
    dim_list = np.random.randint(2, max_random, rank_A)
    dim_trace = np.random.randint(2, max_random, rank_trace)
    for i, (j, k) in enumerate(pair_leg):
        dim_list[j] = dim_trace[i]
        dim_list[k] = dim_trace[i]

    trace_conf = {(f"A.{i}", f"A.{j}") for i, j in pair_leg}

    A = Tensor(name_list, dim_list.tolist()).test()
    B = A.trace(trace_conf)

    res = A.block[{}]
    for i in range(rank_trace):
        res = res.trace(0, pair_leg[i, 0], pair_leg[i, 1])
        for j in range(i + 1, rank_trace):
            if pair_leg[j, 0] > pair_leg[i, 0]:
                pair_leg[j, 0] -= 1
            if pair_leg[j, 1] > pair_leg[i, 0]:
                pair_leg[j, 1] -= 1
            if pair_leg[i, 1] > pair_leg[i, 0]:
                pair_leg[i, 1] -= 1
            if pair_leg[j, 0] > pair_leg[i, 1]:
                pair_leg[j, 0] -= 1
            if pair_leg[j, 1] > pair_leg[i, 1]:
                pair_leg[j, 1] -= 1

    diff = res - B.block[{}]

    print(np.max(np.abs(diff)))
