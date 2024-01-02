#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2022-2023 Chao Wang<1023649157@qq.com>
# Copyright (C) 2023-2024 Hao Zhang<zh970205@mail.ustc.edu.cn>
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


def initial_configuration(configuration):
    # This function initialize the configuration randomly but ensures
    # 1. It is the major term in the infinite temperature.
    # 2. It is half-filling configuration without double occupancy.
    # 3. The total spin z is zero, i.e, the particle numbers of spin up and spin down equal.
    state = configuration.owner
    for l1, l2 in state.sites():
        configuration[l1, l2, 0] = configuration[l1, l2, 1] = (True, 0)
    pool = set()
    random_l1 = TAT.random.uniform_int(0, state.L1 - 1)
    random_l2 = TAT.random.uniform_int(0, state.L2 - 1)
    while len(pool) < state.L1 * state.L2 // 2:
        l1 = random_l1()
        l2 = random_l2()
        if (l1, l2) not in pool:
            pool.add((l1, l2))
            configuration[l1, l2, 0] = configuration[l1, l2, 1] = (True, 1)
    return configuration
