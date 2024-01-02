#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2022-2023 Chao Wang<1023649157@qq.com>
# Copyright (C) 2022-2024 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
import os

N = int(os.environ["N"])


def iter_orbit(state):
    for l1, l2 in state.sites():
        if (l1, l2) != (0, 0):
            yield (l1, l2, 0)
    for l1, l2 in state.sites():
        if (l1, l2) != (state.L1 - 1, state.L2 - 1):
            yield (l1, l2, 1)


def initial_configuration(configuration):
    state = configuration.owner

    total_particle = N
    up_particle = total_particle // 2
    down_particle = total_particle - up_particle

    up_pool = {}
    down_pool = {}

    for orbit in iter_orbit(state):
        if up_particle > 0:
            up_particle -= 1
            up_pool[orbit] = True
        else:
            up_pool[orbit] = False

    for orbit in reversed(list(iter_orbit(state))):
        if down_particle > 0:
            down_particle -= 1
            down_pool[orbit] = True
        else:
            down_pool[orbit] = False

    for orbit in iter_orbit(state):
        up = up_pool[orbit]
        down = down_pool[orbit]

        if not up and not down:
            configuration[orbit] = False, 0
        elif up and not down:
            configuration[orbit] = True, 0
        elif not up and down:
            configuration[orbit] = True, 1
        elif up and down:
            configuration[orbit] = False, 1
        else:
            raise RuntimeError("Program should not run here")
    return configuration
