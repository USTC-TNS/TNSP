#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2024 Hao Zhang<zh970205@mail.ustc.edu.cn>
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


def sample_without_replace(number):
    random = TAT.random.uniform_int(0, number - 1)
    pool = set()
    while True:
        i = random()
        if i not in pool:
            yield i
            pool.add(i)


def initial_configuration(configuration, particle_number=None, up_number=None, down_number=None):
    state = configuration.owner
    for l1, l2 in state.sites():
        configuration[l1, l2, 0] = (False, 0)
    if up_number is None and down_number is None:
        for _, i in zip(range(particle_number), sample_without_replace(state.site_number)):
            l1 = i // state.L2
            l2 = i % state.L2
            configuration[l1, l2, 0] = (True, 0)
    elif particle_number is None:
        for _, i in zip(range(up_number), sample_without_replace(state.site_number // 2)):
            i = i * 2
            l1 = i // state.L2
            l2 = i % state.L2
            configuration[l1, l2, 0] = (True, 0)
        for _, i in zip(range(down_number), sample_without_replace(state.site_number // 2)):
            i = i * 2 + 1
            l1 = i // state.L2
            l2 = i % state.L2
            configuration[l1, l2, 0] = (True, 0)
    return configuration
