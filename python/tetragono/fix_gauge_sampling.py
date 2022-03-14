#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2022 Hao Zhang<zh970205@mail.ustc.edu.cn>
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


def fix_sampling_lattice_guage(state):
    for l1 in range(state.L1):
        for l2 in range(state.L2):
            if l1 != 0 and l1 % 2 == 0:
                fix_gauge_vertical(state, l1 - 1, l2)
    for l1 in range(state.L1):
        for l2 in range(state.L2):
            if l1 != 0 and l1 % 2 == 1:
                fix_gauge_vertical(state, l1 - 1, l2)
    for l1 in range(state.L1):
        for l2 in range(state.L2):
            if l2 != 0 and l2 % 2 == 0:
                fix_gauge_horizontal(state, l1, l2 - 1)
    for l1 in range(state.L1):
        for l2 in range(state.L2):
            if l2 != 0 and l2 % 2 == 1:
                fix_gauge_horizontal(state, l1, l2 - 1)


def fix_gauge_horizontal(state, l1, l2):
    left = state[l1, l2]
    right = state[l1, l2 + 1]
    left_q, left_r = left.qr("r", {"R"}, "R", "L")
    right_q, right_r = right.qr("r", {"L"}, "L", "R")
    big = left_r.contract(right_r, {("R", "L")})
    u, s, v = big.svd({"L"}, "R", "L", "L", "R")
    i = s.same_shape().identity({("L", "R")})
    delta = np.sqrt(np.abs(s.storage))
    delta[delta == 0] = 1
    s.storage /= delta
    i.storage *= delta
    state[l1, l2] = left_q.contract(u, {("R", "L")}).contract(s, {("R", "L")})
    state[l1, l2 + 1] = right_q.contract(v, {("L", "R")}).contract(i, {("L", "R")})


def fix_gauge_vertical(state, l1, l2):
    up = state[l1, l2]
    down = state[l1 + 1, l2]
    up_q, up_r = up.qr("r", {"D"}, "D", "U")
    down_q, down_r = down.qr("r", {"U"}, "U", "D")
    big = up_r.contract(down_r, {("D", "U")})
    u, s, v = big.svd({"U"}, "D", "U", "U", "D")
    i = s.same_shape().identity({("U", "D")})
    delta = np.sqrt(np.abs(s.storage))
    delta[delta == 0] = 1
    s.storage /= delta
    i.storage *= delta
    state[l1, l2] = up_q.contract(u, {("D", "U")}).contract(s, {("D", "U")})
    state[l1 + 1, l2] = down_q.contract(v, {("U", "D")}).contract(i, {("U", "D")})
