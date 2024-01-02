#!/usr/bin/env python
# -*- coding: utf-8 -*-
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


def sites(state):
    for l1, l2 in state.sites():
        if (l1, l2) != (0, 0):
            yield l1, l2, 0
        if (l1, l2) != (state.L1 - 1, state.L2 - 1):
            yield l1, l2, 1


def bonds_z(state):
    for l1, l2 in state.sites():
        if (l1, l2) not in [(0, 0), (state.L1 - 1, state.L2 - 1)]:
            yield (l1, l2, 0), (l1, l2, 1)
