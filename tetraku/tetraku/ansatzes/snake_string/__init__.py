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

import tetragono as tet


def ansatz(state, direction, dimension, base_l1=0, base_l2=0):
    """
    Create an string bond state ansatz along a snake like string.

    This is used for single orbit system.

    Parameters
    ----------
    direction : str
        The direction of this snake like string.
    dimension : int
        The bond dimension of the string.
    base_l1, base_l2 : int
        The start point of the snake string.
    """
    if direction in ["H", "h"]:
        index_to_site = []
        for l1 in range(state.L1):
            for l2 in range(state.L2) if l1 % 2 == 0 else reversed(range(state.L2)):
                index_to_site.append([((l1 + base_l1) % state.L1, (l2 + base_l2) % state.L2, 0)])
        return tet.ansatz_product_state.ansatzes.OpenString(state, index_to_site, dimension)
    elif direction in ["V", "v"]:
        index_to_site = []
        for l2 in range(state.L2):
            for l1 in range(state.L1) if l2 % 2 == 0 else reversed(range(state.L1)):
                index_to_site.append([((l1 + base_l1) % state.L1, (l2 + base_l2) % state.L2, 0)])
        return tet.ansatz_product_state.ansatzes.OpenString(state, index_to_site, dimension)
    else:
        raise RuntimeError("Invalid direction when creating snake string ansatz")
