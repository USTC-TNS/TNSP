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


def ansatz(state, dimension):
    """
    Create an cross like string bond state ansatz

    Parameters
    ----------
    dimension : int
        The bond dimension of the string.
    """
    up_and_left = [*((0, l2) for l2 in range(state.L2)), *((l1, 0) for l1 in range(1, state.L1))]
    up_and_right = [*((0, l2) for l2 in range(state.L2)), *((l1, state.L2 - 1) for l1 in range(1, state.L1))]
    ansatzes = []
    for l1, l2 in up_and_left:
        i1 = l1
        i2 = l2
        index_to_site = []
        while i1 != state.L1 and i2 != state.L2:
            index_to_site.append([(i1, i2, orbit) for orbit in state.physics_edges[i1, i2]])
            i1 += 1
            i2 += 1
        if len(index_to_site) > 1:
            ansatzes.append(tet.ansatz_product_state.ansatzes.OpenString(state, index_to_site, dimension))
    for l1, l2 in up_and_right:
        i1 = l1
        i2 = l2
        index_to_site = []
        while i1 != state.L1 and i2 != -1:
            index_to_site.append([(i1, i2, orbit) for orbit in state.physics_edges[i1, i2]])
            i1 += 1
            i2 -= 1
        if len(index_to_site) > 1:
            ansatzes.append(tet.ansatz_product_state.ansatzes.OpenString(state, index_to_site, dimension))
    return tet.ansatz_product_state.ansatzes.ProductAnsatz(state, ansatzes)
