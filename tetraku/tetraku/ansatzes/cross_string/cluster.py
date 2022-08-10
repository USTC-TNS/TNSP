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


def ansatz(state, l1, l2, dimension):
    """
    Create an cross like string bond state ansatz

    Parameters
    ----------
    l1, l2 : int
        The cluster size.
    dimension : int
        The bond dimension of the string.
    """
    edge_pool = [[[] for i2 in range(l2)] for i1 in range(l1)]  # tensor index -> site index list
    for I1 in range(state.L1):
        for I2 in range(state.L2):
            edge_pool[l1 * I1 // state.L1][l2 * I2 // state.L2].append((I1, I2))
    ansatzes = []
    for i1 in range(l1):
        index_to_site = []
        for i2 in range(l2):
            index_to_site.append([
                (I1, I2, orbit) for I1, I2 in edge_pool[i1][i2] for orbit in state.physics_edges[I1, I2]
            ])
        ansatzes.append(tet.ansatz_product_state.ansatzes.OpenString(state, index_to_site, dimension))
    for i2 in range(l2):
        index_to_site = []
        for i1 in range(l1):
            index_to_site.append([
                (I1, I2, orbit) for I1, I2 in edge_pool[i1][i2] for orbit in state.physics_edges[I1, I2]
            ])
        ansatzes.append(tet.ansatz_product_state.ansatzes.OpenString(state, index_to_site, dimension))
    return tet.ansatz_product_state.ansatzes.ProductAnsatz(state, ansatzes)
