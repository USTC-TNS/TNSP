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

import TAT
from square import *

Tensor = TAT(float)

M = 4
N = 4

dimension_virtual = 10

TAT.set_random_seed(0)


def _initialize_tensor_in_network(i: int, j: int):
    name_list = []
    dimension_list = []
    if i != 0:
        name_list.append("U")
        dimension_list.append(dimension_virtual)
    if j != 0:
        name_list.append("L")
        dimension_list.append(dimension_virtual)
    if i != M - 1:
        name_list.append("D")
        dimension_list.append(dimension_virtual)
    if j != N - 1:
        name_list.append("R")
        dimension_list.append(dimension_virtual)
    return Tensor(name_list, dimension_list).randn()


lattice = [[_initialize_tensor_in_network(i, j) for j in range(N)] for i in range(M)]
vector = Tensor(1)
for i in range(M):
    for j in range(N):
        if i == M - 1 and j == N - 1:
            break
        vector = vector.contract(lattice[i][j].edge_rename({"D": f"D-{j}"}), {("R", "L"), (f"D-{j}", "U")})
        vector /= vector.norm_max()

r2 = vector.edge_rename({f"D-{N - 1}": "U0", "R": "L0"})
del vector
r2 /= r2.norm_max()

print(r2)

for Dc in range(2, 100):
    aux = SquareAuxiliariesSystem(M, N, Dc=Dc)
    for i in range(M):
        for j in range(N):
            aux[i, j] = lattice[i][j]

    r1 = aux[((M - 1, N - 1),)]
    r1 /= r1.norm_max()
    print(Dc, (r1 - r2).norm_max())
