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

from .sc import get_triplet


def measurement(state):
    bonds = []
    L1 = state.L1
    L2 = state.L2
    row_selected_1 = L1 // 2
    row_selected_2 = row_selected_1 - 1
    for l1 in [row_selected_1, row_selected_2]:
        for l2 in range(L2):
            if (l1, l2) != (0, 0) and (l1, l2) != (L1 - 1, L2 - 1):
                bonds.append(((l1, l2, 0), (l1, l2, 1)))
            if l1 != 0:
                bonds.append(((l1 - 1, l2, 1), (l1, l2, 0)))
            if l2 != 0:
                bonds.append(((l1, l2 - 1, 1), (l1, l2, 0)))
    bond_number = len(bonds)

    result = {}
    for i in range(bond_number):
        for j in range(i, bond_number):
            s0, s1 = bonds[i]
            s2, s3 = bonds[j]
            link = []
            site = [s0, s1]
            if s0 == s2:
                link.append((0, 2))
            elif s1 == s2:
                link.append((1, 2))
            else:
                site.append(s2)
            if s0 == s3:
                link.append((0, 3))
            elif s1 == s3:
                link.append((1, 3))
            else:
                site.append(s3)
            result[tuple(site)] = get_triplet(*link)
    return result


def save_result(state, result, step):
    with open("triplet-2row.log", "a", encoding="utf-8") as file:
        print(result, file=file)
