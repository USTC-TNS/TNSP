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


def restrict(configuration):
    owner = configuration._owner
    n_up = 0
    n_down = 0
    for l1 in range(owner.L1):
        for l2 in range(owner.L2):
            for orbit, edge in owner.physics_edges[l1, l2].items():
                site_config = configuration[l1, l2, orbit]
                symmetry, index = site_config
                if symmetry.fermi == 2:
                    n_up += 1
                    n_down += 1
                elif symmetry.fermi == 1:
                    if index == 0:
                        n_up += 1
                    else:
                        n_down += 1
    spin_number = owner.total_symmetry.fermi
    return n_up == spin_number // 2
