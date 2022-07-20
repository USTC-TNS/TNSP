#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2022 Chao Wang<1023649157@qq.com>
#                and Hao Zhang<zh970205@mail.ustc.edu.cn>
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


def restrict(configuration, replacement=None):
    if replacement is None:
        owner = configuration.owner
        n_up = 0
        n_down = 0
        for l1 in range(owner.L1):
            for l2 in range(owner.L2):
                for orbit in owner.physics_edges[l1, l2]:
                    site_config = configuration[l1, l2, orbit]
                    _, index = site_config
                    if index == 0:
                        n_up += 1
                    else:
                        n_down += 1
        site_number = owner.site_number
        return n_up == 37
    else:
        n_up_old = 0
        n_down_old = 0
        n_up_new = 0
        n_down_new = 0
        for [l1, l2, orbit], new_site_config in replacement.items():
            old_site_config = configuration[l1, l2, orbit]
            _, old_index = old_site_config
            _, new_index = new_site_config
            if old_index == 0:
                n_up_old += 1
            else:
                n_down_old += 1
            if new_index == 0:
                n_up_new += 1
            else:
                n_down_new += 1
        return n_up_new == n_up_old
