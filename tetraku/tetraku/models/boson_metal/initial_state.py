#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2022 Chao Wang<1023649157@qq.com>
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
import tetragono as tet


def initial_configuration(conf):
    state = conf.owner
    particle = 0
    for L1 in range(state.L1):
        for L2 in range(state.L2):
            if (L1 + L2) % 2 == 0:
                if particle < 37:
                    particle += 1
                    conf[L1, L2, 0] = TAT.No.Symmetry(), 0
                else:
                    conf[L1, L2, 0] = TAT.No.Symmetry(), 1
    for L1 in range(state.L1):
        for L2 in range(state.L2):
            if (L1 + L2) % 2 == 1:
                if particle < 37:
                    particle += 1
                    conf[L1, L2, 0] = TAT.No.Symmetry(), 0
                else:
                    conf[L1, L2, 0] = TAT.No.Symmetry(), 1
    return conf
