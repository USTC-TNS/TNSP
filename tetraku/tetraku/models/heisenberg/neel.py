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

import TAT


def initial_configuration(conf):
    for l1 in range(conf.owner.L1):
        for l2 in range(conf.owner.L2):
            if (l1 + l2) % 2 == 0:
                conf[l1, l2, 0] = TAT.No.Symmetry(), 0
            else:
                conf[l1, l2, 0] = TAT.No.Symmetry(), 1
    return conf
