#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2022-2023 Chao Wang<1023649157@qq.com>
# Copyright (C) 2022-2024 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

import os

N = int(os.environ["N"])


class Count:

    def __init__(self, owner):
        self.owner = owner

        Symmetry = owner.Tensor.model.Symmetry
        self.c0 = Symmetry(False), 0
        self.c1 = Symmetry(True), 0
        self.c2 = Symmetry(True), 1
        self.c3 = Symmetry(False), 1

        self.up = 0
        self.down = 0

    def __call__(self, config):
        if config == self.c0:
            pass
        elif config == self.c1:
            self.up += 1
        elif config == self.c2:
            self.down += 1
        elif config == self.c3:
            self.up += 1
            self.down += 1
        else:
            raise RuntimeError("Invalid config")


def restrict(configuration, replacement=None):
    if replacement is None:
        owner = configuration.owner
        count = Count(owner)
        for l1, l2 in owner.sites():
            if (l1, l2) != (0, 0):
                count(configuration[l1, l2, 0])
            if (l1, l2) != (owner.L1 - 1, owner.L2 - 1):
                count(configuration[l1, l2, 1])
        return count.up + count.down == N
    else:
        # Replace is always valid
        return True
