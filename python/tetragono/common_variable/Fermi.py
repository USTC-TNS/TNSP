#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2020-2021 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

CTensor = TAT.Fermi.Z.Tensor
Tensor = TAT.Fermi.D.Tensor


class FakeEdge:

    def __init__(self, direction):
        self.direction = direction

    def __getitem__(self, x):
        return (list(x), self.direction)


Fedge = FakeEdge(False)
Tedge = FakeEdge(True)

CC = Tensor(["O0", "O1", "I0", "I1"], [Fedge[(0, 1), (1, 1)], Fedge[(0, 1), (1, 1)], Tedge[(0, 1), (-1, 1)], Tedge[(0, 1), (-1, 1)]]).zero()
CC[{"O0": (1, 0), "O1": (0, 0), "I0": (0, 0), "I1": (-1, 0)}] = 1
CC[{"O0": (0, 0), "O1": (1, 0), "I0": (-1, 0), "I1": (0, 0)}] = 1
