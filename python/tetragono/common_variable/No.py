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

CTensor = TAT.No.Z.Tensor
Tensor = TAT.No.D.Tensor

Sx = Tensor(["I0", "O0"], [2, 2])
Sx.blocks[Sx.names] = [[0, 0.5], [0.5, 0]]
Sy = CTensor(["I0", "O0"], [2, 2])
Sy.blocks[Sy.names] = [[0, -0.5j], [0.5j, 0]]
Sz = Tensor(["I0", "O0"], [2, 2])
Sz.blocks[Sz.names] = [[0.5, 0], [0, -0.5]]

SxSx = Sx.edge_rename({"I0": "I1", "O0": "O1"}).contract(Sx, set()).to(float)
SySy = Sy.edge_rename({"I0": "I1", "O0": "O1"}).contract(Sy, set()).to(float)
SzSz = Sz.edge_rename({"I0": "I1", "O0": "O1"}).contract(Sz, set()).to(float)

SS = SxSx + SySy + SzSz
