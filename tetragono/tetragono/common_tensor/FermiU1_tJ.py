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
from .tensor_toolkit import Fedge, Tedge, rename_io

Tensor = TAT.FermiU1.Z.Tensor

# Empty, Down, Up
EF = Fedge[(0, 0), (+1, -1), (+1, +1)]
ET = Tedge[(0, 0), (-1, +1), (-1, -1)]

CPU = Tensor(["O0", "I0", "T"], [EF, ET, Fedge[(-1, -1),]]).range(1, 0)
CPD = Tensor(["O0", "I0", "T"], [EF, ET, Fedge[(-1, +1),]]).range(1, 0)
CMU = Tensor(["O0", "I0", "T"], [EF, ET, Tedge[(+1, +1),]]).range(1, 0)
CMD = Tensor(["O0", "I0", "T"], [EF, ET, Tedge[(+1, -1),]]).range(1, 0)

C0UC1U = rename_io(CPU, [0]).contract(rename_io(CMU, [1]), {("T", "T")})
C0DC1D = rename_io(CPD, [0]).contract(rename_io(CMD, [1]), {("T", "T")})
C1UC0U = rename_io(CPU, [1]).contract(rename_io(CMU, [0]), {("T", "T")})
C1DC0D = rename_io(CPD, [1]).contract(rename_io(CMD, [0]), {("T", "T")})
CC = C0UC1U + C0DC1D + C1UC0U + C1DC0D

# 2 Si = CP pauli_i CM

# 2 Sx = CPD CMU + CPU CMD
# 2 Sy = i CPD CMU - i CPU CMD
# 2 Sz = CPU CMU - CPD CMD

# 4 SzSz = (CPU CMU - CPD CMD) (CPU CMU - CPD CMD)
# 4 (SxSx + SySy) = (CPD CMU + CPU CMD) (CPD CMU + CPU CMD) - (CPD CMU - CPU CMD)(CPD CMU - CPU CMD)
#                 = 2 CPD CMU CPU CMD + 2 CPU CMD CPD CMU
# 2 (SxSx + SySy) = CPD CMU CPU CMD + CPU CMD CPD CMU
#                 = CPD0 CMU0 CPU1 CMD1 + CPU0 CMD0 CPD1 CMU1
#                 = - CPD0 CMD1 CPU1 CMU0 - CPU0 CMU1 CPD1 CMD0
#                 = - C0DC1D C1UC0U - C0UC1U C1DC0D

Sz2 = CPU.contract(CMU, {("I0", "O0"), ("T", "T")}) - CPD.contract(CMD, {("I0", "O0"), ("T", "T")})
SzSz4 = rename_io(Sz2, [0]).contract(rename_io(Sz2, [1]), set())
SxSxSySy2 = -1 * (C0DC1D.contract(C1UC0U, {
    ("I0", "O0"),
    ("O1", "I1"),
}) + C0UC1U.contract(C1DC0D, {
    ("I0", "O0"),
    ("O1", "I1"),
}))

SS = SzSz4 / 4 + SxSxSySy2 / 2

n = CPU.contract(CMU, {("I0", "O0"), ("T", "T")}) + CPD.contract(CMD, {("I0", "O0"), ("T", "T")})
nn = rename_io(n, [0]).contract(rename_io(n, [1]), set())
