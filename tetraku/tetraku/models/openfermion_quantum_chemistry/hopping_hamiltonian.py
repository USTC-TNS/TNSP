#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2024 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
from tetragono.common_tensor.tensor_toolkit import rename_io, kronecker_product

Tensor = TAT.FermiZ2.D.Tensor
EF = ([(False, 1), (True, 1)], False)
ET = ([(False, 1), (True, 1)], True)

CP = Tensor(["O0", "I0", "T"], [EF, ET, ([(True, 1)], False)]).zero()
CP[{"O0": (True, 0), "I0": (False, 0), "T": (True, 0)}] = 1
CM = Tensor(["O0", "I0", "T"], [EF, ET, ([(True, 1)], True)]).zero()
CM[{"O0": (False, 0), "I0": (True, 0), "T": (True, 0)}] = 1
I = Tensor(["O0", "I0"], [EF, ET]).identity({("I0", "O0")})
C0daggerC1 = rename_io(CP, [0]).contract(rename_io(CM, [1]), {("T", "T")})
C1daggerC0 = rename_io(CP, [1]).contract(rename_io(CM, [0]), {("T", "T")})
CC = C0daggerC1 + C1daggerC0


def hopping_hamiltonians(state):
    hamiltonians = {}
    for s0, _ in state.physics_edges:
        for s1, _ in state.physics_edges:
            if s0 < s1:
                hamiltonians[s0, s1] = CC
    return hamiltonians
