#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2020-2024 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

Tensor = TAT.FermiU1BoseU1.Z.Tensor

EF = Fedge[(0, 0), (+1, +1), (+1, -1), (+2, 0)]
ET = Tedge[(0, 0), (-1, -1), (-1, +1), (-2, 0)]


def _generate_Up_and_Down(_name, _spin):
    CP = Tensor(["O0", "I0", "T"], [EF, ET, Fedge[
        (-1, -_spin),
    ]]).range_(1, 0)
    CM = Tensor(["O0", "I0", "T"], [EF, ET, Tedge[
        (+1, +_spin),
    ]]).range_(1, 0)
    C0C1 = rename_io(CP, [0]).contract(rename_io(CM, [1]), {("T", "T")})
    C1C0 = rename_io(CP, [1]).contract(rename_io(CM, [0]), {("T", "T")})
    CC = C0C1 + C1C0

    I = Tensor(["O0", "I0"], [EF, ET]).identity_({("I0", "O0")})

    N = rename_io(CP, [0]).contract(rename_io(CM, [0]), {("T", "T"), ("I0", "O0")})

    return type(_name, (object,), locals())


Up = _generate_Up_and_Down("Up", +1)
Down = _generate_Up_and_Down("Down", -1)

NN = Up.N.contract(Down.N, {("I0", "O0")})
CSCS = Up.CC + Down.CC
