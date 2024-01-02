#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2021-2024 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

import tetragono as tet
from tetragono.common_tensor.tensor_toolkit import kronecker_product, rename_io
from .tools import bonds_z


def measurement(state):
    Sz = tet.common_tensor.No.Sz.to(float)
    SzSzSzSz = kronecker_product(rename_io(Sz, [0]), rename_io(Sz, [1]), rename_io(Sz, [2]), rename_io(Sz, [3]))
    result = {(*bond1, *bond2): SzSzSzSz for bond1 in bonds_z(state) for bond2 in bonds_z(state) if bond1 != bond2}
    return result


def save_result(state, result, whole_result):
    with open("BzBz.log", "a", encoding="utf-8") as file:
        print(result, file=file)
