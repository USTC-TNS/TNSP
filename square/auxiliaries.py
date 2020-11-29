#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

__all__ = ["SquareAuxiliariesSystem"]

CTensor: type = TAT.Tensor.ZNo
Tensor: type = TAT.Tensor.DNo


class SquareAuxiliariesSystem:
    __slots__ = ["_M", "_N", "_dimension_cut", "_lattice", "_auxiliaries"]

    def __init__(self, M: int, N: int, Dc: int):
        self._M: int = M
        self._N: int = N
        self._dimension_cut: int = Dc
        self._lattice: list[list[Tensor]] = [[Tensor(1) for _ in range(self._N)] for _ in range(self._M)]
        self._auxiliaries = {}

    def _set_auxiliaries(self, kind: str):
        # ...
        pass

    def __get_auxiliaries(self, kind: str, i: int, j: int):
        pass
        # check if need to call set_auxiliaries
        # return data from _auxiliaries

    def __setitem__(self, position, value):
        self._lattice[position] = value
        # TODO update aux

    def __getitem__(self, type_and_position):
        pass
        # TODO
