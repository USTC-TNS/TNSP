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

from __future__ import annotations
from multimethod import multimethod
import TAT
from .abstract_network_lattice import AbstractNetworkLattice
from .auxiliaries import SquareAuxiliariesSystem
from .simple_update_lattice import SimpleUpdateLattice

__all__ = ["SamplingGradientLattice"]

CTensor: type = TAT.Tensor.ZNo
Tensor: type = TAT.Tensor.DNo


class SamplingGradientLattice(AbstractNetworkLattice):

    @multimethod
    def __init__(self, M: int, N: int, D: int = 2, Dc: int = 2, d: int = 2):
        super().__init__(M, N, D, d)
        self.dimension_cut: int = Dc

        # TODO setitem?
        self._spin: list[list[int]] = [[0 for _ in range(self.N)] for _ in range(self.M)]
        self.auxiliaries: SquareAuxiliariesSystem = SquareAuxiliariesSystem(self.M, self.N, self.dimension_cut)

    @multimethod
    def __init__(self, other: SimpleUpdateLattice, Dc: int = 2):
        super().__init__(other)
        self.dimension_cut: int = Dc
        self.auxiliaries: SquareAuxiliariesSystem = SquareAuxiliariesSystem(self.M, self.N, self.dimension_cut)

        for i in range(self.M):
            for j in range(self.N):
                to_multiple = self.lattice[i][j]
                if i != self.M - 1:
                    to_multiple = to_multiple.multiple(other.environment["D", i, j], "D", "U")
                if j != self.N - 1:
                    to_multiple = to_multiple.multiple(other.environment["R", i, j], "R", "U")
                self.lattice[i][j] = to_multiple
