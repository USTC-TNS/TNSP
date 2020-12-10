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
from typing import List, Tuple
from multimethod import multimethod
import TAT
from .abstract_network_lattice import AbstractNetworkLattice
from .auxiliaries import SquareAuxiliariesSystem
from . import simple_update_lattice

__all__ = ["SamplingGradientLattice"]

CTensor: type = TAT.Tensor.ZNo
Tensor: type = TAT.Tensor.DNo


class SpinConfiguration(SquareAuxiliariesSystem):
    __slots__ = ["lattice", "configuration"]

    @multimethod
    def __init__(self, lattice: SamplingGradientLattice) -> None:
        super().__init__(lattice.M, lattice.N, lattice.dimension_cut)
        self.lattice: SamplingGradientLattice = lattice
        self.configuration: List[List[int]] = [[-1 for _ in range(self.lattice.N)] for _ in range(self.lattice.M)]

    @multimethod
    def __init__(self, other: SpinConfiguration) -> None:
        super().__init__(other)
        self.lattice: SamplingGradientLattice = other.lattice
        self.configuration: List[List[int]] = [[other.configuration[i][j] for j in range(self.lattice.N)] for i in range(self.lattice.M)]

    def __setitem__(self, position: Tuple[int, int], value: int) -> None:
        x, y = position
        if self.configuration[x][y] != value:
            super().__setitem__((x, y), self.lattice[x, y].shrink({"P": value}))


class SamplingGradientLattice(AbstractNetworkLattice):
    __slots__ = ["dimension_cut", "spin"]

    @multimethod
    def __init__(self, M: int, N: int, *, D: int, Dc: int, d: int) -> None:
        super().__init__(M, N, D=D, d=d)
        self.dimension_cut: int = Dc

        self.spin: SpinConfiguration = SpinConfiguration(self.M, self.N, self.dimension_cut)

    @multimethod
    def __init__(self, other: SamplingGradientLattice) -> None:
        super().__init__(other)
        self.dimension_cut: int = other.dimension_cut
        self.spin: SpinConfiguration = SpinConfiguration(other.spin)

    @multimethod
    def __init__(self, other: simple_update_lattice.SimpleUpdateLattice, *, Dc: int = 2) -> None:
        super().__init__(other)
        self.dimension_cut: int = Dc
        self.auxiliaries: SquareAuxiliariesSystem = SquareAuxiliariesSystem(self.M, self.N, self.dimension_cut)

        for i in range(self.M):
            for j in range(self.N):
                to_multiple = self[i, j]
                to_multiple = other.try_multiple(to_multiple, i, j, "D")
                to_multiple = other.try_multiple(to_multiple, i, j, "R")
                self[i, j] = to_multiple
