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
from .abstract_lattice import AbstractLattice

__all__ = ["AbstractNetworkLattice"]

CTensor: type = TAT.Tensor.ZNo
Tensor: type = TAT.Tensor.DNo


class AbstractNetworkLattice(AbstractLattice):

    @multimethod
    def __init__(self, M: int, N: int, *, D: int, d: int):
        super().__init__(M, N, d=d)
        self.dimension_virtual: int = D

        # TODO better structor for two rank array
        self.lattice: list[list[Tensor]] = []

        self._initialize_network()

    @multimethod
    def __init__(self, other: AbstractNetworkLattice):
        super().__init__(other)

        self.dimension_virtual: int = other.dimension_virtual
        self.lattice: list[list[Tensor]] = [[j for j in i] for i in other.lattice]

    def _initialize_network(self) -> None:
        self.lattice = [[self._initialize_tensor_in_network(i, j) for j in range(self.N)] for i in range(self.M)]

    def _initialize_tensor_in_network(self, i: int, j: int) -> Tensor:
        name_list = ["P"]
        dimension_list = [self.dimension_physics]
        if i != 0:
            name_list.append("U")
            dimension_list.append(self.dimension_virtual)
        if j != 0:
            name_list.append("L")
            dimension_list.append(self.dimension_virtual)
        if i != self.M - 1:
            name_list.append("D")
            dimension_list.append(self.dimension_virtual)
        if j != self.N - 1:
            name_list.append("R")
            dimension_list.append(self.dimension_virtual)
        return Tensor(name_list, dimension_list).randn()