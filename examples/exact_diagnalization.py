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
import numpy as np
from TAT.Tensor import DNo as Tensor

SzSz = np.reshape([1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1], [4, 4]) / 4.

SxSx = np.reshape([0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0], [4, 4]) / 4.

SySy = np.reshape([0, 0, 0, -1, 0, 0, 1, 0, 0, 1, 0, 0, -1, 0, 0, 0], [4, 4]) / 4.

SS = SxSx + SySy + SzSz


class SpinLattice():

    def __init__(self, node_names, approximate_energy=0):
        self.node_number = len(node_names)
        self.state_vector = Tensor(node_names, [2 for _ in range(self.node_number)]).randn()
        self.bonds = []
        self.energy = 0.
        self.approximate_energy = abs(approximate_energy)

    def set_bond(self, n1, n2, matrix):
        sn1 = str(n1)
        sn2 = str(n2)
        operator = Tensor([sn1, sn2, f"_{sn1}", f"_{sn2}"], [2, 2, 2, 2])
        operator.block[{}] = np.reshape(matrix, [2, 2, 2, 2])
        self.bonds.append(operator)

    def update(self):
        norm_max = float(self.state_vector.norm_max())
        self.energy = self.approximate_energy - norm_max
        self.state_vector /= norm_max
        state_vector_temporary = self.state_vector.same_shape().zero()
        for i in self.bonds:
            sn1, sn2 = i.name[:2]
            this_term = i.contract_all_edge(self.state_vector).edge_rename({f"_{sn1}": sn1, f"_{sn2}": sn2})
            state_vector_temporary += this_term
        self.state_vector *= self.approximate_energy
        self.state_vector -= state_vector_temporary  # v <- (1-H)v


class SquareSpinLattice(SpinLattice):

    def __init__(self, n1, n2):
        self.n1 = n1
        self.n2 = n2
        self.node_names = [f"{i}.{j}" for i in range(self.n1) for j in range(self.n2)]
        super().__init__(self.node_names)

    def set_bond(self, p1, p2, matrix):
        super().set_bond(f"{p1[0]}.{p1[1]}", f"{p2[0]}.{p2[1]}", matrix)
