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

import TAT

Sx = TAT.Tensor.ZNo(["I", "O"], [2, 2])
Sx.block[{}] = [[0, 0.5], [0.5, 0]]
Sy = TAT.Tensor.ZNo(["I", "O"], [2, 2])
Sy.block[{}] = [[0, -0.5j], [0.5j, 0]]
Sz = TAT.Tensor.ZNo(["I", "O"], [2, 2])
Sz.block[{}] = [[0.5, 0], [0, -0.5]]

SxSx = Sx.edge_rename({"I": "I0", "O": "O0"}).contract_all_edge(Sx.edge_rename({"I": "I1", "O": "O1"})).to(float)
SySy = Sy.edge_rename({"I": "I0", "O": "O0"}).contract_all_edge(Sy.edge_rename({"I": "I1", "O": "O1"})).to(float)
SzSz = Sz.edge_rename({"I": "I0", "O": "O0"}).contract_all_edge(Sz.edge_rename({"I": "I1", "O": "O1"})).to(float)

SS = SxSx + SySy + SzSz


def translate(ps, operator):
    translater = {}
    for i, j in enumerate(ps):
        translater[f"I{i}"] = j
        translater[f"O{i}"] = f"_{j}"
    return operator.edge_rename(translater)


class SpinLattice():

    def __init__(self, node_names, approximate_energy=0):
        self.node_number = len(node_names)
        self.state_vector = Tensor(node_names, [2 for _ in range(self.node_number)]).randn()
        self.bonds = []
        self.energy = 0.
        self.approximate_energy = abs(approximate_energy)

    def update(self):
        norm_max = float(self.state_vector.norm_max())
        self.energy = self.approximate_energy - norm_max
        self.state_vector /= norm_max
        state_vector_temporary = self.state_vector.same_shape().zero()
        for i in self.bonds:
            sn1, sn2 = (j for j in i.name if j.name[0] != "_")
            this_term = self.state_vector.contract_all_edge(i).edge_rename({f"_{sn1}": sn1, f"_{sn2}": sn2})
            state_vector_temporary += this_term
        self.state_vector *= self.approximate_energy
        self.state_vector -= state_vector_temporary  # v <- (1-H)v

    def observe(self, operator):
        names = [i.name for i in operator.name if i.name[0] != "_"]
        Ov = self.state_vector.to(complex).contract_all_edge(operator).edge_rename({f"_{n}": n for n in names})
        vOv = Ov.contract_all_edge(self.state_vector.to(complex))
        vv = self.state_vector.contract_all_edge(self.state_vector)
        return complex(vOv).real / float(vv)

    def set_bond(self, ps, operator):
        self.bonds.append(translate(ps, operator))


class SquareSpinLattice(SpinLattice):

    def __init__(self, n1, n2, approximate_energy=0):
        self.n1 = n1
        self.n2 = n2
        self.node_names = [f"{i}.{j}" for i in range(self.n1) for j in range(self.n2)]
        super().__init__(self.node_names, approximate_energy)

    def resolve(self, p):
        return f"{p[0]}.{p[1]}"

    def set_bond(self, ps, operator):
        pss = [self.resolve(i) for i in ps]
        return super().set_bond(pss, operator)
