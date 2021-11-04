#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2020-2021 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
from .common_variable import clear_line
from .abstract_state import AbstractState

__all__ = ["ExactState"]


class ExactState(AbstractState):
    __slots__ = ["vector"]

    def __init__(self, abstract: AbstractState) -> None:
        super()._init_by_copy(abstract)
        self.vector: self.Tensor = self._construct_vector()

    def _construct_vector(self) -> self.Tensor:
        names: list[str] = [f"P_{i}_{j}" for i in range(self.L1) for j in range(self.L2)]
        edges: list[self.Edge] = [self.physics_edges[i, j] for i in range(self.L1) for j in range(self.L2)]
        names.append("T")
        edges.append(self.get_total_symmetry_edge())
        vector: self.Tensor = self.Tensor(names, edges).randn()
        vector /= vector.norm_2()
        return vector

    def update(self, total_step: int = 1, approximnate_energy: float = -0.5) -> float:
        total_approximate_energy: float = abs(approximnate_energy) * self.L1 * self.L2
        energy: float = 0
        for step in range(total_step):
            temporary_vector: self.Tensor = self.vector.same_shape().zero()
            for positions, value in self._hamiltonians.items():
                temporary_vector += self.vector.contract(value.edge_rename({f"O{t}": f"P_{i}_{j}" for t, [i, j] in enumerate(positions)}),
                                                         {(f"P_{i}_{j}", f"I{t}") for t, [i, j] in enumerate(positions)})
            self.vector *= total_approximate_energy
            self.vector -= temporary_vector
            # v <- a v - H v = (a - H) v => E = a - |v'|/|v|
            norm: float = float(self.vector.norm_2())
            energy: float = total_approximate_energy - norm
            self.vector /= norm
            print(clear_line, f"Exact update, {total_step=}, {step=}, energy={energy / (self.L1 * self.L2)}", end="\r")
        print(clear_line, f"Exact update done, {total_step=}, energy={energy / (self.L1 * self.L2)}")
        return energy / (self.L1 * self.L2)

    def observe(self, positions: tuple[tuple[int, int], ...], observer: self.Tensor | None) -> float:
        if len(positions) == 0:
            result: self.Tensor = self.vector
        else:
            result: self.Tensor = self.vector.contract(observer.edge_rename({f"O{t}": f"P_{i}_{j}" for t, [i, j] in enumerate(positions)}),
                                                       {(f"P_{i}_{j}", f"I{t}") for t, [i, j] in enumerate(positions)})
        TT_pair: set[tuple[str, str]] = {("T", "T")}
        result = result.contract(self.vector.conjugate(), {(f"P_{i}_{j}", f"P_{i}_{j}") for i in range(self.L1) for j in range(self.L2)} | TT_pair)
        return float(result)

    def observe_energy(self) -> float:
        energy: float = 0
        for positions, observer in self._hamiltonians.items():
            energy += self.observe(positions, observer)
        return energy / self.observe(tuple(), None) / (self.L1 * self.L2)
