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
from multimethod import multimethod

from .abstract_state import AbstractState

__all__ = ["AbstractLattice"]


class AbstractLatticeVirtualBond:
    __slots__ = ["owner"]

    def __init__(self, owner: AbstractState) -> None:
        self.owner: AbstractState = owner

    def __getitem__(self, where: tuple[tuple[int, int], str]) -> self.owner.Tensor | None:
        l1l2, direction = where
        return self.owner._virtual_bond[l1l2][direction]

    def __setitem__(self, where: tuple[tuple[int, int] | type(...), str], value) -> None:
        l1l2, direction = where
        if l1l2 == ...:
            for i in range(self.owner.L1):
                for j in range(self.owner.L2):
                    self.owner._set_virtual_bond((i, j), direction, value)
        else:
            self.owner._set_virtual_bond(l1l2, direction, value)


class AbstractLattice(AbstractState):
    __slots__ = ["_virtual_bond"]

    @multimethod
    def __init__(self, abstract: AbstractState) -> None:
        super().__init__(abstract)

        self._virtual_bond: dict[tuple[int, int], dict[str, self.Edge]] = {(l1, l2): self._default_bonds(l1, l2) for l1 in range(self.L1) for l2 in range(self.L2)}

    @multimethod
    def __init__(self, other: AbstractLattice) -> None:
        super().__init__(other)

        self._virtual_bond: dict[tuple[int, int], dict[str, self.Edge]] = {i: {k: l for k, l in j.items()} for i, j in other._virtual_bond.items()}

    def _construct_tensor(self, l1: int, l2: int) -> self.Tensor:
        names: list[str] = ["P"]
        edges: list[self.Edge] = [self._physics_edges[l1][l2]]
        for i, j in self._virtual_bond[l1, l2].items():
            names.append(i)
            edges.append(j)
        return self.Tensor(names, edges).randn()

    def _default_bonds(self, l1: int, l2: int) -> dict[str, self.Edge]:
        result: dict[str, self.Edge] = {}
        if l1 == l2 == 0:
            if self.total_symmetry:
                result["T"] = self.get_total_symmetry_edge()
        if l1 != 0:
            result["U"] = None
        if l1 != self.L1 - 1:
            result["D"] = None
        if l2 != 0:
            result["L"] = None
        if l2 != self.L2 - 1:
            result["R"] = None
        return result

    @property
    def virtual_bond(self) -> AbstractLatticeVirtualBond:
        return AbstractLatticeVirtualBond(self)

    def _set_virtual_bond_single(self, l1l2: tuple[int, int], direction: str, edge: self.Edge) -> None:
        if l1l2 in self._virtual_bond:
            site = self._virtual_bond[l1l2]
            if direction in site:
                site[direction] = edge

    def _set_virtual_bond(self, l1l2: tuple[int, int], direction: str, edge) -> None:
        if isinstance(edge, self.Edge):
            edge = edge
        else:
            edge = self.Edge(edge)
        self._set_virtual_bond_single(l1l2, direction, edge)
        l1, l2 = l1l2
        if direction == "L":
            l2 -= 1
            direction = "R"
        elif direction == "R":
            l2 += 1
            direction = "L"
        elif direction == "U":
            l1 -= 1
            direction = "D"
        elif direction == "D":
            l1 += 1
            direction = "U"
        else:
            raise ValueError("Invalid direction when setting virtual bond")
        self._set_virtual_bond_single((l1, l2), direction, edge.conjugated())
