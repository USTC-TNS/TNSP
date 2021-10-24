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

__all__ = ["AbstractState"]


class AbstractStatePhysicsEdge:
    __slots__ = ["owner"]

    def __init__(self, owner: AbstractState) -> None:
        self.owner: AbstractState = owner

    def __getitem__(self, l1l2: tuple[int, int]) -> self.owner.Edge:
        l1, l2 = l1l2
        return self.owner._physics_edges[l1][l2]

    def __setitem__(self, l1l2: tuple[int, int], edge: self.owner.Edge) -> None:
        l1, l2 = l1l2
        self.owner._physics_edges[l1][l2] = self.owner._construct_physics_edge(edge)


class AbstractStateHamiltonian:
    __slots__ = ["owner"]

    def __init__(self, owner: AbstractState) -> None:
        self.owner: AbstractState = owner

    def __getitem__(self, points: tuple[tuple[int, int], ...]) -> self.owner.Tensor:
        return self.owner._hamiltonians[points]

    def __setitem__(self, points: tuple[tuple[int, int], ...], tensor: self.owner.Tensor):
        self.owner._set_hamiltonian(points, tensor)

    @property
    def vertical_bond(self) -> None:
        raise RuntimeError("Getting hamiltonians is not allowed")

    @vertical_bond.setter
    def vertical_bond(self, tensor: self.owner.Tensor) -> None:
        for i in range(self.owner.L1 - 1):
            for j in range(self.owner.L2):
                self.owner._set_hamiltonian(((i, j), (i + 1, j)), tensor)

    @property
    def horizontal_bond(self) -> None:
        raise RuntimeError("Getting hamiltonians is not allowed")

    @horizontal_bond.setter
    def horizontal_bond(self, tensor: self.owner.Tensor) -> None:
        for i in range(self.owner.L1):
            for j in range(self.owner.L2 - 1):
                self.owner._set_hamiltonian(((i, j), (i, j + 1)), tensor)


class AbstractState:
    __slots__ = ["Tensor", "Edge", "Symmetry", "L1", "L2", "_physics_edges", "_hamiltonians", "_total_symmetry"]

    def __init__(self, Tensor: type, L1: int, L2: int) -> None:
        self.Tensor: type = Tensor
        self.Edge: type = Tensor.model.Edge
        self.Symmetry: type = Tensor.model.Symmetry

        self.L1: int = L1
        self.L2: int = L2
        self._physics_edges: list[list[self.Edge]] = [[None for l2 in range(self.L2)] for l1 in range(self.L1)]
        self._hamiltonians: dict[tuple[tuple[int, int], ...], self.Tensor] = {}  # ((int, int), ...) -> Tensor
        self._total_symmetry: self.Symmetry | None = None

    def _init_by_copy(self, other: AbstractState) -> None:
        self.Tensor: type = other.Tensor
        self.Edge: type = other.Edge
        self.Symmetry: type = other.Symmetry
        self.L1: int = other.L1
        self.L2: int = other.L2
        self._physics_edges: list[list[self.Edge]] = [[other._physics_edges[i][j] for j in range(self.L2)] for i in range(self.L1)]
        self._hamiltonians: dict[tuple[tuple[int, int], ...], self.Tensor] = other._hamiltonians.copy()
        self._total_symmetry: self.Symmetry | None = other.total_symmetry

    @property
    def total_symmetry(self) -> self.Symmetry:
        return self._total_symmetry

    @total_symmetry.setter
    def total_symmetry(self, value) -> None:
        if isinstance(value, self.Symmetry):
            self._total_symmetry = value
        else:
            self._total_symmetry = self.Symmetry(value)

    def is_fermi(self) -> bool:
        return hasattr(self.Edge, "arrow")

    def get_total_symmetry_edge(self) -> self.Edge:
        if self.is_fermi():
            return self.Edge([(-self._total_symmetry, 1)], True)
        else:
            return self.Edge([(-self._total_symmetry, 1)])

    def _construct_physics_edge(self, edge) -> self.Edge:
        if isinstance(edge, self.Edge):
            result = edge
        else:
            result = self.Edge(edge)
        if self.is_fermi():
            if result.arrow != False:
                raise ValueError("Edge arrow of physics bond should be False")
        return result

    @property
    def physics_edges(self) -> AbstractStatePhysicsEdge:
        return AbstractStatePhysicsEdge(self)

    @physics_edges.setter
    def physics_edges(self, edge) -> None:
        edge = self._construct_physics_edge(edge)
        self._physics_edges = [[edge for l2 in range(self.L2)] for l1 in range(self.L1)]

    def _set_hamiltonian(self, points: tuple[tuple[int, int], ...], tensor: self.Tensor) -> None:
        body: int = len(points)
        if {f"{i}" for i in tensor.names} != {f"{i}{j}" for i in ["I", "O"] for j in range(body)}:
            raise ValueError("Wrong hamiltonian name")
        for i in range(body):
            edge_out: self.Edge = tensor.edges(f"O{i}")
            if edge_out != self._physics_edges[points[i][0]][points[i][1]]:
                print(edge_out, self._physics_edges[points[i][0]][points[i][1]])
                print(edge_out == self._physics_edges[points[i][0]][points[i][1]])
                raise ValueError("Wrong hamiltonian edge")
            edge_in: self.Edge = tensor.edges(f"I{i}")
            if edge_out.conjugated() != edge_in:
                raise ValueError("Wrong hamiltonian edge")

        self._hamiltonians[points] = tensor

    @property
    def hamiltonians(self) -> AbstractStateHamiltonian:
        return AbstractStateHamiltonian(self)

    @hamiltonians.setter
    def hamiltonians(self, tensor: self.Tensor) -> None:
        for i in range(self.L1):
            for j in range(self.L2):
                self._set_hamiltonian(((i, j),), tensor)
