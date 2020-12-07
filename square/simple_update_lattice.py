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
from . import sampling_gradient_lattice

__all__ = ["SimpleUpdateLattice"]

CTensor: type = TAT.Tensor.ZNo
Tensor: type = TAT.Tensor.DNo

clear_line = "\u001b[2K"


class SimpleUpdateLattice(AbstractNetworkLattice):

    @multimethod
    def __init__(self, M: int, N: int, *, D: int, d: int = 2):
        super().__init__(M, N, D=D, d=d)

        self.environment: dict[tuple[str, int, int], Tensor] = {}
        # 第一个str可以是"D"或者"R"

    @multimethod
    def __init__(self, other: sampling_gradient_lattice.SamplingGradientLattice):
        super().__init__(other)

        self.environment: dict[tuple[str, int, int], Tensor] = {}

    def _construct_environment(self) -> None:
        for i in range(self.M):
            for j in range(self.N - 1):
                self.environment["R", i, j] = Tensor([",SVD_U", ",SVD_V"], [self.dimension_virtual, self.dimension_virtual]).identity({(",SVD_U", ",SVD_V")})

        for i in range(self.M - 1):
            for j in range(self.N):
                self.environment["D", i, j] = Tensor([",SVD_U", ",SVD_V"], [self.dimension_virtual, self.dimension_virtual]).identity({(",SVD_U", ",SVD_V")})

    def simple_update(self, time: int, delta_t: float, new_dimension: int = 0) -> None:
        if new_dimension != 0:
            self.dimension_virtual = new_dimension
        updater: dict[tuple[tuple[int, int], ...], Tensor] = {}
        for positions, term in self.hamiltonian.items():
            site_number: int = len(positions)
            updater[positions] = (-delta_t * term).exponential({(f"I{i}", f"O{i}") for i in range(site_number)}, 8)
        for t in range(time):
            print(f"Simple updating, total_step={time}, delta_t={delta_t}, dimension={self.dimension_virtual}, step={t}", end="\r")
            for positions, term in updater.items():
                self._single_term_simple_update(positions, term)
            for positions, term in reversed(updater.items()):
                self._single_term_simple_update(positions, term)
        print(f"{clear_line}Simple update done, total_step={time}, delta_t={delta_t}, dimension={self.dimension_virtual}")

    def _single_term_simple_update(self, positions: tuple[tuple[int, int], ...], updater: Tensor):
        if len(positions) == 1:
            return self._single_term_simple_update_single_site(positions[0], updater)
        if len(positions) == 2:
            position_1, position_2 = positions
            if position_1[0] == position_2[0]:
                if position_1[1] == position_2[1] + 1:
                    return self._single_term_simple_update_double_site_nearest_horizontal(position_2, updater)
                if position_2[1] == position_1[1] + 1:
                    return self._single_term_simple_update_double_site_nearest_horizontal(position_1, updater)
            if position_1[1] == position_2[1]:
                if position_1[0] == position_2[0] + 1:
                    return self._single_term_simple_update_double_site_nearest_vertical(position_2, updater)
                if position_2[0] == position_1[0] + 1:
                    return self._single_term_simple_update_double_site_nearest_vertical(position_1, updater)
        raise NotImplementedError("Unsupported simple update style")

    def _single_term_simple_update_double_site_nearest_horizontal(self, position: tuple[int, int], updater: Tensor):
        i, j = position
        left = self.lattice[i][j]
        left = self._try_multiple(left, i, j, "L")
        left = self._try_multiple(left, i, j, "U")
        left = self._try_multiple(left, i, j, "D")
        left = self._try_multiple(left, i, j, "R")
        right = self.lattice[i][j + 1]
        right = self._try_multiple(right, i, j + 1, "U")
        right = self._try_multiple(right, i, j + 1, "D")
        right = self._try_multiple(right, i, j + 1, "R")
        left_q, left_r = left.qr("r", {"P", "R"}, "R", "L")
        right_q, right_r = right.qr("r", {"P", "L"}, "L", "R")
        u, s, v = left_r.edge_rename({"P": "P0"}) \
            .contract(right_r.edge_rename({"P": "P1"}), {("R", "L")}) \
            .contract(updater, {("P0", "I0"), ("P1", "I1")}) \
            .svd({"L", "O0"}, "R", "L", self.dimension_virtual)
        s /= s.norm_max()
        self.environment["R", i, j] = s
        u = u.contract(left_q, {("L", "R")}).edge_rename({"O0": "P"})
        u = self._try_multiple(u, i, j, "L", True)
        u = self._try_multiple(u, i, j, "U", True)
        u = self._try_multiple(u, i, j, "D", True)
        u /= u.norm_max()
        self.lattice[i][j] = u
        v = v.contract(right_q, {("R", "L")}).edge_rename({"O1": "P"})
        v = self._try_multiple(v, i, j + 1, "U", True)
        v = self._try_multiple(v, i, j + 1, "D", True)
        v = self._try_multiple(v, i, j + 1, "R", True)
        v /= v.norm_max()
        self.lattice[i][j + 1] = v

    def _single_term_simple_update_double_site_nearest_vertical(self, position: tuple[int, int], updater: Tensor):
        i, j = position
        up = self.lattice[i][j]
        up = self._try_multiple(up, i, j, "L")
        up = self._try_multiple(up, i, j, "U")
        up = self._try_multiple(up, i, j, "D")
        up = self._try_multiple(up, i, j, "R")
        down = self.lattice[i + 1][j]
        down = self._try_multiple(down, i + 1, j, "L")
        down = self._try_multiple(down, i + 1, j, "D")
        down = self._try_multiple(down, i + 1, j, "R")
        up_q, up_r = up.qr("r", {"P", "D"}, "D", "U")
        down_q, down_r = down.qr("r", {"P", "U"}, "U", "D")
        u, s, v = up_r.edge_rename({"P": "P0"}) \
            .contract(down_r.edge_rename({"P": "P1"}), {("D", "U")}) \
            .contract(updater, {("P0", "I0"), ("P1", "I1")}) \
            .svd({"U", "O0"}, "D", "U", self.dimension_virtual)
        s /= s.norm_max()
        self.environment["D", i, j] = s
        u = u.contract(up_q, {("U", "D")}).edge_rename({"O0": "P"})
        u = self._try_multiple(u, i, j, "L", True)
        u = self._try_multiple(u, i, j, "U", True)
        u = self._try_multiple(u, i, j, "R", True)
        u /= u.norm_max()
        self.lattice[i][j] = u
        v = v.contract(down_q, {("D", "U")}).edge_rename({"O1": "P"})
        v = self._try_multiple(v, i + 1, j, "L", True)
        v = self._try_multiple(v, i + 1, j, "D", True)
        v = self._try_multiple(v, i + 1, j, "R", True)
        v /= v.norm_max()
        self.lattice[i + 1][j] = v

    def _single_term_simple_update_single_site(self, position: tuple[int, int], updater: Tensor):
        i, j = position
        self.lattice[i][j] = self.lattice[i][j].contract(updater, {("P", "I0")}).edge_rename({"O0": "P"})

    def _try_multiple(self, tensor: Tensor, i: int, j: int, direction: str, division: bool = False) -> Tensor:
        if direction == "L":
            if ("R", i, j - 1) in self.environment:
                return tensor.multiple(self.environment["R", i, j - 1], "L", "v", division)
        if direction == "U":
            if ("D", i - 1, j) in self.environment:
                return tensor.multiple(self.environment["D", i - 1, j], "U", "v", division)
        if direction == "D":
            if ("D", i, j) in self.environment:
                return tensor.multiple(self.environment["D", i, j], "D", "u", division)
        if direction == "R":
            if ("R", i, j) in self.environment:
                return tensor.multiple(self.environment["R", i, j], "R", "u", division)
        return tensor

    def auxiliary_observe(self):
        # TODO aux observe in su lattice
        pass