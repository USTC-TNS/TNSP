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

from .auxiliaries import Auxiliaries
from .exact_state import ExactState
from .abstract_state import AbstractState
from .abstract_lattice import AbstractLattice
from .common_variable import clear_line


class SimpleUpdateLatticeEnvironment:

    def __init__(self, owner: SimpleUpdateLattice) -> None:
        self.owner = owner

    def __getitem__(self, where: tuple[tuple[int, int], str]) -> self.Tensor | None:
        [l1, l2], direction = where
        if direction == "R":
            if 0 <= l1 < self.owner.L1 and 0 <= l2 < self.owner.L2 - 1:
                return self.owner._environment_h[l1][l2]
        elif direction == "L":
            l2 -= 1
            if 0 <= l1 < self.owner.L1 and 0 <= l2 < self.owner.L2 - 1:
                return self.owner._environment_h[l1][l2]
        elif direction == "D":
            if 0 <= l1 < self.owner.L1 - 1 and 0 <= l2 < self.owner.L2:
                return self.owner._environment_v[l1][l2]
        elif direction == "U":
            l1 -= 1
            if 0 <= l1 < self.owner.L1 - 1 and 0 <= l2 < self.owner.L2:
                return self.owner._environment_v[l1][l2]
        else:
            raise ValueError("Invalid direction")
        # out of lattice
        return None

    def __setitem__(self, where: tuple[tuple[int, int], str], value: self.Tensor | None) -> None:
        [l1, l2], direction = where
        if direction == "R":
            if 0 <= l1 < self.owner.L1 and 0 <= l2 < self.owner.L2 - 1:
                self.owner._environment_h[l1][l2] = value
                return
        elif direction == "L":
            l2 -= 1
            if 0 <= l1 < self.owner.L1 and 0 <= l2 < self.owner.L2 - 1:
                self.owner._environment_h[l1][l2] = value
                return
        elif direction == "D":
            if 0 <= l1 < self.owner.L1 - 1 and 0 <= l2 < self.owner.L2:
                self.owner._environment_v[l1][l2] = value
                return
        elif direction == "U":
            l1 -= 1
            if 0 <= l1 < self.owner.L1 - 1 and 0 <= l2 < self.owner.L2:
                self.owner._environment_v[l1][l2] = value
                return
        else:
            raise ValueError("Invalid direction")
        raise ValueError("Environment out of lattice")


class SimpleUpdateLattice(AbstractLattice):
    __slots__ = ["_lattice", "_environment_v", "_environment_h", "_auxiliaries"]

    def __init__(self, abstract: AbstractLattice) -> None:
        super().__init__(abstract)

        self._lattice: list[list[self.Tensor]] = [[self._construct_tensor(l1, l2) for l2 in range(self.L2)] for l1 in range(self.L1)]
        self._environment_h: list[list[self.Tensor | None]] = [[None for l2 in range(self.L2 - 1)] for l1 in range(self.L1)]
        self._environment_v: list[list[self.Tensor | None]] = [[None for l2 in range(self.L2)] for l1 in range(self.L1 - 1)]
        self._auxiliaries: Auxiliaries | None = None

    def __getitem__(self, l1l2: tuple[int, int]) -> self.Tensor:
        l1, l2 = l1l2
        return self._lattice[l1][l2]

    def __setitem__(self, l1l2: tuple[int, int], value: self.Tensor) -> None:
        l1, l2 = l1l2
        self._lattice[l1][l2] = value

    @property
    def environment(self) -> SimpleUpdateLatticeEnvironment:
        return SimpleUpdateLatticeEnvironment(self)

    def update(self, total_step: int, delta_tau: float, new_dimension: int) -> None:
        updater: dict[tuple[tuple[int, int], ...], self.Tensor] = {}
        for positions, hamiltonian_term in self._hamiltonians.items():
            site_number: int = len(positions)
            updater[positions] = (-delta_tau * hamiltonian_term).exponential({(f"I{i}", f"O{i}") for i in range(site_number)}, step=8)
        for step in range(total_step):
            print(clear_line, f"Simple update, {total_step=}, {delta_tau=}, {new_dimension=}, {step=}", end="\r")
            for positions, term in updater.items():
                self._single_term_simple_update(positions, term, new_dimension)
            for positions, term in reversed(updater.items()):
                self._single_term_simple_update(positions, term, new_dimension)
        print(clear_line, f"Simple update done, {total_step=}, {delta_tau=}, {new_dimension=}")
        self._update_virtual_bond()

    def _update_virtual_bond(self) -> None:
        for l1 in range(self.L1):
            for l2 in range(self.L2):
                if l1 != self.L1 - 1:
                    self.virtual_bond[(l1, l2), "D"] = self[l1, l2].edges("D")
                if l2 != self.L2 - 1:
                    self.virtual_bond[(l1, l2), "R"] = self[l1, l2].edges("R")

    def _single_term_simple_update(self, positions: tuple[tuple[int, int], ...], updater: self.Tensor, new_dimension: int) -> None:
        if len(positions) == 1:
            return self._single_term_simple_update_single_site(positions[0], updater)
        if len(positions) == 2:
            position_1, position_2 = positions
            if position_1[0] == position_2[0]:
                if position_1[1] == position_2[1] + 1:
                    return self._single_term_simple_update_double_site_nearest_horizontal(position_2, updater, new_dimension)
                if position_2[1] == position_1[1] + 1:
                    return self._single_term_simple_update_double_site_nearest_horizontal(position_1, updater, new_dimension)
            if position_1[1] == position_2[1]:
                if position_1[0] == position_2[0] + 1:
                    return self._single_term_simple_update_double_site_nearest_vertical(position_2, updater, new_dimension)
                if position_2[0] == position_1[0] + 1:
                    return self._single_term_simple_update_double_site_nearest_vertical(position_1, updater, new_dimension)
        raise NotImplementedError("Unsupported simple update style")

    def _single_term_simple_update_single_site(self, position: tuple[int, int], updater: self.Tensor) -> None:
        i, j = position
        self[i, j] = self[i, j].contract(updater, {("P", "I0")}).edge_rename({"O0": "P"})

    def _single_term_simple_update_double_site_nearest_horizontal(self, position: tuple[int, int], updater: self.Tensor, new_dimension: int) -> None:
        i, j = position
        left = self[i, j]
        left = self._try_multiple(left, i, j, "L")
        left = self._try_multiple(left, i, j, "U")
        left = self._try_multiple(left, i, j, "D")
        left = self._try_multiple(left, i, j, "R")
        right = self[i, j + 1]
        right = self._try_multiple(right, i, j + 1, "U")
        right = self._try_multiple(right, i, j + 1, "D")
        right = self._try_multiple(right, i, j + 1, "R")
        left_q, left_r = left.qr("r", {"P", "R"}, "R", "L")
        right_q, right_r = right.qr("r", {"P", "L"}, "L", "R")
        u, s, v = left_r.edge_rename({"P": "P0"}) \
            .contract(right_r.edge_rename({"P": "P1"}), {("R", "L")}) \
            .contract(updater, {("P0", "I0"), ("P1", "I1")}) \
            .svd({"L", "O0"}, "R", "L", "L", "R", new_dimension)
        s /= s.norm_sum()
        self.environment[(i, j), "R"] = s
        left_q = self._try_multiple(left_q, i, j, "L", True)
        left_q = self._try_multiple(left_q, i, j, "U", True)
        left_q = self._try_multiple(left_q, i, j, "D", True)
        u = u.contract(left_q, {("L", "R")}).edge_rename({"O0": "P"})
        self[i, j] = u
        right_q = self._try_multiple(right_q, i, j + 1, "U", True)
        right_q = self._try_multiple(right_q, i, j + 1, "D", True)
        right_q = self._try_multiple(right_q, i, j + 1, "R", True)
        v = v.contract(right_q, {("R", "L")}).edge_rename({"O1": "P"})
        self[i, j + 1] = v

    def _single_term_simple_update_double_site_nearest_vertical(self, position: tuple[int, int], updater: self.Tensor, new_dimension: int) -> None:
        i, j = position
        up = self[i, j]
        up = self._try_multiple(up, i, j, "L")
        up = self._try_multiple(up, i, j, "U")
        up = self._try_multiple(up, i, j, "D")
        up = self._try_multiple(up, i, j, "R")
        down = self[i + 1, j]
        down = self._try_multiple(down, i + 1, j, "L")
        down = self._try_multiple(down, i + 1, j, "D")
        down = self._try_multiple(down, i + 1, j, "R")
        up_q, up_r = up.qr("r", {"P", "D"}, "D", "U")
        down_q, down_r = down.qr("r", {"P", "U"}, "U", "D")
        u, s, v = up_r.edge_rename({"P": "P0"}) \
            .contract(down_r.edge_rename({"P": "P1"}), {("D", "U")}) \
            .contract(updater, {("P0", "I0"), ("P1", "I1")}) \
            .svd({"U", "O0"}, "D", "U","U","D", new_dimension)
        s /= s.norm_sum()
        self.environment[(i, j), "D"] = s
        up_q = self._try_multiple(up_q, i, j, "L", True)
        up_q = self._try_multiple(up_q, i, j, "U", True)
        up_q = self._try_multiple(up_q, i, j, "R", True)
        u = u.contract(up_q, {("U", "D")}).edge_rename({"O0": "P"})
        self[i, j] = u
        down_q = self._try_multiple(down_q, i + 1, j, "L", True)
        down_q = self._try_multiple(down_q, i + 1, j, "D", True)
        down_q = self._try_multiple(down_q, i + 1, j, "R", True)
        v = v.contract(down_q, {("D", "U")}).edge_rename({"O1": "P"})
        self[i + 1, j] = v

    def _try_multiple(self, tensor: self.Tensor, i: int, j: int, direction: str, division: bool = False) -> self.Tensor:
        environment_tensor = self.environment[(i, j), direction]
        if environment_tensor is not None:
            if division:
                environment_tensor = environment_tensor.map(lambda x: 0 if x == 0 else 1. / x)
            if direction == "L":
                tensor = tensor.contract(environment_tensor, {("L", "R")})
            if direction == "R":
                tensor = tensor.contract(environment_tensor, {("R", "L")})
            if direction == "U":
                tensor = tensor.contract(environment_tensor, {("U", "D")})
            if direction == "D":
                tensor = tensor.contract(environment_tensor, {("D", "U")})
        return tensor

    def exact_state(self) -> ExactState:
        result: ExactState = ExactState(self)
        for l1 in range(self.L1):
            for l2 in range(self.L2):
                rename_map: dict[str, str] = {}
                rename_map["P"] = f"P_{l1}_{l2}"
                if l1 != self.L1 - 1:
                    rename_map["D"] = f"D_{l2}"
                this: self.Tensor = self[l1, l2].edge_rename(rename_map)
                this = self._try_multiple(this, l1, l2, "L")
                this = self._try_multiple(this, l1, l2, "U")
                if l1 == l2 == 0:
                    result.vector = this
                else:
                    contract_pair: set[tuple[int, int]] = set()
                    if l2 != 0:
                        contract_pair.add(("R", "L"))
                    if l1 != 0:
                        contract_pair.add((f"D_{l2}", "U"))
                    result.vector = result.vector.contract(this, contract_pair)
        return result

    def initialize_auxiliaries(self, Dc: int) -> None:
        self._auxiliaries = Auxiliaries(self.L1, self.L2, Dc, self.Tensor)
        for l1 in range(self.L1):
            for l2 in range(self.L2):
                this_site: self.Tensor = self[l1, l2]
                this_site = self._try_multiple(this_site, l1, l2, "L")
                this_site = self._try_multiple(this_site, l1, l2, "U")
                contract_pairs: set[tuple[str, str]]
                if l1 == l2 == 0:
                    contract_pairs = {("P", "P"), ("T", "T")}
                else:
                    contract_pairs = {("P", "P")}
                double_site = this_site.edge_rename({
                    "L": "L1",
                    "R": "R1",
                    "U": "U1",
                    "D": "D1"
                }).contract(this_site.conjugate().edge_rename({
                    "L": "L2",
                    "R": "R2",
                    "U": "U2",
                    "D": "D2"
                }), contract_pairs)
                merge_map: dict[str, list[str]] = {}
                if l1 != 0:
                    merge_map["U"] = ["U1", "U2"]
                if l2 != 0:
                    merge_map["L"] = ["L1", "L2"]
                if l1 != self.L1 - 1:
                    merge_map["D"] = ["D1", "D2"]
                if l2 != self.L2 - 1:
                    merge_map["R"] = ["R1", "R2"]
                self._auxiliaries[l1, l2] = double_site.merge_edge(merge_map, parity_exclude_name_merge_set={"R", "D"}, parity_exclude_name_reverse_set={"R1", "R2", "D1", "D2"})
        # protocol:
        # The double layer merging order is [normal tensor, conjugated tensor].
        # In auxiliaries, every site contain the environment in the left and up direction, if it exists.
        # When reversing and merging edge, edge left and up DO NOT apply sign,
        # and edge down and right DO apply sign.

    def observe(self, positions: tuple[tuple[int, int], ...], observer: self.Tensor | None) -> float:
        if self._auxiliaries is None:
            raise RuntimeError("Need to initialize auxiliary before call observe")
        # print("observing", positions)

        body: int = len(positions)
        if body == 0:
            return float(self._auxiliaries())
        # TODO detailed positions specific algorithm can be optimized

        outer: self.Tensor = self._auxiliaries(positions)
        outer = outer.edge_rename({f"D{i}": f"U{i}" for i in range(body)} | {f"U{i}": f"D{i}" for i in range(body)} | {f"L{i}": f"R{i}" for i in range(body)} | {f"R{i}": f"L{i}" for i in range(body)})

        inner_single: self.Tensor = self.Tensor(1)
        remained_names: set[str] = set()
        for index, position in enumerate(positions):
            l1, l2 = position
            this_site: self.Tensor = self[l1, l2]
            this_site = self._try_multiple(this_site, l1, l2, "L")
            this_site = self._try_multiple(this_site, l1, l2, "U")
            this_site = this_site.edge_rename({name: f"{name}{index}" for name in this_site.names if name != "T"})
            contract_set: set[tuple[str, str]] = set()
            if l1 != 0:
                remained_names.add(f"U{index}")
            if l2 != 0:
                remained_names.add(f"L{index}")
            if l1 != self.L1 - 1:
                remained_names.add(f"D{index}")
            if l2 != self.L2 - 1:
                remained_names.add(f"R{index}")

            for other_index in range(index):
                other_position: tuple[int, int] = positions[other_index]
                m1, m2 = other_position
                if m1 == l1 and m2 + 1 == l2:
                    # former - this
                    contract_set.add((f"R{other_index}", f"L{index}"))
                    remained_names.remove(f"L{index}")
                    remained_names.remove(f"R{other_index}")
                elif m1 == l1 and m2 - 1 == l2:
                    # this - former
                    contract_set.add((f"L{other_index}", f"R{index}"))
                    remained_names.remove(f"R{index}")
                    remained_names.remove(f"L{other_index}")
                elif m1 + 1 == l1 and m2 == l2:
                    # former
                    #   |
                    # this
                    contract_set.add((f"D{other_index}", f"U{index}"))
                    remained_names.remove(f"U{index}")
                    remained_names.remove(f"D{other_index}")
                elif m1 - 1 == l1 and m2 == l2:
                    # this
                    #   |
                    # former
                    contract_set.add((f"U{other_index}", f"D{index}"))
                    remained_names.remove(f"D{index}")
                    remained_names.remove(f"U{other_index}")
            inner_single = inner_single.contract(this_site, contract_set)
        inner_1: self.Tensor = inner_single.edge_rename({name: f"{name}.1" for name in remained_names})
        inner_2: self.Tensor = inner_single.edge_rename({name: f"{name}.2" for name in remained_names}).conjugate()
        if "T" in inner_single.names:
            with_T: set[tuple[str, str]] = {("T", "T")}
        else:
            with_T: set[tuple[str, str]] = set()
        big: self.Tensor = inner_1.contract(observer, {(f"P{i}", f"I{i}") for i in range(body)}).contract(inner_2, {(f"O{i}", f"P{i}") for i in range(body)} | with_T)
        merge_parity_set = {name for name in remained_names if name[0] == "R" or name[0] == "D"}
        reverse_parity_set = {f"{name}.1" for name in merge_parity_set} | {f"{name}.2" for name in merge_parity_set}
        inner: self.Tensor = big.merge_edge({name: [f"{name}.1", f"{name}.2"] for name in remained_names},
                                            parity_exclude_name_merge_set=merge_parity_set,
                                            parity_exclude_name_reverse_set=reverse_parity_set)
        return float(inner.contract(outer, {(name, name) for name in inner.names}))

    def observe_energy(self) -> float:
        energy: float = 0
        for positions, observer in self._hamiltonians.items():
            # print("observing", positions)
            energy += self.observe(positions, observer)
        return energy / self.observe(tuple(), None) / (self.L1 * self.L2)

    # TODO full update
