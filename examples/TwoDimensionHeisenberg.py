#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2019-2020 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

from typing import List, Optional, Tuple, Dict
import numpy as np
import pickle
from TAT.Tensor import DNo as Tensor
from TAT.Singular import DNo as Singular


class TwoDimensionHeisenberg:
    raw_base: List[Tensor] = [Tensor(["Phy"], [2]) for _ in range(2)]
    raw_base[0].block()[:] = [1, 0]
    raw_base[1].block()[:] = [0, 1]

    hamiltonian = Tensor("I0 I1 O0 O1".split(" "), [2, 2, 2, 2])
    hamiltonian.block()[:] = np.array([1 / 4., 0, 0, 0, 0, -1 / 4., 2 / 4., 0, 0, 2 / 4., -1 / 4., 0, 0, 0, 0, 1 / 4.]).reshape([2, 2, 2, 2])

    hamiltonian_square = hamiltonian.contract(hamiltonian, {("I0", "O0"), ("I1", "O1")})

    identity = Tensor("I0 I1 O0 O1".split(" "), [2, 2, 2, 2])
    identity.block()[:] = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]).reshape([2, 2, 2, 2])

    __slots__ = ["L1", "L2", "D", "lattice", "spin", "_auxiliaries", "_lattice_spin", "environment"]

    def _create_tensor(self, l1: int, l2: int) -> Tensor:
        name_list = ["Left", "Right", "Up", "Down"]
        if l1 == 0:
            name_list.remove("Up")
        if l2 == 0:
            name_list.remove("Left")
        if l1 == self.L1 - 1:
            name_list.remove("Down")
        if l2 == self.L2 - 1:
            name_list.remove("Right")
        dimension_list = [2, *[self.D for _ in name_list]]
        name_list = ["Phy", *name_list]
        result = Tensor(name_list, dimension_list)
        result.block()[:] = np.random.rand(*dimension_list)
        return result

    def _initialize_spin(self) -> None:
        for l1 in range(self.L1):
            for l2 in range(self.L2):
                self.spin[l1][l2] = (l1 + l2) % 2

    @staticmethod
    def _two_line_to_one_line(udlr_name: List[str], line_1: List[Tensor], line_2: List[Tensor], cut: int) -> List[Tensor]:
        [up, down, left, right] = udlr_name
        up1 = up + "1"
        up2 = up + "2"
        down1 = down + "1"
        down2 = down + "2"
        left1 = left + "1"
        left2 = left + "2"
        right1 = right + "1"
        right2 = right + "2"

        length = len(line_1)
        if len(line_1) != len(line_2):
            raise Exception("Different Length in Two Line to One Line")
        double_line = []
        for i in range(length):
            double_line.append(line_1[i].edge_rename({left: left1, right: right1}).contract(line_2[i].edge_rename({left: left2, right: right2}), {(down, up)}))

        for i in range(length - 1):
            "虽然实际上是range(length - 2), 但是多计算一个以免角标merge的麻烦"
            [u, s, v] = double_line[i].svd({right1, right2}, left, right)
            double_line[i] = v
            double_line[i + 1] = double_line[i + 1].contract(u, {(left1, right1), (left2, right2)}).multiple(s, left, 'u')

        for i in reversed(range(length - 1)):
            [u, s, v] = double_line[i].edge_rename({up: up1, down: down1}) \
                .contract(double_line[i + 1].edge_rename({up: up2, down: down2}), {(right, left)}) \
                .svd({left, up1, down1}, right, left, cut)
            double_line[i + 1] = v.edge_rename({up2: up, down2: down})
            double_line[i] = u.multiple(s, right, 'u').edge_rename({up1: up, down1: down})

        return double_line

    def _get_auxiliaries(self, kind: str, l1: int, l2: int, cut: int) -> Tensor:
        if (kind, l1, l2) not in self._auxiliaries:
            if kind == "up-to-down":
                if l1 == -1:
                    for j in range(self.L2):
                        self._auxiliaries[kind, l1, j] = Tensor(1)
                elif -1 < l1 < self.L1:
                    line_1 = [self._get_auxiliaries(kind, l1 - 1, j, cut) for j in range(self.L2)]
                    line_2 = [self._get_lattice_spin(l1, j) for j in range(self.L2)]
                    result = self._two_line_to_one_line(["Up", "Down", "Left", "Right"], line_1, line_2, cut)
                    for j in range(self.L2):
                        self._auxiliaries[kind, l1, j] = result[j]
                else:
                    raise Exception("Wrong Auxiliaries Position")
            elif kind == "down-to-up":
                if l1 == self.L1:
                    for j in range(self.L2):
                        self._auxiliaries[kind, l1, j] = Tensor(1)
                elif -1 < l1 < self.L1:
                    line_1 = [self._get_auxiliaries(kind, l1 + 1, j, cut) for j in range(self.L2)]
                    line_2 = [self._get_lattice_spin(l1, j) for j in range(self.L2)]
                    result = self._two_line_to_one_line(["Down", "Up", "Left", "Right"], line_1, line_2, cut)
                    for j in range(self.L2):
                        self._auxiliaries[kind, l1, j] = result[j]
                else:
                    raise Exception("Wrong Auxiliaries Position")
            elif kind == "left-to-right":
                if l2 == -1:
                    for i in range(self.L1):
                        self._auxiliaries[kind, i, l2] = Tensor(1)
                elif -1 < l2 < self.L2:
                    line_1 = [self._get_auxiliaries(kind, i, l2 - 1, cut) for i in range(self.L1)]
                    line_2 = [self._get_lattice_spin(i, l2) for i in range(self.L1)]
                    result = self._two_line_to_one_line(["Left", "Right", "Up", "Down"], line_1, line_2, cut)
                    for i in range(self.L1):
                        self._auxiliaries[kind, i, l2] = result[i]
                else:
                    raise Exception("Wrong Auxiliaries Position")
            elif kind == "right-to-left":
                if l2 == self.L2:
                    for i in range(self.L1):
                        self._auxiliaries[kind, i, l2] = Tensor(1)
                elif -1 < l2 < self.L2:
                    line_1 = [self._get_auxiliaries(kind, i, l2 + 1, cut) for i in range(self.L1)]
                    line_2 = [self._get_lattice_spin(i, l2) for i in range(self.L1)]
                    result = self._two_line_to_one_line(["Right", "Left", "Up", "Down"], line_1, line_2, cut)
                    for i in range(self.L1):
                        self._auxiliaries[kind, i, l2] = result[i]
                else:
                    raise Exception("Wrong Auxiliaries Position")
            elif kind == "up-to-down-3":
                if l1 == -1:
                    self._auxiliaries[kind, l1, l2] = Tensor(1)
                elif -1 < l1 < self.L1:
                    """
                    D1 D2 D3
                    |  |  |
                    """
                    self._auxiliaries[kind, l1, l2] = self._get_auxiliaries(kind, l1 - 1, l2, cut) \
                        .contract(self._get_auxiliaries("left-to-right", l1, l2 - 1, cut), {("Down1", "Up")}).edge_rename({"Down": "Down1"}) \
                        .contract(self._get_lattice_spin(l1, l2), {("Down2", "Up"), ("Right", "Left")}).edge_rename({"Down": "Down2"}) \
                        .contract(self._get_auxiliaries("right-to-left", l1, l2 + 1, cut), {("Down3", "Up"), ("Right", "Left")}).edge_rename({"Down": "Down3"})
                else:
                    raise Exception("Wrong Auxiliaries Position In Three Line Type")
            elif kind == "down-to-up-3":
                if l1 == self.L1:
                    self._auxiliaries[kind, l1, l2] = Tensor(1)
                elif -1 < l1 < self.L1:
                    """
                    |  |  |
                    U1 U2 U3
                    """
                    self._auxiliaries[kind, l1, l2] = self._get_auxiliaries(kind, l1 + 1, l2, cut) \
                        .contract(self._get_auxiliaries("left-to-right", l1, l2 - 1, cut), {("Up1", "Down")}).edge_rename({"Up": "Up1"}) \
                        .contract(self._get_lattice_spin(l1, l2), {("Up2", "Down"), ("Right", "Left")}).edge_rename({"Up": "Up2"}) \
                        .contract(self._get_auxiliaries("right-to-left", l1, l2 + 1, cut), {("Up3", "Down"), ("Right", "Left")}).edge_rename({"Up": "Up3"})
                else:
                    raise Exception("Wrong Auxiliaries Position In Three Line Type")
            elif kind == "left-to-right-3":
                if l2 == -1:
                    self._auxiliaries[kind, l1, l2] = Tensor(1)
                elif -1 < l2 < self.L2:
                    """
                    R1 -
                    R2 -
                    R3 -
                    """
                    self._auxiliaries[kind, l1, l2] = self._get_auxiliaries(kind, l1, l2 - 1, cut) \
                        .contract(self._get_auxiliaries("up-to-down", l1 - 1, l2, cut), {("Right1", "Left")}).edge_rename({"Right": "Right1"}) \
                        .contract(self._get_lattice_spin(l1, l2), {("Right2", "Left"), ("Down", "Up")}).edge_rename({"Right": "Right2"}) \
                        .contract(self._get_auxiliaries("down-to-up", l1 + 1, l2, cut), {("Right3", "Left"), ("Down", "Up")}).edge_rename({"Right": "Right3"})
                else:
                    raise Exception("Wrong Auxiliaries Position In Three Line Type")
            elif kind == "right-to-left-3":
                if l2 == self.L2:
                    self._auxiliaries[kind, l1, l2] = Tensor(1)
                elif -1 < l2 < self.L2:
                    """
                    - L1
                    - L2
                    - L3
                    """
                    self._auxiliaries[kind, l1, l2] = self._get_auxiliaries(kind, l1, l2 + 1, cut) \
                        .contract(self._get_auxiliaries("up-to-down", l1 - 1, l2, cut), {("Left1", "Right")}).edge_rename({"Left": "Left1"}) \
                        .contract(self._get_lattice_spin(l1, l2), {("Left2", "Right"), ("Down", "Up")}).edge_rename({"Left": "Left2"}) \
                        .contract(self._get_auxiliaries("down-to-up", l1 + 1, l2, cut), {("Left3", "Right"), ("Down", "Up")}).edge_rename({"Left": "Left3"})
                else:
                    raise Exception("Wrong Auxiliaries Position In Three Line Type")
            else:
                raise Exception("Wrong Auxiliaries Kind")
        return self._auxiliaries[kind, l1, l2]

    def _try_to_delete_auxiliaries(self, index: str, i: int, j: int) -> bool:
        if (index, i, j) in self._auxiliaries:
            del self._auxiliaries[index, i, j]
            return True
        else:
            return False

    def _refresh_auxiliaries(self, l1: int, l2: int) -> None:
        self._lattice_spin[l1][l2] = None
        self._refresh_line("right", l2)
        self._refresh_line("left", l2)
        self._refresh_line("down", l1)
        self._refresh_line("up", l1)
        for i in range(self.L1):
            if i < l1:
                self._try_to_delete_auxiliaries("down-to-up-3", i, l2)
            elif i > l1:
                self._try_to_delete_auxiliaries("up-to-down-3", i, l2)
            else:
                self._try_to_delete_auxiliaries("down-to-up-3", i, l2)
                self._try_to_delete_auxiliaries("up-to-down-3", i, l2)
        for j in range(self.L2):
            if j < l2:
                self._try_to_delete_auxiliaries("right-to-left-3", l1, j)
            elif j > l2:
                self._try_to_delete_auxiliaries("left-to-right-3", l1, j)
            else:
                self._try_to_delete_auxiliaries("right-to-left-3", l1, j)
                self._try_to_delete_auxiliaries("left-to-right-3", l1, j)

    def _refresh_line(self, kind: str, index: int) -> None:
        if kind == "right":
            if index != self.L2:
                flag = False
                for i in range(self.L1):
                    flag = self._try_to_delete_auxiliaries("left-to-right", i, index)
                    self._try_to_delete_auxiliaries("up-to-down-3", i, index + 1)
                    self._try_to_delete_auxiliaries("down-to-up-3", i, index + 1)
                if flag:
                    self._refresh_line(kind, index + 1)
        elif kind == "left":
            if index != -1:
                flag = False
                for i in range(self.L1):
                    flag = self._try_to_delete_auxiliaries("right-to-left", i, index)
                    self._try_to_delete_auxiliaries("up-to-down-3", i, index - 1)
                    self._try_to_delete_auxiliaries("down-to-up-3", i, index - 1)
                if flag:
                    self._refresh_line(kind, index - 1)
        elif kind == "down":
            if index != self.L1:
                flag = False
                for j in range(self.L2):
                    flag = self._try_to_delete_auxiliaries("up-to-down", index, j)
                    self._try_to_delete_auxiliaries("left-to-right-3", index + 1, j)
                    self._try_to_delete_auxiliaries("right-to-left-3", index + 1, j)
                if flag:
                    self._refresh_line(kind, index + 1)
        elif kind == "up":
            if index != -1:
                flag = False
                for j in range(self.L2):
                    flag = self._try_to_delete_auxiliaries("down-to-up", index, j)
                    self._try_to_delete_auxiliaries("left-to-right-3", index - 1, j)
                    self._try_to_delete_auxiliaries("right-to-left-3", index - 1, j)
                if flag:
                    self._refresh_line(kind, index + 1)
        else:
            raise Exception("Wrong Type in Refresh Line")

    def __init__(self, L1: int = 4, L2: int = 4, D: int = 4):
        self.L1: int = L1
        self.L2: int = L2
        self.D: int = D

        self.lattice: List[List[Tensor]] = [[self._create_tensor(l1, l2) for l2 in range(self.L2)] for l1 in range(self.L1)]

        self.spin: List[List[int]] = [[0 for _ in range(self.L2)] for _ in range(self.L1)]
        self._initialize_spin()

        self._lattice_spin: List[List[Optional[Tensor]]] = [[None for _ in range(self.L2)] for _ in range(self.L1)]

        self._auxiliaries: Dict[Tuple[str, int, int], Tensor] = {}

        self.environment: Dict[Tuple[str, int, int], Singular] = {}

    def _absorb_environment(self, *, remove: bool = True, division: bool = False) -> None:
        self.refresh_all_auxiliaries()
        for direction in ["Right", "Down"]:
            for l1 in range(self.L1):
                for l2 in range(self.L2):
                    if (direction, l1, l2) in self.environment:
                        self.lattice[l1][l2].multiple(self.environment[direction, l1, l2], direction, "u", division)
                        if remove:
                            del self.environment[direction, l1, l2]

    def _single_term_simple_update(self, updater: Tensor, direction: str, l1: int, l2: int) -> None:
        if direction == "Right":
            """
             22
            1LR1
             33
            """
            left = self.lattice[l1][l2]
            if ("Right", l1, l2 - 1) in self.environment:
                left.multiple(self.environment["Right", l1, l2 - 1], "Left", "v")
            if ("Down", l1 - 1, l2) in self.environment:
                left.multiple(self.environment["Down", l1 - 1, l2], "Up", "v")
            if ("Down", l1, l2) in self.environment:
                left.multiple(self.environment["Down", l1, l2], "Down", "u")
            if ("Right", l1, l2) in self.environment:
                left.multiple(self.environment["Right", l1, l2], "Right", "u")
            right = self.lattice[l1][l2 + 1]
            if ("Right", l1, l2 + 1) in self.environment:
                right.multiple(self.environment["Right", l1, l2 + 1], "Right", "u")
            if ("Down", l1 - 1, l2 + 1) in self.environment:
                right.multiple(self.environment["Down", l1 - 1, l2 + 1], "Up", "v")
            if ("Down", l1, l2 + 1) in self.environment:
                right.multiple(self.environment["Down", l1, l2 + 1], "Down", "u")
            u: Tensor
            s: Singular
            v: Tensor
            u, s, v = left.edge_rename({"Up": "Up1", "Down": "Down1", "Phy": "Phy0"}) \
                .contract(right.edge_rename({"Up": "Up2", "Down": "Down2", "Phy": "Phy1"}), {("Right", "Left")}) \
                .contract(updater, {("Phy0", "I0"), ("Phy1", "I1")}) \
                .svd({"Left", "Up1", "Down1", "O0"}, "Right", "Left", self.D)
            u /= u.norm_max()
            v /= v.norm_max()
            s.normalize_max()
            self.environment["Right", l1, l2] = s
            self.lattice[l1][l2] = u.edge_rename({"Up1": "Up", "Down1": "Down", "O0": "Phy"})
            if ("Right", l1, l2 - 1) in self.environment:
                self.lattice[l1][l2].multiple(self.environment["Right", l1, l2 - 1], "Left", "v", True)
            if ("Down", l1 - 1, l2) in self.environment:
                self.lattice[l1][l2].multiple(self.environment["Down", l1 - 1, l2], "Up", "v", True)
            if ("Down", l1, l2) in self.environment:
                self.lattice[l1][l2].multiple(self.environment["Down", l1, l2], "Down", "u", True)
            self.lattice[l1][l2 + 1] = v.edge_rename({"Up2": "Up", "Down2": "Down", "O1": "Phy"})
            if ("Right", l1, l2 + 1) in self.environment:
                self.lattice[l1][l2 + 1].multiple(self.environment["Right", l1, l2 + 1], "Right", "u", True)
            if ("Down", l1 - 1, l2 + 1) in self.environment:
                self.lattice[l1][l2 + 1].multiple(self.environment["Down", l1 - 1, l2 + 1], "Up", "v", True)
            if ("Down", l1, l2 + 1) in self.environment:
                self.lattice[l1][l2 + 1].multiple(self.environment["Down", l1, l2 + 1], "Down", "u", True)
        elif direction == "Down":
            """
             1
            2U3
            2D3
             1
            """
            up = self.lattice[l1][l2]
            if ("Down", l1 - 1, l2) in self.environment:
                up.multiple(self.environment["Down", l1 - 1, l2], "Up", "v")
            if ("Right", l1, l2 - 1) in self.environment:
                up.multiple(self.environment["Right", l1, l2 - 1], "Left", "v")
            if ("Right", l1, l2) in self.environment:
                up.multiple(self.environment["Right", l1, l2], "Right", "u")
            if ("Down", l1, l2) in self.environment:
                up.multiple(self.environment["Down", l1, l2], "Down", "u")
            down = self.lattice[l1 + 1][l2]
            if ("Down", l1 + 1, l2) in self.environment:
                down.multiple(self.environment["Down", l1 + 1, l2], "Down", "u")
            if ("Right", l1 + 1, l2 - 1) in self.environment:
                down.multiple(self.environment["Right", l1 + 1, l2 - 1], "Left", "v")
            if ("Right", l1 + 1, l2) in self.environment:
                down.multiple(self.environment["Right", l1 + 1, l2], "Right", "u")
            u: Tensor
            s: Singular
            v: Tensor
            u, s, v = up.edge_rename({"Left": "Left1", "Right": "Right1", "Phy": "Phy0"}) \
                .contract(down.edge_rename({"Left": "Left2", "Right": "Right2", "Phy": "Phy1"}), {("Down", "Up")}) \
                .contract(updater, {("Phy0", "I0"), ("Phy1", "I1")}) \
                .svd({"Up", "Left1", "Right1", "O0"}, "Down", "Up", self.D)
            u /= u.norm_max()
            v /= v.norm_max()
            s.normalize_max()
            self.environment["Down", l1, l2] = s
            self.lattice[l1][l2] = u.edge_rename({"Left1": "Left", "Right1": "Right", "O0": "Phy"})
            if ("Down", l1 - 1, l2) in self.environment:
                self.lattice[l1][l2].multiple(self.environment["Down", l1 - 1, l2], "Up", "v", True)
            if ("Right", l1, l2 - 1) in self.environment:
                self.lattice[l1][l2].multiple(self.environment["Right", l1, l2 - 1], "Left", "v", True)
            if ("Right", l1, l2) in self.environment:
                self.lattice[l1][l2].multiple(self.environment["Right", l1, l2], "Right", "u", True)
            self.lattice[l1 + 1][l2] = v.edge_rename({"Left2": "Left", "Right2": "Right", "O1": "Phy"})
            if ("Down", l1 + 1, l2) in self.environment:
                self.lattice[l1 + 1][l2].multiple(self.environment["Down", l1 + 1, l2], "Down", "u", True)
            if ("Right", l1 + 1, l2 - 1) in self.environment:
                self.lattice[l1 + 1][l2].multiple(self.environment["Right", l1 + 1, l2 - 1], "Left", "v", True)
            if ("Right", l1 + 1, l2) in self.environment:
                self.lattice[l1 + 1][l2].multiple(self.environment["Right", l1 + 1, l2], "Right", "u", True)
        else:
            raise Exception("Wrong direction in Simple Update")

    def _single_simple_update(self, updater: Tensor) -> None:
        # LR
        for l1 in range(self.L1):
            for l2 in range(self.L2 - 1):
                self._single_term_simple_update(updater, "Right", l1, l2)
        # UD
        for l2 in range(self.L2):
            for l1 in range(self.L1 - 1):
                self._single_term_simple_update(updater, "Down", l1, l2)
        # DU
        for l2 in reversed(range(self.L2)):
            for l1 in reversed(range(self.L1 - 1)):
                self._single_term_simple_update(updater, "Down", l1, l2)
        # RL
        for l1 in reversed(range(self.L1)):
            for l2 in reversed(range(self.L2 - 1)):
                self._single_term_simple_update(updater, "Right", l1, l2)

    def simple_update(self, time: int, delta_t: float, *, new_D: Optional[int] = None):
        if new_D is not None:
            self.D = new_D
        updater = self.identity - delta_t * self.hamiltonian - delta_t * delta_t * self.hamiltonian_square / 2
        for step in range(time):
            self._single_simple_update(updater)
            print(step)
        return self

    def _get_lattice_spin(self, l1: int, l2: int, index: int = -1) -> Tensor:
        if index == -1:
            if not self._lattice_spin[l1][l2]:
                self._lattice_spin[l1][l2] = self.lattice[l1][l2].contract(self.raw_base[self.spin[l1][l2]], {("Phy", "Phy")})
            return self._lattice_spin[l1][l2]
        else:
            return self.lattice[l1][l2].contract(self.raw_base[index], {("Phy", "Phy")})

    def save(self, file_name: str):
        with open(file_name, "wb") as file:
            pickle.dump(self, file)

    def gradient_descent(self, time: int, markov_chain_length: int, cut: int, *, log: Optional[str] = None, energy_only: bool = False):
        if log:
            file = open(log, "a")
        if energy_only:
            self._absorb_environment(remove=False, division=False)
        else:
            self._absorb_environment(remove=True, division=False)
        # AdaDelta
        gradient_s = [[j.same_shape().zero() for j in i] for i in self.lattice]
        gradient_delta = [[j.same_shape().zero() for j in i] for i in self.lattice]
        rho = 0.9
        eps = 1e5
        # AdaDelta
        for step in range(time):
            energy, gradient = self._markov_chain(markov_chain_length, cut, energy_only)
            if not energy_only:
                for l1 in range(self.L1):
                    for l2 in range(self.L2):
                        # AdaDelta
                        gradient_s[l1][l2] = rho * gradient_s[l1][l2] + (1 - rho) * gradient[l1][l2] * gradient[l1][l2]
                        g = ((gradient_delta[l1][l2] + eps) / (gradient_s[l1][l2] + eps)).sqrt() * gradient[l1][l2]
                        self.lattice[l1][l2] -= g
                        gradient_delta[l1][l2] = rho * gradient_delta[l1][l2] + (1 - rho) * g * g
                        # AdaDelta
                self.refresh_all_auxiliaries()
            if log:
                print(energy / (self.L1 * self.L2), file=file)
            print(step, energy / (self.L1 * self.L2))
        if energy_only:
            self._absorb_environment(remove=False, division=True)
        if log:
            file.close()
        return self

    def refresh_all_auxiliaries(self) -> None:
        for l1 in range(self.L1):
            for l2 in range(self.L2):
                self._refresh_auxiliaries(l1, l2)

    def _markov_chain(self, markov_chain_length: int, cut: int, energy_only: bool) -> Tuple[float, List[List[Tensor]]]:
        summation_of_spin_energy = 0.
        summation_of_spin_gradient = [[j.same_shape().zero() for j in i] for i in self.lattice]
        summation_of_product = [[j.same_shape().zero() for j in i] for i in self.lattice]

        for _ in range(markov_chain_length):
            energy, gradient = self._single_markov_chain(cut, energy_only)
            summation_of_spin_energy += energy
            if not energy_only:
                for l1 in range(self.L1):
                    for l2 in range(self.L2):
                        this_gradient = gradient[l1][l2].contract(self.raw_base[self.spin[l1][l2]], set())
                        summation_of_spin_gradient[l1][l2] += this_gradient
                        summation_of_product[l1][l2] += this_gradient * energy

        energy = summation_of_spin_energy / markov_chain_length
        if energy_only:
            return energy, [[]]
        else:
            return energy, \
                   [[2 * summation_of_product[l1][l2] / markov_chain_length - 2 * energy * summation_of_spin_gradient[l1][l2] / markov_chain_length \
                     for l2 in range(self.L2)] for l1 in range(self.L1)]

    def _get_wss(self, point1: Tuple[int, int], point2: Tuple[int, int], cut: int, new_index) -> float:
        p1l1, p1l2 = point1
        p2l1, p2l2 = point2
        if p1l1 == p2l1:
            if p2l2 != p1l2 + 1:
                raise Exception("Hopping Style Not Implement")
            wss = self._get_auxiliaries("left-to-right-3", p1l1, p1l2 - 1, cut) \
                .contract(self._get_auxiliaries("up-to-down", p1l1 - 1, p1l2, cut), {("Right1", "Left")}).edge_rename({"Right": "Right1"}) \
                .contract(self._get_lattice_spin(p1l1, p1l2, new_index[0]), {("Right2", "Left"), ("Down", "Up")}).edge_rename({"Right": "Right2"}) \
                .contract(self._get_auxiliaries("down-to-up", p1l1 + 1, p1l2, cut), {("Right3", "Left"), ("Down", "Up")}).edge_rename({"Right": "Right3"}) \
                .contract(self._get_auxiliaries("up-to-down", p2l1 - 1, p2l2, cut), {("Right1", "Left")}).edge_rename({"Right": "Right1"}) \
                .contract(self._get_lattice_spin(p2l1, p2l2, new_index[1]), {("Right2", "Left"), ("Down", "Up")}).edge_rename({"Right": "Right2"}) \
                .contract(self._get_auxiliaries("down-to-up", p2l1 + 1, p2l2, cut), {("Right3", "Left"), ("Down", "Up")}).edge_rename({"Right": "Right3"}) \
                .contract(self._get_auxiliaries("right-to-left-3", p2l1, p2l2 + 1, cut), {("Right1", "Left1"), ("Right2", "Left2"), ("Right3", "Left3")}).value()
            return wss
        else:
            if p1l2 != p2l2 or p1l1 + 1 != p2l1:
                raise Exception("Hopping Style Not Implement")
            wss = self._get_auxiliaries("up-to-down-3", p1l1 - 1, p1l2, cut) \
                .contract(self._get_auxiliaries("left-to-right", p1l1, p1l2 - 1, cut), {("Down1", "Up")}).edge_rename({"Down": "Down1"}) \
                .contract(self._get_lattice_spin(p1l1, p1l2, new_index[0]), {("Down2", "Up"), ("Right", "Left")}).edge_rename({"Down": "Down2"}) \
                .contract(self._get_auxiliaries("right-to-left", p1l1, p1l2 + 1, cut), {("Down3", "Up"), ("Right", "Left")}).edge_rename({"Down": "Down3"}) \
                .contract(self._get_auxiliaries("left-to-right", p2l1, p2l2 - 1, cut), {("Down1", "Up")}).edge_rename({"Down": "Down1"}) \
                .contract(self._get_lattice_spin(p2l1, p2l2, new_index[1]), {("Down2", "Up"), ("Right", "Left")}).edge_rename({"Down": "Down2"}) \
                .contract(self._get_auxiliaries("right-to-left", p2l1, p2l2 + 1, cut), {("Down3", "Up"), ("Right", "Left")}).edge_rename({"Down": "Down3"}) \
                .contract(self._get_auxiliaries("down-to-up-3", p2l1 + 1, p2l2, cut), {("Down1", "Up1"), ("Down2", "Up2"), ("Down3", "Up3")}).value()
            return wss

    def _try_hop(self, point1: Tuple[int, int], point2: Tuple[int, int], cut: int, ws: Optional[float] = None) -> float:
        p1l1, p1l2 = point1
        p2l1, p2l2 = point2
        if p1l1 == p2l1:
            """
            X X X X
            X 1 2 X
            X X X X
            """
            if p2l2 != p1l2 + 1:
                raise Exception("Hopping Style Not Implement")
            if ws is None:
                ws = self._get_auxiliaries("left-to-right-3", p1l1, p1l2, cut) \
                    .contract(self._get_auxiliaries("right-to-left-3", p2l1, p2l2, cut),
                              {("Right1", "Left1"), ("Right2", "Left2"), ("Right3", "Left3")}).value()
            new_index = np.random.randint(2, size=[2])
            wss = self._get_wss(point1, point2, cut, new_index)
            if wss ** 2 / ws ** 2 > np.random.rand():
                if self.spin[p1l1][p1l2] != new_index[0]:
                    self.spin[p1l1][p1l2] = new_index[0]
                    self._refresh_auxiliaries(p1l1, p1l2)
                if self.spin[p2l1][p2l2] != new_index[1]:
                    self.spin[p2l1][p2l2] = new_index[1]
                    self._refresh_auxiliaries(p2l1, p2l2)
                return wss
            else:
                return ws
        else:
            """
            X X X
            X 1 X
            X 2 X
            X X X
            """
            if p1l2 != p2l2 or p1l1 + 1 != p2l1:
                raise Exception("Hopping Style Not Implement")
            if ws is None:
                ws = self._get_auxiliaries("up-to-down-3", p1l1, p1l2, cut) \
                    .contract(self._get_auxiliaries("down-to-up-3", p2l1, p2l2, cut),
                              {("Down1", "Up1"), ("Down2", "Up2"), ("Down3", "Up3")}).value()
            new_index = np.random.randint(2, size=[2])
            wss = self._get_wss(point1, point2, cut, new_index)
            if wss ** 2 / ws ** 2 > np.random.rand():
                if self.spin[p1l1][p1l2] != new_index[0]:
                    self.spin[p1l1][p1l2] = new_index[0]
                    self._refresh_auxiliaries(p1l1, p1l2)
                if self.spin[p2l1][p2l2] != new_index[1]:
                    self.spin[p2l1][p2l2] = new_index[1]
                    self._refresh_auxiliaries(p2l1, p2l2)
                return wss
            else:
                return ws

    def _single_markov_chain(self, cut: int, energy_only: bool) -> Tuple[float, List[List[Tensor]]]:
        ws: Optional[float] = None
        # LR
        for l1 in range(self.L1):
            for l2 in range(self.L2 - 1):
                ws = self._try_hop((l1, l2), (l1, l2 + 1), cut, ws)
        # UD
        for l2 in range(self.L2):
            for l1 in range(self.L1 - 1):
                ws = self._try_hop((l1, l2), (l1 + 1, l2), cut, ws)
        # DU
        for l2 in reversed(range(self.L2)):
            for l1 in reversed(range(self.L1 - 1)):
                ws = self._try_hop((l1, l2), (l1 + 1, l2), cut, ws)
        # RL
        for l1 in reversed(range(self.L1)):
            for l2 in reversed(range(self.L2 - 1)):
                ws = self._try_hop((l1, l2), (l1, l2 + 1), cut, ws)
        # collect data
        """
         1  0  0  0
         0 -1  2  0
         0  2 -1  0
         0  0  0  1
        /4
        """
        energy = 0.
        for l1 in range(self.L1):
            for l2 in range(self.L2 - 1):
                if self.spin[l1][l2] == self.spin[l1][l2 + 1]:
                    energy += ws / 4
                else:
                    energy -= ws / 4
                    wss = self._get_wss((l1, l2), (l1, l2 + 1), cut, [self.spin[l1][l2 + 1], self.spin[l1][l2]])
                    energy += wss * 2 / 4
        for l2 in range(self.L2):
            for l1 in range(self.L1 - 1):
                if self.spin[l1][l2] == self.spin[l1 + 1][l2]:
                    energy += ws / 4
                else:
                    energy -= ws / 4
                    wss = self._get_wss((l1, l2), (l1 + 1, l2), cut, [self.spin[l1 + 1][l2], self.spin[l1][l2]])
                    energy += wss * 2 / 4
        energy /= ws
        if energy_only:
            return energy, [[]]
        else:
            gradient = [[
                self._get_auxiliaries("up-to-down-3", l1 - 1, l2, cut) \
                    .contract(self._get_auxiliaries("left-to-right", l1, l2 - 1, cut), {("Down1", "Up")}) \
                    .contract(self._get_auxiliaries("down-to-up-3", l1 + 1, l2, cut), {("Down", "Up1")}) \
                    .contract(self._get_auxiliaries("right-to-left", l1, l2 + 1, cut), {("Down3", "Up"), ("Up3", "Down")}) \
                    .edge_rename({"Down2": "Up", "Up2": "Down", "Left": "Right", "Right": "Left"}) / ws
                for l2 in range(self.L2)] for l1 in range(self.L1)]
            return energy, gradient

    def end(self):
        pass

    @staticmethod
    def load(file_name: str):
        with open(file_name, "rb") as file:
            return pickle.load(file)


if __name__ == '__main__':
    # import cProfile
    # cProfile.run('fire.Fire({"new": TwoDimensionHeisenberg, "load": load, "test": test})', filename="profile.dat")
    import fire
    def Display(lines, out):
        text = "\n".join(lines) + "\n"
        out.write(text)
    fire.core.Display = Display
    fire.Fire({"new": TwoDimensionHeisenberg, "load": TwoDimensionHeisenberg.load})
