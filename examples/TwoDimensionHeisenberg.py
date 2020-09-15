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
import numpy as np
import pickle
from TAT.Tensor import DNo as Tensor
import fire

raw_base = [Tensor(["Phy"], [2]) for _ in [0, 0]]
raw_base[0].block()[:] = [1, 0]
raw_base[1].block()[:] = [0, 1]


class TwoDimensionHeisenberg():
    __slots__ = ["L1", "L2", "D", "lattice", "spin", "_auxiliaries", "_lattice_spin"]

    def _create_tensor(self, l1, l2):
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

    def _initialize_spin(self):
        for l1 in range(self.L1):
            for l2 in range(self.L2):
                self.spin[l1][l2] = (l1 + l2) % 2
        if False:
            total_size = self.L1 * self.L2
            half_size = total_size // 2
            for _ in range(half_size):
                flag = True
                while flag:
                    l1 = np.random.randint(self.L1)
                    l2 = np.random.randint(self.L2)
                    if self.spin[l1][l2] == 0:
                        flag = False
                        self.spin[l1][l2] = 1

    @staticmethod
    def _two_line_to_one_line(udlr_name, line_1, line_2, cut):
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
            [u, s, v] = double_line[i].edge_rename({up: up1, down: down1})\
                .contract(double_line[i + 1].edge_rename({up: up2, down: down2}), {(right, left)})\
                .svd({left, up1, down1}, right, left, cut)
            double_line[i + 1] = v.edge_rename({up2: up, down2: down})
            double_line[i] = u.multiple(s, right, 'u').edge_rename({up1: up, down1: down})

        return double_line

    def _get_auxiliaries(self, kind, l1, l2, cut):
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
                    self._auxiliaries[kind, l1, l2] = self._get_auxiliaries(kind, l1 - 1, l2, cut)\
                        .contract(self._get_auxiliaries("left-to-right", l1, l2 - 1, cut), {("Down1", "Up")}).edge_rename({"Down": "Down1"})\
                        .contract(self._get_lattice_spin(l1, l2), {("Down2", "Up"), ("Right", "Left")}).edge_rename({"Down": "Down2"})\
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
                    self._auxiliaries[kind, l1, l2] = self._get_auxiliaries(kind, l1 + 1, l2, cut)\
                        .contract(self._get_auxiliaries("left-to-right", l1, l2 - 1, cut), {("Up1", "Down")}).edge_rename({"Up": "Up1"})\
                        .contract(self._get_lattice_spin(l1, l2), {("Up2", "Down"), ("Right", "Left")}).edge_rename({"Up": "Up2"})\
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
                    self._auxiliaries[kind, l1, l2] = self._get_auxiliaries(kind, l1, l2 - 1, cut)\
                        .contract(self._get_auxiliaries("up-to-down", l1 - 1, l2, cut), {("Right1", "Left")}).edge_rename({"Right": "Right1"})\
                        .contract(self._get_lattice_spin(l1, l2), {("Right2", "Left"), ("Down", "Up")}).edge_rename({"Right": "Right2"})\
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
                    self._auxiliaries[kind, l1, l2] = self._get_auxiliaries(kind, l1, l2 + 1, cut)\
                        .contract(self._get_auxiliaries("up-to-down", l1 - 1, l2, cut), {("Left1", "Right")}).edge_rename({"Left": "Left1"})\
                        .contract(self._get_lattice_spin(l1, l2), {("Left2", "Right"), ("Down", "Up")}).edge_rename({"Left": "Left2"})\
                        .contract(self._get_auxiliaries("down-to-up", l1 + 1, l2, cut), {("Left3", "Right"), ("Down", "Up")}).edge_rename({"Left": "Left3"})
                else:
                    raise Exception("Wrong Auxiliaries Position In Three Line Type")
            else:
                raise Exception("Wrong Auxiliaries Kind")
        return self._auxiliaries[kind, l1, l2]

    def _try_to_delete_auxiliaries(self, index, i, j):
        if (index, i, j) in self._auxiliaries:
            del self._auxiliaries[index, i, j]
            return True
        else:
            return False

    def _refresh_auxiliaries(self, l1, l2):
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

    def _refresh_line(self, kind, index):
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
        self.L1 = L1
        self.L2 = L2
        self.D = D

        self.lattice = [[self._create_tensor(l1, l2) for l2 in range(self.L2)] for l1 in range(self.L1)]

        self.spin = [[0 for l2 in range(self.L2)] for l1 in range(self.L1)]
        self._initialize_spin()

        self._lattice_spin = [[None for l2 in range(self.L2)] for l1 in range(self.L1)]

        self._auxiliaries = {}

    def _get_lattice_spin(self, l1, l2, index=-1):
        if index == -1:
            if not self._lattice_spin[l1][l2]:
                self._lattice_spin[l1][l2] = self.lattice[l1][l2].contract(raw_base[self.spin[l1][l2]], {("Phy", "Phy")})
            return self._lattice_spin[l1][l2]
        else:
            return self.lattice[l1][l2].contract(raw_base[index], {("Phy", "Phy")})

    def save(self, file_name: str):
        with open(file_name, "wb") as file:
            pickle.dump(self, file)

    def gradient_descent(self, *, time: int, step_size: float, markov_chain_length: int, cut: int, log: str = None):
        if log:
            file = open(log, "a")
        for _ in range(time):
            energy, gradient = self._markov_chain(markov_chain_length, cut)
            for l1 in range(self.L1):
                for l2 in range(self.L2):
                    self.lattice[l1][l2] -= step_size * gradient[l1][l2]
            if log:
                print(energy / (self.L1 * self.L2), file=file)
            print(energy / (self.L1 * self.L2))
        if log:
            file.close()
        return self

    def _markov_chain(self, markov_chain_length, cut):
        summation_of_spin_energy = 0
        summation_of_spin_gradient = [[j.same_shape().zero() for j in i] for i in self.lattice]
        summation_of_product = [[j.same_shape().zero() for j in i] for i in self.lattice]

        for _ in range(markov_chain_length):
            energy, gradient = self._single_markov_chain(cut)
            summation_of_spin_energy += energy
            for l1 in range(self.L1):
                for l2 in range(self.L2):
                    this_gradient = gradient[l1][l2].contract(raw_base[self.spin[l1][l2]], set())
                    summation_of_spin_gradient[l1][l2] += this_gradient
                    summation_of_product[l1][l2] += this_gradient * energy

        energy = summation_of_spin_energy / markov_chain_length
        return energy, [[2 * summation_of_product[l1][l2] / markov_chain_length - 2 * energy * summation_of_spin_gradient[l1][l2] / markov_chain_length\
                         for l2 in range(self.L2)] for l1 in range(self.L1)]

    def _get_wss(self, point1, point2, cut, new_index):
        p1l1, p1l2 = point1
        p2l1, p2l2 = point2
        if p1l1 == p2l1:
            if p2l2 != p1l2 + 1:
                raise Exception("Hopping Style Not Implement")
            wss = self._get_auxiliaries("left-to-right-3", p1l1, p1l2 - 1, cut)\
                .contract(self._get_auxiliaries("up-to-down", p1l1 - 1, p1l2, cut), {("Right1", "Left")}).edge_rename({"Right": "Right1"})\
                .contract(self._get_lattice_spin(p1l1, p1l2, new_index[0]), {("Right2", "Left"), ("Down", "Up")}).edge_rename({"Right": "Right2"})\
                .contract(self._get_auxiliaries("down-to-up", p1l1 + 1, p1l2, cut), {("Right3", "Left"), ("Down", "Up")}).edge_rename({"Right": "Right3"})\
                .contract(self._get_auxiliaries("up-to-down", p2l1 - 1, p2l2, cut), {("Right1", "Left")}).edge_rename({"Right": "Right1"})\
                .contract(self._get_lattice_spin(p2l1, p2l2, new_index[1]), {("Right2", "Left"), ("Down", "Up")}).edge_rename({"Right": "Right2"})\
                .contract(self._get_auxiliaries("down-to-up", p2l1 + 1, p2l2, cut), {("Right3", "Left"), ("Down", "Up")}).edge_rename({"Right": "Right3"})\
                .contract(self._get_auxiliaries("right-to-left-3", p2l1, p2l2 + 1, cut), {("Right1", "Left1"), ("Right2", "Left2"), ("Right3", "Left3")}).value()
            return wss
        else:
            if p1l2 != p2l2 or p1l1 + 1 != p2l1:
                raise Exception("Hopping Style Not Implement")
            wss = self._get_auxiliaries("up-to-down-3", p1l1 - 1, p1l2, cut)\
                .contract(self._get_auxiliaries("left-to-right", p1l1, p1l2 - 1, cut), {("Down1", "Up")}).edge_rename({"Down": "Down1"})\
                .contract(self._get_lattice_spin(p1l1, p1l2, new_index[0]), {("Down2", "Up"),("Right", "Left")}).edge_rename({"Down": "Down2"})\
                .contract(self._get_auxiliaries("right-to-left", p1l1, p1l2 + 1, cut), {("Down3", "Up"), ("Right", "Left")}).edge_rename({"Down": "Down3"})\
                .contract(self._get_auxiliaries("left-to-right", p2l1, p2l2 - 1, cut), {("Down1", "Up")}).edge_rename({"Down": "Down1"})\
                .contract(self._get_lattice_spin(p2l1, p2l2, new_index[1]), {("Down2", "Up"),("Right", "Left")}).edge_rename({"Down": "Down2"})\
                .contract(self._get_auxiliaries("right-to-left", p2l1, p2l2 + 1, cut), {("Down3", "Up"), ("Right", "Left")}).edge_rename({"Down": "Down3"})\
                .contract(self._get_auxiliaries("down-to-up-3", p2l1 + 1, p2l2, cut), {("Down1", "Up1"), ("Down2", "Up2"), ("Down3", "Up3")}).value()
            return wss

    def _try_hop(self, point1, point2, cut, ws=None):
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
                ws = self._get_auxiliaries("left-to-right-3", p1l1, p1l2, cut)\
                    .contract(self._get_auxiliaries("right-to-left-3", p2l1, p2l2, cut),\
                              {("Right1", "Left1"), ("Right2", "Left2"), ("Right3", "Left3")}).value()
            new_index = np.random.randint(2, size=[2])
            wss = self._get_wss(point1, point2, cut, new_index)
            if wss**2 / ws**2 > np.random.rand():
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
                ws = self._get_auxiliaries("up-to-down-3", p1l1, p1l2, cut)\
                    .contract(self._get_auxiliaries("down-to-up-3", p2l1, p2l2, cut),\
                              {("Down1", "Up1"), ("Down2", "Up2"), ("Down3", "Up3")}).value()
            new_index = np.random.randint(2, size=[2])
            wss = self._get_wss(point1, point2, cut, new_index)
            if wss**2 / ws**2 > np.random.rand():
                if self.spin[p1l1][p1l2] != new_index[0]:
                    self.spin[p1l1][p1l2] = new_index[0]
                    self._refresh_auxiliaries(p1l1, p1l2)
                if self.spin[p2l1][p2l2] != new_index[1]:
                    self.spin[p2l1][p2l2] = new_index[1]
                    self._refresh_auxiliaries(p2l1, p2l2)
                return wss
            else:
                return ws

    def _single_markov_chain(self, cut):
        ws = None
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
        energy = 0
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
        gradient = [[
            self._get_auxiliaries("up-to-down-3", l1 - 1, l2, cut)\
            .contract(self._get_auxiliaries("left-to-right", l1, l2 - 1, cut), {("Down1", "Up")})\
            .contract(self._get_auxiliaries("down-to-up-3", l1 + 1, l2, cut), {("Down", "Up1")})\
            .contract(self._get_auxiliaries("right-to-left", l1, l2 + 1, cut), {("Down3", "Up"), ("Up3", "Down")})\
            .edge_rename({"Down2": "Up", "Up2": "Down", "Left": "Right", "Right": "Left"}) / ws
            for l2 in range(self.L2)] for l1 in range(self.L1)]
        # print(self.spin, energy)
        return energy, gradient

    def end(self):
        pass

    @staticmethod
    def load(file_name: str):
        with open(file_name, "rb") as file:
            result = pickle.load(file)
            result._refresh_auxiliaries(0, 0)
            result._refresh_auxiliaries(result.L1 - 1, result.L2 - 2)
            return result


if __name__ == '__main__':
    #import cProfile
    #cProfile.run('fire.Fire({"new": TwoDimensionHeisenberg, "load": load, "test": test})', filename="profile.dat")
    fire.Fire({"new": TwoDimensionHeisenberg, "load": TwoDimensionHeisenberg.load})
