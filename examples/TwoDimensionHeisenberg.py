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
    __slots__ = ["L1", "L2", "D", "lattice", "spin", "auxiliaries"]

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
    def _two_line_to_one_line(line_1, line_2, udlr_name, cut):
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
            double_line.push_back(line_1[i].edge_rename({left: left1, right: right1}), line_2[i].edge_rename({left: left2, right: right2}), {(down, up)})

        for i in range(length - 1):
            [u, s, v] = double_line[i].svd({right1, right2}, left, right)
            double_line[i] = v
            double_line[i + 1] = double_line[i + 1].contract(u, {(left1, right1), (left2, right2)}).multiple(s, left, 'u')

        for i in range(length - 2, -1, -1):
            [u, s, v] = double[i].edge_rename({up: up1, down: down1})\
                .contract(double_line[i + 1].edge_rename({up: up2, down: down2}), {(right, left)})\
                .svd({left, up1, down1}, right, left)
            double_line[i + 1] = v.edge_rename({up2: up, down2: down})
            double_line[i] = u.multiple(s, right, 'u').edge_rename({up1: up, down1: down})

        return double_line

    def _get_auxiliaries(self, kind, l1, l2, cut):
        if (kind, l1, l2) not in self.auxiliaries:
            if kind == "up-to-down":
                if l1 == -1:
                    for j in range(self.L2):
                        self.auxiliaries[kind, -1, j] = Tensor(1)
                elif l1 < self.L1:
                    line_1 = [self._get_auxiliaries(kind, l1 - 1, j, cut) for j in range(self.L2)]
                    line_2 = [self.lattice[l1, j].contract(raw_base[self.spin[l1][j]], {("Phy", "Phy")}) for j in range(self.L2)]
                    result = _two_line_to_one_line(["u", "d", "l", "r"], line_1, line_2, cut)
                    for j in range(self.L2):
                        self.auxiliaries[kind, l1, j] = result[j]
                else:
                    raise Exception("Wrong Auxiliaries Position")
            elif kind == "down-to-up":
                pass
            elif kind == "left-to-right":
                pass
            elif kind == "right-to-left":
                pass
            elif kind == "up-to-down-3":
                pass
            elif kind == "down-to-up-3":
                pass
            elif kind == "left-to-right-3":
                if l2 == -1:
                    self.auxiliaries[kind, l1, -1] = Tensor(1)
                elif l2 < self.L2:
                    self.auxiliaries[kind, l1, l2] = self._get_auxiliaries[kind, l1, l2 - 1]\
                        .contract(self._get_auxiliaries("up-to-down", l1 - 1, l2, cut), {("Right1", "Left")}).edge_rename({"Right": "Right1"})\
                        .contract(self.lattice[l1, l2].contract(raw_base[self.spin[l1][l2]], {("Phy", "Phy")}), {("Right2", "Left"), ("Down", "Up")}).edge_rename({"Right": "Right2"})\
                        .contract(self._get_auxiliaries("down-to-up", l1+1, l2, cut), {("Right3", "Left"), ("Down", "Up")}).edge_rename({"Right": "Right3"})
                else:
                    raise Exception("Wrong Auxiliaries Position In Three Line Type")
            elif kind == "right-to-left-3":
                pass
            else:
                raise Exception("Wrong Auxiliaries Kind")
        return self.auxiliaries[kind, l1, l2]

    def refresh_auxiliaries(self, l1, l2):
        pass
        # TODO

    def __init__(self, L1: int = 4, L2: int = 4, D: int = 4):
        self.L1 = L1
        self.L2 = L2
        self.D = D

        self.lattice = [[self._create_tensor(l1, l2) for l2 in range(self.L2)] for l1 in range(self.L1)]

        self.spin = [[0 for l2 in range(self.L2)] for l1 in range(self.L1)]
        self._initialize_spin()

        self.auxiliaries = {}

    def save(self, file_name: str):
        with open(file_name, "wb") as file:
            pickle.dump(self, file)

    def gradient_descent(self, *, time, step_size, markov_chain_length, Dc):
        for _ in range(time):
            energy, gradient, real_markov_chain = self.markov_chain(markov_chain_length, Dc)
            for l1 in range(self.L1):
                for l2 in range(self.L2):
                    self.lattice[l1][l2] -= self.step_size * gradient[l1][l2] * parameter

    def _markov_chain(self, markov_chain_length, Dc):
        real_markov_chain_length = 0
        summation_of_spin_energy = 0
        summation_of_spin_gradient = [[j.same_shape().zero() for j in i] for i in self.lattice]
        summation_of_product = [[j.same_shape().zero() for j in i] for i in self.lattice]

        while real_markov_chain_length < markov_chain_length:
            # TODO
            pass

    def _single_markov_chain(self, Dc):
        #self.aux_system
        #def get_aux_up():
        #def get_aux_down():
        pass


def load(file_name: str):
    with open(file_name, "rb") as file:
        return pickle.load(file)


if __name__ == '__main__':
    fire.Fire({"new": TwoDimensionHeisenberg, "load": load})
