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
import pickle
import TAT
import fire


class OneDimensionHeisenberg():

    def __init__(self, length: int = 4, dimension: int = 4):
        self.length = length
        self.dimension = dimension
        self.lattice = [TAT.Tensor.DNo(["Phy", "Right"], [2, self.dimension]).set(lambda: np.random.rand()) if i == 0 else\
                        TAT.Tensor.DNo(["Phy", "Left"], [2, self.dimension]).set(lambda: np.random.rand()) if i == self.length - 1 else\
                        TAT.Tensor.DNo(["Phy", "Left", "Right"], [2, self.dimension, self.dimension]).set(lambda: np.random.rand()) for i in range(self.length)]
        self.hamiltonian = TAT.Tensor.DNo("I0 I1 O0 O1".split(" "), [2, 2, 2, 2])
        self.hamiltonian.block()[:] = np.array([1 / 4., 0, 0, 0, 0, -1 / 4., 2 / 4., 0, 0, 2 / 4., -1 / 4., 0, 0, 0, 0, 1 / 4.]).reshape([2, 2, 2, 2])
        self.identity = TAT.Tensor.DNo("I0 I1 O0 O1".split(" "), [2, 2, 2, 2])
        self.identity.block()[:] = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]).reshape([2, 2, 2, 2])

    def drop(self):
        pass

    def update(self, time: int, delta_t: float, *, print_step: int = 0, show_tensor: bool = False, first_step_print: bool = False):
        updater = self.identity - delta_t * self.hamiltonian
        if first_step_print:
            print("step = 0\nEnergy = ", self.energy())
            if show_tensor:
                self._show()
        for i in range(time):
            self._update_once(updater)
            if print_step > 0 and i % print_step == print_step - 1 or i == time - 1:
                print("step = ", i + 1, "\nEnergy = ", self.energy())
                if show_tensor:
                    self._show()
        return self

    def energy(self) -> float:
        left_pool = {}
        right_pool = {}

        def get_left(i: int):
            if i == -1:
                return TAT.Tensor.DNo(1)
            if i not in left_pool:
                left_pool[i] = get_left(i - 1)\
                    .contract(self.lattice[i], {("Right1", "Left")}).edge_rename({"Right": "Right1"})\
                    .contract(self.lattice[i], {("Right2", "Left"), ("Phy", "Phy")}).edge_rename({"Right": "Right2"})
            return left_pool[i]

        def get_right(i: int):
            if i == self.length:
                return TAT.Tensor.DNo(1)
            if i not in right_pool:
                right_pool[i] = get_right(i + 1)\
                    .contract(self.lattice[i], {("Left1", "Right")}).edge_rename({"Left": "Left1"})\
                    .contract(self.lattice[i], {("Left2", "Right"), ("Phy", "Phy")}).edge_rename({"Left": "Left2"})
            return right_pool[i]

        energy = 0
        for i in range(self.length - 1):
            energy += get_left(i-1)\
                .contract(self.lattice[i], {("Right1", "Left")}).edge_rename({"Right": "Right1", "Phy": "PhyA"})\
                .contract(self.lattice[i+1], {("Right1", "Left")}).edge_rename({"Right": "Right1", "Phy": "PhyB"})\
                .contract(self.hamiltonian, {("PhyA", "I0"), ("PhyB", "I1")})\
                .contract(self.lattice[i], {("Right2", "Left"), ("O0", "Phy")}).edge_rename({"Right": "Right2"})\
                .contract(self.lattice[i+1], {("Right2", "Left"), ("O1", "Phy")}).edge_rename({"Right": "Right2"})\
                .contract(get_right(i+2), {("Right1", "Left1"), ("Right2", "Left2")})
        energy /= get_right(0)
        return energy.value() / self.length

    def _show(self):
        for i in self.lattice:
            print(i)

    def _update_once(self, updater):
        for i in range(self.length - 1):
            AB = self.lattice[i].edge_rename({"Phy": "PhyA"})\
                .contract(self.lattice[i+1].edge_rename({"Phy": "PhyB"}), {("Right", "Left")})
            ABH = AB.contract(updater, {("PhyA", "I0"), ("PhyB", "I1")})
            [u, s, v] = ABH.svd({"Left", "O0"}, "Right", "Left", self.dimension)
            self.lattice[i] = u.edge_rename({"O0": "Phy"})
            self.lattice[i + 1] = v.multiple(s, "Left", 'v').edge_rename({"O1": "Phy"})
            self.lattice[i] = self.lattice[i] / self.lattice[i].norm_max()
            self.lattice[i + 1] = self.lattice[i + 1] / self.lattice[i + 1].norm_max()
        for i in range(self.length - 2, -1, -1):
            AB = self.lattice[i + 1].edge_rename({"Phy": "PhyA"})\
                .contract(self.lattice[i].edge_rename({"Phy": "PhyB"}), {("Left", "Right")})
            ABH = AB.contract(updater, {("PhyA", "I0"), ("PhyB", "I1")})
            [u, s, v] = ABH.svd({"Right", "O0"}, "Left", "Right", self.dimension)
            self.lattice[i + 1] = u.edge_rename({"O0": "Phy"})
            self.lattice[i] = v.multiple(s, "Right", 'v').edge_rename({"O1": "Phy"})
            self.lattice[i + 1] = self.lattice[i + 1] / self.lattice[i + 1].norm_max()
            self.lattice[i] = self.lattice[i] / self.lattice[i].norm_max()

    def save(self, file_name: str):
        with open(file_name, "wb") as file:
            pickle.dump(self, file)

    def __str__(self):
        return f"[One Dimension Heisenberg System with L={self.length} and D={self.dimension}]"


def load(file_name: str):
    with open(file_name, "rb") as file:
        return pickle.load(file)


if __name__ == '__main__':
    fire.Fire({"new": OneDimensionHeisenberg, "load": load})
