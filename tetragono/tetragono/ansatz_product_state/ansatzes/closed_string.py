#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2022 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
from .abstract_ansatz import AbstractAnsatz


class ClosedString(AbstractAnsatz):

    __slots__ = ["owner", "length", "index_to_site", "cut_dimension", "tensor_list"]

    def _construct_hat(self, sites, number):
        names = ["C", *(f"P{i}" for i, _ in enumerate(sites))]
        edges = [number, *(self.owner.physics_edges[site].conjugated() for site in sites)]
        return self.owner.Tensor(names, edges).zero()

    def _construct_tensor(self, index):
        """
        Construct tensor at specified index, with all element initialized by random number.

        Parameters
        ----------
        index : int
            The index of the tensor.

        Returns
        -------
        Tensor
            The result tensor.
        """
        names = [f"P{i}" for i, _ in enumerate(self.index_to_site[index])]
        edges = [self.owner.physics_edges[site] for site in self.index_to_site[index]]
        names.append("L")
        edges.append(self.cut_dimension)
        names.append("R")
        edges.append(self.cut_dimension)
        return self.owner.Tensor(names, edges).randn()

    def __init__(self, owner, index_to_site, cut_dimension):
        """
        Create closed string ansatz by given index_to_site map and cut_dimension.

        Parameters
        ----------
        owner : AnsatzProductState
            The ansatz product state used to create closed string.
        index_to_site : list[list[tuple[int, int, int]]]
            The sites array to specify the string shape.
        cut_dimension : int
            The dimension cut of the string.
        """
        super().__init__(owner)
        self.length = len(index_to_site)
        self.index_to_site = [site for site in index_to_site]
        self.cut_dimension = cut_dimension

        self.tensor_list = np.array([self._construct_tensor(index) for index in range(self.length)])

    def weight_and_delta(self, configurations, calculate_delta):
        number = len(configurations)
        hat_list = []
        fat_list = []
        for index in range(self.length):
            sites = self.index_to_site[index]
            orbits = [orbit for l1, l2, orbit in sites]
            edges = [f"P{orbit}" for orbit in orbits]
            hat = self._construct_hat(sites, number)
            storage = hat.blocks[hat.names]
            for c_index, configuration in enumerate(configurations):
                location = tuple((c_index, *(configuration[site][1] for site in sites)))
                storage[location] = 1
            hat_list.append(hat)
            fat_list.append(hat_list[index].contract(self.tensor_list[index], {(edge, edge) for edge in edges}))

        left = [None for index in range(self.length)]
        for index in range(self.length):
            if index == 0:
                left[index] = fat_list[index]
            else:
                left[index] = left[index - 1].contract(fat_list[index], {("R", "L")}, {"C"})

        last_left = left[self.length - 1].trace({("L", "R")})
        weights = [last_left[{"C": c_index}] for c_index in range(number)]
        if not calculate_delta:
            return weights, None

        right = [None for index in range(self.length)]
        for index in reversed(range(self.length)):
            if index == self.length - 1:
                right[index] = fat_list[index]
            else:
                right[index] = right[index + 1].contract(fat_list[index], {("L", "R")}, {"C"})

        holes = [None for index in range(self.length)]
        for index in range(self.length):
            if index == 0:
                holes[index] = right[index + 1].contract(hat_list[index], set(), {"C"}).edge_rename({
                    "L": "R",
                    "R": "L"
                })
            elif index == self.length - 1:
                holes[index] = left[index - 1].contract(hat_list[index], set(), {"C"}).edge_rename({"L": "R", "R": "L"})
            else:
                holes[index] = (
                    left[index - 1]  #
                    .contract(right[index + 1], {("L", "R")}, {"C"})  #
                    .contract(hat_list[index], set(), {"C"})  #
                    .edge_rename({
                        "L": "R",
                        "R": "L"
                    }))

        deltas = [
            np.array([holes[index].shrink({"C": c_index}) for index in range(self.length)]) for c_index in range(number)
        ]
        return weights, deltas

    def refresh_auxiliaries(self):
        pass

    def ansatz_prod_sum(self, a, b):
        result = 0.0
        for ai, bi in zip(self.tensors(a), self.tensors(b)):
            dot = ai.contract(bi, {(name, name) for name in ai.names})[{}]
            result += dot
        return result

    def ansatz_conjugate(self, a):
        return np.array([i.conjugate(default_is_physics_edge=True) for i in self.tensors(a)])

    def tensors(self, delta):
        if delta is None:
            delta = self.tensor_list
        for i, [_, value] in enumerate(zip(self.tensor_list, delta)):
            recv = yield value
            if self.fixed:
                recv = None
            if recv is not None:
                # When not setting value, input delta could be an iterator
                delta[i] = recv

    def elements(self, delta):
        for index, tensor in enumerate(self.tensors(delta)):
            storage = tensor.transpose(self.tensor_list[index].names).storage
            length = len(storage)
            for i in range(length):
                recv = yield storage[i]
                if self.fixed:
                    recv = None
                if recv is not None:
                    if tensor.names != self.tensor_list[index].names:
                        raise RuntimeError("Trying to set tensor element which mismatches the edge names.")
                    storage[i] = recv

    def tensor_count(self, delta):
        for _ in self.tensors(delta):
            pass
        return self.length

    def element_count(self, delta):
        return sum(tensor.norm_num() for tensor in self.tensors(delta))

    def buffers(self, delta):
        for index, tensor in enumerate(self.tensors(delta)):
            yield tensor.storage

    def normalize_ansatz(self, log_ws=None):
        if log_ws is None:
            return self.length
        param = np.exp(log_ws / self.length)
        for tensor in self.tensor_list:
            tensor /= param

    def show(self):
        result = self.__class__.__name__ + f" with length={self.length} dimension={self.cut_dimension}"
        return result
