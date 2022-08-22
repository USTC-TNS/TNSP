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
from ..state import AnsatzProductState
from .abstract_ansatz import AbstractAnsatz


class ClosedString(AbstractAnsatz):

    __slots__ = ["owner", "length", "index_to_site", "cut_dimension", "tensor_list", "_left_to_right", "_right_to_left"]

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
        self.owner: AnsatzProductState = owner
        self.length = len(index_to_site)
        self.index_to_site = [site for site in index_to_site]
        self.cut_dimension = cut_dimension

        self.tensor_list = np.array([self._construct_tensor(index) for index in range(self.length)])

        self.refresh_auxiliaries()

    def _go_from_left(self, configuration, *, try_only):
        """
        Go along the auxiliaries chain from left to right.

        Parameters
        ----------
        configuration : list[list[int]]
            The configuration to calculate.
        try_only : bool
            Whether to avoid to calculate tensor, go as far as possible only.

        Returns
        -------
        int, Tensor
            The index calculated, if try_only=False, it equals to length of configuration, and the result tensor.
        """
        result = self._left_to_right
        left = None
        index = 0
        while index < len(configuration):
            config = configuration[index]
            if config not in result:
                if try_only:
                    break
                else:
                    next_left = self.tensor_list[index].shrink({f"P{orbit}": conf for orbit, conf in enumerate(config)})
                    if left is not None:
                        next_left = left.contract(next_left, {("R", "L")})
                    result[config] = {}, next_left
            result, left = result[config]
            index += 1
        return index, left

    def _go_from_right(self, configuration, *, try_only):
        """
        Go along the auxiliaries chain from right to left.

        Parameters
        ----------
        configuration : list[list[int]]
            The configuration to calculate.
        try_only : bool
            Whether to avoid to calculate tensor, go as far as possible only.

        Returns
        -------
        int, Tensor
            The index calculated, if try_only=False, it equals to length of configuration, and the result tensor.
        """
        result = self._right_to_left
        right = None
        index = 0
        while index < len(configuration):
            config = configuration[index]
            if config not in result:
                if try_only:
                    break
                else:
                    next_right = self.tensor_list[self.length - 1 - index].shrink(
                        {f"P{orbit}": conf for orbit, conf in enumerate(config)})
                    if right is not None:
                        next_right = right.contract(next_right, {("L", "R")})
                    result[config] = {}, next_right
            result, right = result[config]
            index += 1
        return index, right

    def _get_index_configuration(self, site_configuration):
        return [tuple(site_configuration[l1, l2, orbit][1] for l1, l2, orbit in sites) for sites in self.index_to_site]

    def _weight(self, site_configuration):
        index_configuration = self._get_index_configuration(site_configuration)
        index, left = self._go_from_left(index_configuration, try_only=True)
        if index == 0:
            index = None
        else:
            index -= 1
        _, right = self._go_from_right(index_configuration[:index:-1], try_only=False)
        if left is None:
            return right.trace({("L", "R")})[{}]
        elif right is None:
            return left.trace({("L", "R")})[{}]
        else:
            return left.contract(right, {("R", "L"), ("L", "R")})[{}]

    def _delta(self, site_configuration):
        index_configuration = self._get_index_configuration(site_configuration)
        result = []
        _, _ = self._go_from_left(index_configuration[::1], try_only=False)
        _, _ = self._go_from_right(index_configuration[::-1], try_only=False)
        for index in range(self.length):
            _, left = self._go_from_left(index_configuration[0:index:1], try_only=False)
            _, right = self._go_from_right(index_configuration[-1:index:-1], try_only=False)
            if left is None:
                this_tensor = right
            elif right is None:
                this_tensor = left
            else:
                this_tensor = left.contract(right, {("L", "R")})
            this_tensor = this_tensor.edge_rename({"R": "L", "L": "R"})
            this_tensor = this_tensor.expand({
                f"P{orbit}": (conf, self.owner.physics_edges[self.index_to_site[index][orbit]].dimension)
                for orbit, conf in enumerate(index_configuration[index])
            })
            result.append(this_tensor)
        return np.array(result)

    def refresh_auxiliaries(self):
        self._left_to_right = {}
        self._right_to_left = {}

    def ansatz_prod_sum(self, a, b):
        result = 0.0
        for ai, bi in zip(self.buffers(a), self.buffers(b)):
            dot = ai.contract(bi, {(name, name) for name in ai.names})[{}]
            result += dot
        return result

    def ansatz_conjugate(self, a):
        return np.array([i.conjugate(default_is_physics_edge=True) for i in self.buffers(a)])

    def buffers(self, delta):
        if delta is None:
            delta = self.tensor_list
        for i, [_, value] in enumerate(zip(self.tensor_list, delta)):
            recv = yield value
            if recv is not None:
                # When not setting value, input delta could be an iterator
                delta[i] = recv

    def elements(self, delta):
        for index, tensor in enumerate(self.buffers(delta)):
            storage = tensor.transpose(self.tensor_list[index].names).storage
            length = len(storage)
            for i in range(length):
                recv = yield storage[i]
                if recv is not None:
                    if tensor.names != self.tensor_list[index].names:
                        raise RuntimeError("Trying to set tensor element which mismatches the edge names.")
                    storage[i] = recv

    def buffer_count(self, delta):
        return self.length

    def element_count(self, delta):
        return sum(tensor.norm_num() for tensor in self.buffers(delta))

    def buffers_for_mpi(self, delta):
        for index, tensor in enumerate(self.buffers(delta)):
            yield tensor.storage

    def normalize_ansatz(self):
        for tensor in self.tensor_list:
            tensor /= tensor.norm_max()
