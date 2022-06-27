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
from ..auxiliaries.safe_toolkit import safe_rename, safe_contract
from .abstract_ansatz import AbstractAnsatz
from ..common_toolkit import MPI, mpi_comm


class OpenString(AbstractAnsatz):

    __slots__ = [
        "multiple_product_state", "length", "index_to_site", "site_to_index", "cut_dimension", "tensor_list",
        "_left_to_right", "_right_to_left"
    ]

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
        names = ["P"]
        edges = [self.multiple_product_state.physics_edges[self.index_to_site[index]]]
        if index != 0:
            names.append("L")
            edges.append(self.cut_dimension)
        if index != self.length - 1:
            names.append("R")
            edges.append(self.cut_dimension)
        return self.multiple_product_state.Tensor(names, edges).randn()

    def __init__(self, multiple_product_state, index_to_site, cut_dimension):
        """
        Create open string ansatz by given index_to_site map and cut_dimension.

        Parameters
        ----------
        multiple_product_state : MultipleProductState
            The multiple product state used to create open string.
        index_to_site : list[tuple[int, int, int]]
            The sites array to specify the string shape.
        cut_dimension : int
            The dimension cut of the string.
        """
        self.multiple_product_state = multiple_product_state
        self.length = len(index_to_site)
        self.index_to_site = [site for site in index_to_site]
        self.site_to_index = {site: index for index, site in enumerate(index_to_site)}
        self.cut_dimension = cut_dimension

        self.tensor_list = np.array([self._construct_tensor(index) for index in range(self.length)])

        self._refresh_auxiliaries()

    def _refresh_auxiliaries(self):
        """
        Refresh the auxiliaries tensors. it is needed to call it after tensor_list updated.
        """
        self._left_to_right = {}
        self._right_to_left = {}

    def _go_from_left(self, configuration, *, try_only):
        """
        Go along the auxiliaries chain from left to right.

        Parameters
        ----------
        configuration : list[int]
            The configuration to calculate.
        try_only : bool
            Whether to avoid to calculate tensor, go as far as possible only.

        Returns
        -------
        int, Tensor
            The index calculated, if try_only=False, it equals to length of configuration, and the result tensor.
        """
        result = self._left_to_right
        left = self.multiple_product_state.Tensor(1)
        index = 0
        while index < len(configuration):
            config = configuration[index]
            if config not in result:
                if try_only:
                    break
                else:
                    result[config] = {}, left.contract(self.tensor_list[index].shrink({"P": config}), {("R", "L")})
            result, left = result[config]
            index += 1
        return index, left

    def _go_from_right(self, configuration, *, try_only):
        """
        Go along the auxiliaries chain from right to left.

        Parameters
        ----------
        configuration : list[int]
            The configuration to calculate.
        try_only : bool
            Whether to avoid to calculate tensor, go as far as possible only.

        Returns
        -------
        int, Tensor
            The index calculated, if try_only=False, it equals to length of configuration, and the result tensor.
        """
        result = self._right_to_left
        right = self.multiple_product_state.Tensor(1)
        index = 0
        while index < len(configuration):
            config = configuration[index]
            if config not in result:
                if try_only:
                    break
                else:
                    result[config] = {}, right.contract(self.tensor_list[self.length - 1 - index].shrink({"P": config}),
                                                        {("L", "R")})
            result, right = result[config]
            index += 1
        return index, right

    def weight(self, site_configuration):
        index_configuration = [site_configuration[l1][l2][orbit][1] for l1, l2, orbit in self.index_to_site]
        index, left = self._go_from_left(index_configuration, try_only=True)
        if index == 0:
            index = None
        else:
            index -= 1
        _, right = self._go_from_right(index_configuration[:index:-1], try_only=False)
        return safe_contract(left, right, {("R", "L")})[{}]

    def delta(self, site_configuration):
        index_configuration = [site_configuration[l1][l2][orbit][1] for l1, l2, orbit in self.index_to_site]
        result = []
        _, _ = self._go_from_left(index_configuration[::1], try_only=False)
        _, _ = self._go_from_right(index_configuration[::-1], try_only=False)
        for index in range(self.length):
            _, left = self._go_from_left(index_configuration[0:index:1], try_only=False)
            _, right = self._go_from_right(index_configuration[-1:index:-1], try_only=False)
            this_tensor = safe_rename(left, {"R": "L"}).contract(safe_rename(right, {"L": "R"}), set())
            this_tensor = this_tensor.conjugate().expand({
                "P": (index_configuration[index],
                      self.multiple_product_state.physics_edges[self.index_to_site[index]].dimension)
            })
            result.append(this_tensor)
        return np.array(result)

    @staticmethod
    def allreduce_delta(delta):
        requests = []
        for tensor in delta:
            requests.append(mpi_comm.Iallreduce(MPI.IN_PLACE, tensor.storage))
        MPI.Request.Waitall(requests)

    def apply_gradient(self, gradient, step_size, relative):
        if relative:
            gradient = gradient * (max(i.norm_max() for i in self.tensor_list) / max(i.norm_max() for i in gradient))
        self.tensor_list -= step_size * gradient
        self._refresh_auxiliaries()
