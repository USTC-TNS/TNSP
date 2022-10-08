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
import torch
from ...common_toolkit import seed_differ
from .abstract_ansatz import AbstractAnsatz


class OpenString(AbstractAnsatz):

    __slots__ = ["length", "index_to_site", "cut_dimension", "tensor_list"]

    def numpy_array(self, array):
        # Create an empty np array to avoid numpy FutureWarning
        length = len(array)
        result = np.empty(length, dtype=object)
        for i in range(length):
            result[i] = array[i]
        return result

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
        # order: P, L, R
        return torch.randn([
            np.prod([self.owner.physics_edges[site].dimension for site in self.index_to_site[index]]),
            self.cut_dimension if index != 0 else 1, self.cut_dimension if index != self.length - 1 else 1
        ],
                           dtype=torch.float64)

    def __init__(self, owner, index_to_site, cut_dimension):
        """
        Create open string ansatz by given index_to_site map and cut_dimension.

        Parameters
        ----------
        owner : AnsatzProductState
            The ansatz product state used to create open string.
        index_to_site : list[list[tuple[int, int, int]]]
            The sites array to specify the string shape.
        cut_dimension : int
            The dimension cut of the string.
        """
        super().__init__(owner)
        self.length = len(index_to_site)
        self.index_to_site = [site for site in index_to_site]
        self.cut_dimension = cut_dimension

        torch.manual_seed(seed_differ.random_int())
        self.tensor_list = self.numpy_array([self._construct_tensor(index) for index in range(self.length)])

    def _get_hat_fat(self, configurations):
        hat_list = []
        fat_list = []
        configs = [config._configuration for config in configurations]
        for index in range(self.length):
            sites = self.index_to_site[index]
            # order: C, P
            hat = torch.tensor(configs[0].get_hat(configs, sites,
                                                  [self.owner.physics_edges[site].dimension for site in sites]),
                               dtype=torch.float64)
            hat_list.append(hat)
            fat_list.append(torch.einsum("cp,plr->clr", hat_list[index], self.tensor_list[index]))
        return hat_list, fat_list

    def _get_left(self, fat_list):
        left = [None for index in range(self.length)]
        for index in range(self.length):
            if index == 0:
                left[index] = fat_list[index]
            else:
                left[index] = torch.einsum("clm,cmr->clr", left[index - 1], fat_list[index])
        return left

    def _get_right(self, fat_list):
        right = [None for index in range(self.length)]
        for index in reversed(range(self.length)):
            if index == self.length - 1:
                right[index] = fat_list[index]
            else:
                right[index] = torch.einsum("cmr,clm->clr", right[index + 1], fat_list[index])
        return right

    def weight_and_delta(self, configurations, calculate_delta):
        number = len(configurations)

        hat_list, fat_list = self._get_hat_fat(configurations)

        left = self._get_left(fat_list)

        last_left = torch.einsum("cmm->c", left[self.length - 1]).cpu()
        weights = last_left.tolist()
        if not calculate_delta:
            return weights, None
        elif self.fixed:
            this_delta = self.numpy_array([torch.zeros_like(i) for i in self.tensor_list])
            return weights, [this_delta for _ in configurations]

        right = self._get_right(fat_list)

        holes = [None for index in range(self.length)]
        for index in range(self.length):
            if index == 0:
                holes[index] = torch.einsum("clr,cp->cprl", right[index + 1], hat_list[index])
            elif index == self.length - 1:
                holes[index] = torch.einsum("clr,cp->cprl", left[index - 1], hat_list[index])
            else:
                holes[index] = torch.einsum("cmr,clm,cp->cprl", left[index - 1], right[index + 1], hat_list[index])

        deltas = [
            self.numpy_array([holes[index][c_index].contiguous()
                              for index in range(self.length)])
            for c_index in range(number)
        ]
        return weights, deltas

    def refresh_auxiliaries(self):
        pass

    def ansatz_prod_sum(self, a, b):
        result = 0.0
        for ai, bi in zip(self.tensors(a), self.tensors(b)):
            result = result + torch.dot(ai.reshape([-1]), bi.reshape([-1]))
        return result.cpu().item()

    def ansatz_conjugate(self, a):
        a = [i.conj() for i in self.tensors(a)]
        return self.numpy_array(a)

    def tensors(self, delta):
        if delta is None:
            for i, tensor in enumerate(self.tensor_list):
                recv = yield tensor
                if self.fixed:
                    recv = None
                if recv is not None:
                    self.tensor_list[i] = recv.detach().clone()
        else:
            for i, [_, value] in enumerate(zip(self.tensor_list, delta)):
                recv = yield value
                if self.fixed:
                    recv = None
                if recv is not None:
                    delta[i] = recv.clone()

    def elements(self, delta):
        for tensor in self.tensors(delta):
            # Should be tensor.view for pytorch
            # But there is no equivalent function for numpy array.
            # So use reshape here.
            flatten = tensor.view([-1])
            length = len(flatten)
            for i in range(length):
                recv = yield flatten[i]
                if self.fixed:
                    recv = None
                if recv is not None:
                    flatten[i] = recv

    def tensor_count(self, delta):
        return len([None for _ in self.tensors(delta)])

    def element_count(self, delta):
        # Should use tensor.view here for pytorch
        return sum(len(tensor.reshape([-1])) for tensor in self.tensors(delta))

    def buffers(self, delta):
        yield from self.tensors(delta)

    def normalize_ansatz(self, log_ws=None):
        if log_ws is None:
            return self.length
        param = np.exp(log_ws / self.length)
        for tensor in self.tensor_list:
            tensor /= param

    def show(self):
        result = self.__class__.__name__ + f" with length={self.length} dimension={self.cut_dimension}"
        return result
