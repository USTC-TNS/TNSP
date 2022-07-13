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
import TAT
from .abstract_ansatz import AbstractAnsatz
from ...common_toolkit import MPI, mpi_comm


class ConvolutionalNeural(AbstractAnsatz):

    __slots__ = ["ansatz_product_state", "network", "dtype"]

    def __init__(self, ansatz_product_state, network):
        """
        Create convolution neural ansatz for a given ansatz product state.

        Parameters
        ----------
        ansatz_product_state : AnsatzProductState
            The ansatz product state used to create open string.
        network : torch.nn.Module
            The pytorch nerual network model object.
        """
        super().__init__()
        self.ansatz_product_state = ansatz_product_state
        self.network = network
        self.dtype = next(self.network.parameters()).dtype

    def __call__(self, x):
        """
        Get the weight of configuration x.

        Parameters
        ----------
        x : list[list[int]]
            The configuration matrix.

        Returns
        -------
        float
            The weight of the given configuration
        """
        h = self.network(x)
        s = torch.prod(h.reshape([h.shape[0], -1]), -1)
        return s

    def create_x(self, configuration):
        """
        Get the configuration x as input of network from dict configuration.

        Parameters
        ----------
        configuration : list[list[dict[int, EdgePoint]]]
            The configuration dict.

        Returns
        -------
        list[list[list[int]]]
            The configuration as input of network, where three dimensions are channel, width and height.
        """
        return [[[-1 if configuration[l1][l2][0][1] == 0 else 1
                  for l2 in range(self.ansatz_product_state.L2)]
                 for l1 in range(self.ansatz_product_state.L1)]]

    def weight(self, configuration):
        x = torch.tensor([self.create_x(configuration)], dtype=self.dtype)
        weight = self(x)[0]
        return float(weight)

    def delta(self, configuration):
        x = torch.tensor([self.create_x(configuration)], dtype=self.dtype)
        weight = self(x)[0]
        self.network.zero_grad()
        weight.backward()
        return np.array([np.array(i.grad) for i in self.network.parameters()], dtype=object)

    def weight_and_delta(self, configurations, calculate_delta):
        xs = torch.tensor([self.create_x(configuration) for configuration in configurations], dtype=self.dtype)
        weight = self(xs)
        if calculate_delta:
            number = len(configurations)
            delta = []
            for i in range(number):
                self.network.zero_grad()
                weight[i].backward()
                this_delta = np.array([np.array(i.grad) for i in self.network.parameters()], dtype=object)
                delta.append(this_delta)
        else:
            delta = None
        return weight.tolist(), delta

    def get_norm_max(self, delta):
        if delta is None:
            delta = self.network.parameters()
        with torch.no_grad():
            return max(float(torch.linalg.norm(i.reshape([-1]), np.inf)) for i in self.network.parameters())

    def refresh_auxiliaries(self):
        pass

    @staticmethod
    def delta_dot_sum(a, b):
        result = 0.0
        for ai, bi in zip(a, b):
            result += float(np.dot(ai.reshape([-1]), bi.reshape([-1])))
        return result

    @staticmethod
    def delta_update(a, b):
        for ai, bi in zip(a, b):
            ai += bi

    def buffers(self, delta):
        if delta is None:
            for tensor in self.network.parameters():
                recv = yield tensor.data
                if recv is not None:
                    tensor.data = recv
        else:
            length = len(delta)
            for i in range(length):
                recv = yield delta[i]
                if recv is not None:
                    delta[i] = recv

    def elements(self, delta):
        for tensor in self.buffers(delta):
            flatten = tensor.reshape([-1])
            length = len(flatten)
            for i in range(length):
                recv = yield flatten[i]
                if recv is not None:
                    flatten[i] = recv

    def buffer_count(self, delta):
        if delta is None:
            delta = self.parameters()
        length = len(list(delta))
        return length

    def element_count(self, delta):
        return sum(len(tensor.reshape([-1])) for tensor in self.buffers(delta))

    def buffers_for_mpi(self, delta):
        yield from self.buffers(delta)
