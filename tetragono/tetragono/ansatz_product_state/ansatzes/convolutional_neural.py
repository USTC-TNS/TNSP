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
from ..state import AnsatzProductState


class ConvolutionalNeural(AbstractAnsatz):

    __slots__ = ["owner", "network", "dtype"]

    def __init__(self, owner, network):
        """
        Create convolution neural ansatz for a given ansatz product state.

        The state should have only single orbit for every site.

        Parameters
        ----------
        owner : AnsatzProductState
            The ansatz product state used to create open string.
        network : torch.nn.Module
            The pytorch nerual network model object.
        """
        self.owner: AnsatzProductState = owner
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
        return [[[-1 if configuration[l1, l2, 0][1] == 0 else 1
                  for l2 in range(self.owner.L2)]
                 for l1 in range(self.owner.L1)]]

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

    def refresh_auxiliaries(self):
        pass

    def ansatz_prod_sum(self, a, b):
        result = 0.0
        for ai, bi in zip(self.buffers(a), self.buffers(b)):
            result += float(np.dot(ai.reshape([-1]), bi.reshape([-1])))
        return result

    def ansatz_conjugate(self, a):
        # CNN network is always real without imaginary part
        a = [i for i in self.buffers(a)]
        length = len(a)
        # Create an empty np array to avoid numpy FutureWarning
        result = np.empty(length, dtype=object)
        for i in range(length):
            result[i] = a[i]
        return result

    def buffers(self, delta):
        if delta is None:
            for tensor in self.network.parameters():
                recv = yield tensor.data
                if recv is not None:
                    tensor.data = torch.tensor(np.array(recv).real.copy())
                    # convert to numpy and get real part then convert it back to torch tensor
                    # because of https://github.com/pytorch/pytorch/issues/82610
        else:
            for i, [_, value] in enumerate(zip(self.network.parameters(), delta)):
                recv = yield value
                if recv is not None:
                    # When not setting value, input delta could be an iterator
                    delta[i] = recv.real.copy()
                    # copy it to ensure the stride is contiguous

    def elements(self, delta):
        for tensor in self.buffers(delta):
            # Should be tensor.view for pytorch
            # But there is no equivalent function for numpy array.
            # So use reshape here.
            flatten = tensor.reshape([-1])
            length = len(flatten)
            for i in range(length):
                recv = yield flatten[i]
                if recv is not None:
                    flatten[i] = recv.real

    def buffer_count(self, delta):
        delta = self.network.parameters()
        delta = list(delta)
        length = len(delta)
        return length

    def element_count(self, delta):
        # Should use tensor.view here for pytorch
        return sum(len(tensor.reshape([-1])) for tensor in self.buffers(delta))

    def buffers_for_mpi(self, delta):
        yield from self.buffers(delta)

    def recovery_real(self, delta=None):
        if delta is None:
            return True
        return np.array([i.real for i in delta], dtype=object)
