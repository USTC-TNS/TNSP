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
from .abstract_ansatz import AbstractAnsatz


class ConvolutionalNeural(AbstractAnsatz):

    __slots__ = ["network", "dtype"]

    def numpy_array(self, array):
        # Create an empty np array to avoid numpy FutureWarning
        length = len(array)
        result = np.empty(length, dtype=object)
        for i in range(length):
            result[i] = array[i]
        return result

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
        super().__init__(owner)
        self.network = network
        self.dtype = next(self.network.parameters()).dtype

    def get_weights(self, xs):
        """
        Get the weight of configuration x.

        Parameters
        ----------
        xs : list[list[list[list[int]]]]
            The configuration matrix.

        Returns
        -------
        float
            The weight of the given configuration
        """
        result = self.network(xs.to(device=next(self.network.parameters()).device))
        s = torch.prod(result.reshape([result.shape[0], -1]), -1)
        return s

    def weight_and_delta(self, configurations, calculate_delta):
        configs = [config._configuration for config in configurations]
        xs = torch.tensor(configs[0].export_orbit0(configs), dtype=self.dtype) * 2 - 1
        weight = self.get_weights(xs)
        if calculate_delta:
            number = len(configurations)
            delta = []
            for i in range(number):
                if self.fixed:
                    this_delta = self.numpy_array([torch.zeros_like(i) for i in self.network.parameters()])
                    delta.append(this_delta)
                else:
                    self.network.zero_grad()
                    weight[i].backward(retain_graph=True)
                    this_delta = self.numpy_array([i.grad.detach().clone() for i in self.network.parameters()])
                    delta.append(this_delta)
        else:
            delta = None
        return np.array(weight.detach().clone().cpu(), copy=False), delta

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
            for tensor in self.network.parameters():
                recv = yield tensor.data
                if self.fixed:
                    recv = None
                if recv is not None:
                    tensor.data = recv.real.detach().clone()
        else:
            for i, [_, value] in enumerate(zip(self.network.parameters(), delta)):
                recv = yield value
                if self.fixed:
                    recv = None
                if recv is not None:
                    # When not setting value, input delta could be an iterator
                    delta[i] = recv.real.clone()
                    # copy it to ensure the stride is contiguous

    def elements(self, delta):
        for tensor in self.tensors(delta):
            # Should be tensor.view for pytorch
            # But there is no equivalent function for numpy array.
            # So use reshape here.
            flatten = tensor.reshape([-1])
            length = len(flatten)
            for i in range(length):
                recv = yield flatten[i]
                if self.fixed:
                    recv = None
                if recv is not None:
                    flatten[i] = recv.real

    def tensor_count(self, delta):
        return len([None for _ in self.tensors(delta)])

    def element_count(self, delta):
        # Should use tensor.view here for pytorch
        return sum(len(tensor.reshape([-1])) for tensor in self.tensors(delta))

    def buffers(self, delta):
        yield from self.tensors(delta)

    def recovery_real(self, delta=None):
        return self.numpy_array([i.real for i in delta])

    def show(self):
        result = self.__class__.__name__
        return result
