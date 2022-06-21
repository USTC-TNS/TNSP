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
from ..common_toolkit import MPI, mpi_comm


class ConvolutionalNeural(AbstractAnsatz, torch.nn.Module):

    __slots__ = ["multiple_product_state", "network", "dtype"]

    def __init__(self, multiple_product_state, network):
        """
        Create convolution neural ansatz for a given multiple product state.

        Parameters
        ----------
        multiple_product_state : MultipleProductState
            The multiple product state used to create open string.
        network : torch.nn.Module
            The pytorch nerual network model object.
        """
        super().__init__()
        self.multiple_product_state = multiple_product_state
        self.network = network
        self.dtype = next(self.network.parameters()).dtype

    def forward(self, x):
        """
        Get the weight of configuration x

        Parameters
        ----------
        x : list[list[x]]
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
        configuration : dict[tuple[int, int, int], int]
            The configuration dict.

        Returns
        -------
        list[list[x]]
            The configuration list as input of network.
        """
        return torch.tensor(
            [[[-1 if configuration[l1, l2, 0] == 0 else 1
               for l2 in range(self.multiple_product_state.L2)]
              for l1 in range(self.multiple_product_state.L1)]],
            dtype=self.dtype)

    def weight(self, configuration):
        x = self.create_x(configuration)
        weight = self(x)[0]
        return float(weight)

    def delta(self, configuration):
        x = self.create_x(configuration)
        weight = self(x)[0]
        self.zero_grad()
        weight.backward()
        return np.array([np.array(i.grad) for i in self.parameters()], dtype=object)

    @staticmethod
    def allreduce_delta(delta):
        requests = []
        for tensor in delta:
            requests.append(mpi_comm.Iallreduce(MPI.IN_PLACE, tensor))
        MPI.Request.Waitall(requests)

    def apply_gradient(self, gradient, step_size, relative):
        with torch.no_grad():
            if relative:
                norm = (max(float(torch.linalg.norm(i.reshape([-1]), np.inf)) for i in self.parameters()) /
                        max(np.linalg.norm(i.reshape([-1]), np.inf) for i in gradient))
                gradient = gradient * norm
            for state, grad in zip(self.parameters(), gradient):
                state.data -= step_size * grad
