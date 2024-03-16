#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2020-2024 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
from ..utility import mpi_size, mpi_rank, seed_differ
from .state import SamplingNeuralState, Configuration, index_tensor_element


def _possible_hopping(pool, configuration):
    if configuration in pool:
        result = list(pool[configuration])
    else:
        result = [configuration]
    if len(result) != 1 and configuration in result:
        result.remove(configuration)
    return result


class SweepSampling:
    """
    Sweep sampling on sampling neural state.
    """

    __slots__ = ["owner", "alpha", "hopping_hamiltonians", "iterator"]

    def __init__(self, owner: SamplingNeuralState, configurations, alpha=1.0, hopping_hamiltonians=None):
        self.owner = owner
        self.alpha = alpha
        if hopping_hamiltonians is None:
            self.hopping_hamiltonians = self.owner._hamiltonians
        else:
            self.hopping_hamiltonians = hopping_hamiltonians

        self.iterator = self.sweep_sampling(configurations)

    def __call__(self):
        return next(self.iterator)

    def sweep_sampling(self, configurations):
        """
        Perform sweep sampling on state.

        Yields
        ------
        configurations, amplitudes, weights
            The weights are always |amplitudes|^{2 alpha}
        """
        amplitudes = self.owner(configurations, enable_grad=False)
        weights = amplitudes.abs()**(2 * self.alpha)
        while True:
            for _ in range(self.owner.site_number):
                # Try hopping
                configurations_cpu = configurations.cpu()
                configurations_cpu_s = configurations_cpu.clone()
                hopping_number_fixing = []
                hopping_pool = list(self.hopping_hamiltonians.items())
                for configuration, configuration_s in zip(configurations_cpu, configurations_cpu_s):
                    positions, hamiltonian = hopping_pool[TAT.random.uniform_int(0, len(hopping_pool) - 1)()]
                    element_pool = index_tensor_element(hamiltonian)
                    positions_configuration = tuple(configuration[l1l2o].item() for l1l2o in positions)
                    possible_hopping = _possible_hopping(element_pool, positions_configuration)
                    hopping_number = len(possible_hopping)
                    positions_configuration_s = possible_hopping[TAT.random.uniform_int(0, hopping_number - 1)()]
                    hopping_number_s = len(_possible_hopping(element_pool, positions_configuration_s))
                    hopping_number_fixing.append(hopping_number / hopping_number_s)
                    for l1l2o, value in zip(positions, positions_configuration_s):
                        configuration_s[l1l2o] = value
                configurations_s = configurations_cpu_s.to(device=configurations.device)
                amplitudes_s = self.owner(configurations_s, enable_grad=False)
                weights_s = amplitudes_s.abs()**(2 * self.alpha)
                p = (weights_s / weights) * torch.tensor(hopping_number_fixing, device=self.owner.device)
                go = torch.rand_like(p) < p
                configurations = torch.where(go.reshape([-1, 1, 1, 1]), configurations_s, configurations)
                amplitudes = torch.where(go, amplitudes_s, amplitudes)
                weights = torch.where(go, weights_s, weights)

            yield configurations, amplitudes, weights, torch.ones_like(weights, dtype=torch.int64)


class DirectSampling:
    """
    Direct sampling.
    """

    __slots__ = ["owner", "alpha", "batch_size"]

    def __init__(self, owner, alpha, batch_size):
        self.owner = owner
        self.alpha = alpha
        self.batch_size = batch_size

    def __call__(self):
        configurations, amplitudes, weights, multiplicies = self.owner.network.generate(
            self.batch_size,
            self.alpha,
        )
        return configurations, amplitudes, weights, multiplicies


class ErgodicSampling:
    """
    Ergodic sampling.
    """

    __slots__ = ["owner", "total_step", "iterator", "batch_size"]

    def __init__(self, owner, batch_size):
        self.owner = owner
        self.batch_size = batch_size

        # Calculate total step count for outside usage.
        self.total_step = 1
        for [l1, l2, orbit], edge in self.owner.physics_edges:
            self.total_step *= edge.dimension

        self.iterator = self.ergodic_sampling()

    def _zero_configuration(self):
        configuration = Configuration(self.owner).export_configuration()
        for [l1, l2, orbit], edge in self.owner.physics_edges:
            configuration[l1, l2, orbit] = 0
        return configuration

    def _next_configuration(self, configuration):
        for [l1, l2, orbit], edge in self.owner.physics_edges:
            configuration[l1, l2, orbit] += 1
            if configuration[l1, l2, orbit] == edge.dimension:
                configuration[l1, l2, orbit] = 0
            else:
                return configuration

    def __call__(self):
        return next(self.iterator)

    def ergodic_sampling(self):
        """
        Perform ergodic sampling on state.

        Yields
        ------
        configurations, amplitudes, weights
            The weights are always 1
        """
        # The current configuration.
        index = 0
        configuration = self._zero_configuration()
        configuration_pool = []
        while True:
            if index % mpi_size == mpi_rank:
                configuration_pool.append(configuration.copy())
            index += 1
            if len(configuration_pool) == self.batch_size or index == self.total_step:
                configurations = torch.tensor(np.array(configuration_pool), device=self.owner.device)
                amplitudes = self.owner(configurations, enable_grad=False)
                yield configurations, amplitudes, torch.ones_like(amplitudes.real), torch.ones_like(amplitudes,
                                                                                                    dtype=torch.int64)
                configuration_pool = []
                if index == self.total_step:
                    return

            configuration = self._next_configuration(configuration)
