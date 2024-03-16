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
from ..utility import mpi_size, mpi_rank, seed_differ, bcast_number, bcast_buffer
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

    __slots__ = ["owner", "hopping_hamiltonians", "alpha", "iterator"]

    def __init__(self, owner: SamplingNeuralState, configurations, total_size, hopping_hamiltonians, alpha):
        self.owner = owner
        if hopping_hamiltonians is None:
            self.hopping_hamiltonians = self.owner._hamiltonians
        else:
            self.hopping_hamiltonians = hopping_hamiltonians
        self.alpha = alpha

        total_size_rank = total_size // mpi_size + (mpi_rank < total_size % mpi_size)
        repeat = -(-total_size_rank // len(configurations))
        configurations = np.tile(configurations, [repeat, 1, 1, 1])[:total_size_rank]
        configurations = torch.tensor(configurations, device=self.owner.device)
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

    __slots__ = ["owner", "total_size", "alpha"]

    def __init__(self, owner, total_size, alpha):
        self.owner = owner
        self.total_size = total_size
        self.alpha = alpha

    def __call__(self):
        if mpi_rank == 0:
            configurations, amplitudes, weights, multiplicities = self.owner.network.generate(
                self.total_size,
                self.alpha,
            )
            unique_size = len(multiplicities)
        else:
            unique_size = 0
        unique_size = bcast_number(unique_size, dtype=int)
        if mpi_rank != 0:
            L1 = self.owner.L1
            L2 = self.owner.L2
            orbit = max(orbit for [l1, l2, orbit], edge in self.owner.physics_edges) + 1

            configurations = torch.empty([unique_size, L1, L2, orbit], dtype=torch.int64, device=self.owner.device)
            amplitudes = torch.empty([unique_size], dtype=self.owner.dtype, device=self.owner.device)
            weights = torch.ones_like(amplitudes.real)
            multiplicities = torch.ones_like(amplitudes, dtype=torch.int64)

        bcast_buffer(configurations)
        bcast_buffer(amplitudes)
        bcast_buffer(weights)
        bcast_buffer(multiplicities)

        configurations = configurations[mpi_rank::mpi_size]
        amplitudes = amplitudes[mpi_rank::mpi_size]
        weights = weights[mpi_rank::mpi_size]
        multiplicities = multiplicities[mpi_rank::mpi_size]

        return configurations, amplitudes, weights, multiplicities


class ErgodicSampling:
    """
    Ergodic sampling.
    """

    __slots__ = ["owner", "total_step"]

    def __init__(self, owner):
        self.owner = owner

        # Calculate total step count for outside usage.
        self.total_step = 1
        for [l1, l2, orbit], edge in self.owner.physics_edges:
            self.total_step *= edge.dimension

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
        index = 0
        configuration = self._zero_configuration()
        configurations_pool = []
        while True:
            if index % mpi_size == mpi_rank:
                configurations_pool.append(configuration.copy())
            index += 1
            if index == self.total_step:
                break
            configuration = self._next_configuration(configuration)
        configurations_pool = torch.tensor(np.array(configurations_pool), device=self.owner.device)
        amplitudes_pool = self.owner(configurations_pool, enable_grad=False)
        non_zero = amplitudes_pool != 0
        configurations_pool = configurations_pool[non_zero]
        amplitudes_pool = amplitudes_pool[non_zero]
        weights_pool = torch.ones_like(amplitudes_pool.real)
        multiplicities_pool = torch.ones_like(amplitudes_pool, dtype=torch.int64)
        return configurations_pool, amplitudes_pool, weights_pool, multiplicities_pool
