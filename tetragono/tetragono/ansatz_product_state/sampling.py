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
import TAT
from ..common_toolkit import mpi_rank, mpi_size
from ..tensor_element import tensor_element
from .state import Configuration, AnsatzProductState


class Sampling:
    """
    Helper type for run sampling for ansatz product state.
    """

    __slots__ = ["owner", "multichain_number", "_restrict_subspace"]

    def __init__(self, owner, multichain_number, restrict_subspace):
        """
        Create sampling object for the given ansatz product state.

        Parameters
        ----------
        owner : SamplingLattice
            The owner of this sampling object.
        multichain_number : int
            The multichain number.
        restrict_subspace
            A function return bool to restrict sampling subspace.
        """
        self.owner: AnsatzProductState = owner
        self.multichain_number = multichain_number
        self._restrict_subspace = restrict_subspace

    def __call__(self):
        """
        Get the next sampling configuration

        Returns
        -------
        list[tuple[float, Configuration]]
            The sampled weight in importance sampling, and the result configuration system.
        """
        raise NotImplementedError("Not implement in abstract sampling")


class SweepSampling(Sampling):
    """
    Sweep sampling object for ansatz product state.
    """

    __slots__ = ["_sweep_order", "configuration", "_hopping_hamiltonians"]

    def __init__(self, owner, multichain_number, restrict_subspace, hopping_hamiltonians):
        """
        Create sampling object.

        Parameters
        ----------
        owner : AnsatzProductState
            The owner of this sampling object
        multichain_number : int
            The multichain number.
        restrict_subspace
            A function return bool to restrict sampling subspace.
        hopping_hamiltonian : None | dict[tuple[tuple[int, int, int], ...], Tensor]
            The hamiltonian used in hopping, using the state hamiltonian if this is None.
        """
        super().__init__(owner, multichain_number, restrict_subspace)
        config_template = Configuration(self.owner)
        self.configuration = [config_template.copy() for _ in range(multichain_number)]
        if hopping_hamiltonians is not None:
            self._hopping_hamiltonians = hopping_hamiltonians
        else:
            self._hopping_hamiltonians = self.owner._hamiltonians
        # The order for ansatz product state is not important
        self._sweep_order = sorted(self._hopping_hamiltonians.keys())

    def _single_term(self, positions, hamiltonian, ws):
        body = len(positions)
        # tuple[EdgePoint, ...]
        element_pool = tensor_element(hamiltonian)

        # [Configuration]
        configuration_list = []
        # [None | (index in configuration_list, hopping_number, hopping_number_s)] with the same length to chain
        configuration_data = []
        for chain in range(self.multichain_number):
            positions_configuration = tuple(self.configuration[chain][l1l2o] for l1l2o in positions)
            if positions_configuration not in element_pool:
                configuration_data.append(None)
                continue
            possible_hopping = element_pool[positions_configuration]
            if not possible_hopping:
                configuration_data.append(None)
                continue
            hopping_number = len(possible_hopping)
            positions_configuration_s = list(possible_hopping)[TAT.random.uniform_int(0, hopping_number - 1)()]
            hopping_number_s = len(element_pool[positions_configuration_s])
            if self._restrict_subspace is not None:
                replacement = {positions[i]: positions_configuration_s[i] for i in range(body)}
                if not self._restrict_subspace(self.configuration[chain], replacement):
                    # Then wss is zero, hopping failed
                    configuration_data.append(None)
                    continue
            # Copy the original configuration and update the selected site and oribt
            configuration_s = self.configuration[chain].copy()
            for i, l1l2o in enumerate(positions):
                configuration_s[l1l2o] = positions_configuration_s[i]
            configuration_data.append((len(configuration_list), hopping_number / hopping_number_s))
            configuration_list.append(configuration_s)
        # Then calculate the wss
        wss, _ = self.owner.ansatz.weight_and_delta(configuration_list, False)
        for chain in range(self.multichain_number):
            configuration_package = configuration_data[chain]
            if configuration_package is None:
                continue
            index, hopping_number_over_hopping_number_s = configuration_package
            wss_over_ws = wss[index] / ws[chain]
            p = abs(wss_over_ws)**2 * hopping_number_over_hopping_number_s
            if TAT.random.uniform_real(0, 1)() < p:
                # Hopping success, update configuration and ws
                ws[chain] = wss[index]
                self.configuration[chain] = configuration_list[index]
        return ws

    def __call__(self):
        ws, _ = self.owner.ansatz.weight_and_delta(self.configuration, False)
        # Hopping twice from different direction to keep detailed balance.
        for positions in self._sweep_order:
            hamiltonian = self._hopping_hamiltonians[positions]
            ws = self._single_term(positions, hamiltonian, ws)
        for positions in reversed(self._sweep_order):
            hamiltonian = self._hopping_hamiltonians[positions]
            ws = self._single_term(positions, hamiltonian, ws)
        return [
            (np.linalg.norm(ws_i)**2, configuration_i.copy()) for ws_i, configuration_i in zip(ws, self.configuration)
        ]


class ErgodicSampling(Sampling):
    """
    Ergodic sampling.
    """

    __slots__ = ["total_step", "configuration"]

    def __init__(self, owner, multichain_number, restrict_subspace):
        super().__init__(owner, multichain_number, restrict_subspace)

        # The current configuration.
        self.configuration = Configuration(self.owner)

        # Calculate total step count for outside usage.
        self.total_step = 1
        for l1 in range(self.owner.L1):
            for l2 in range(self.owner.L2):
                for orbit, edge in self.owner.physics_edges[l1, l2].items():
                    self.total_step *= edge.dimension

        # Initialize the current configuration
        self._zero_configuration()
        # And apply the offset because of mpi parallel
        for t in range(mpi_rank * self.multichain_number):
            self._next_configuration()

    def _zero_configuration(self):
        for l1 in range(self.owner.L1):
            for l2 in range(self.owner.L2):
                for orbit, edge in self.owner.physics_edges[l1, l2].items():
                    self.configuration[l1, l2, orbit] = edge.get_point_from_index(0)

    def _next_configuration(self):
        for l1 in range(self.owner.L1):
            for l2 in range(self.owner.L2):
                for orbit, edge in self.owner.physics_edges[l1, l2].items():
                    index = edge.get_index_from_point(self.configuration[l1, l2, orbit])
                    index += 1
                    if index == edge.dimension:
                        self.configuration[l1, l2, orbit] = edge.get_point_from_index(0)
                    else:
                        self.configuration[l1, l2, orbit] = edge.get_point_from_index(index)
                        return

    def _current_sampling(self):
        possibility = 1.
        if self._restrict_subspace is not None:
            if not self._restrict_subspace(self.configuration):
                # This configuration should be impossible, so set possibility to infinity, then it will get the correct
                # result when reweigting.
                possibility = np.inf
        return possibility, self.configuration.copy()

    def __call__(self):
        result = []
        for _ in range(self.multichain_number):
            result.append(self._current_sampling())
            self._next_configuration()
        for t in range(self.multichain_number * (mpi_size - 1)):
            self._next_configuration()
        return result
