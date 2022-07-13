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
from ..sampling_tools.tensor_element import tensor_element
from ..common_toolkit import mpi_rank, mpi_size


class Sampling:
    __slots__ = ["_owner", "_restrict_subspace"]

    def __init__(self, owner, restrict_subspace):
        """
        Create sampling object for the given ansatz product state.

        Parameters
        ----------
        owner : SamplingLattice
            The owner of this sampling object.
        restrict_subspace
            A function return bool to restrict sampling subspace.
        """
        self._owner = owner
        self._restrict_subspace = restrict_subspace

    def __call__(self):
        """
        Get the next sampling configuration

        Returns
        -------
        tuple[float, list[list[dict[int, EdgePoint]]]]
            The sampled weight in importance sampling, and the result configuration system.
        """
        raise NotImplementedError("Not implement in abstract sampling")

    def copy_configuration(self, configuration):
        owner = self._owner
        return [[{orbit: configuration[l1][l2][orbit]
                  for orbit in owner.physics_edges[l1, l2]}
                 for l2 in range(owner.L2)]
                for l1 in range(owner.L1)]


class SweepSampling(Sampling):
    """
    Sweep sampling object for ansatz product state.
    """

    __slots__ = ["configuration", "ws", "_hopping_hamiltonians", "_sweep_order"]

    def __init__(self, owner, restrict_subspace, configuration, hopping_hamiltonians):
        """
        Create sampling object.

        Parameters
        ----------
        owner : AnsatzProductState
            The owner of this sampling object
        restrict_subspace
            A function return bool to restrict sampling subspace.
        configuration : list[list[dict[int, EdgePoint]]]
            The initial configuration.
        hopping_hamiltonian : None | dict[tuple[tuple[int, int, int], ...], Tensor]
            The hamiltonian used in hopping, using the state hamiltonian if this is None.
        """
        super().__init__(owner, restrict_subspace)
        self.configuration = self.copy_configuration(configuration)
        [self.ws], _ = self._owner.weight_and_delta([self.configuration], set())
        if hopping_hamiltonians is not None:
            self._hopping_hamiltonians = hopping_hamiltonians
        else:
            self._hopping_hamiltonians = self._owner._hamiltonians
        self._sweep_order = sorted(self._hopping_hamiltonians.keys())

    def _single_term(self, positions, hamiltonian):
        owner = self._owner
        body = hamiltonian.rank // 2
        current_configuration = tuple(self.configuration[l1][l2][orbit] for [l1, l2, orbit] in positions)
        element_pool = tensor_element(hamiltonian)
        if current_configuration not in element_pool:
            return
        possible_hopping = element_pool[current_configuration]
        if len(possible_hopping) == 0:
            return
        hopping_number = len(possible_hopping)
        current_configuration_s, _ = list(possible_hopping.items())[TAT.random.uniform_int(0, hopping_number - 1)()]
        hopping_number_s = len(element_pool[current_configuration_s])
        if self._restrict_subspace is not None:
            replacement = {positions[i]: current_configurations_s[i] for i in range(body)}
            if not self._restrict_subspace(self.configuration, replacement):
                return
        configuration_s = self.copy_configuration(self.configuration)
        for i, [l1, l2, orbit] in enumerate(positions):
            configuration_s[l1][l2][orbit] = current_configuration_s[i]
        [wss], _ = self._owner.weight_and_delta([configuration_s], set())
        p = (np.linalg.norm(wss)**2) / (np.linalg.norm(self.ws)**2) * hopping_number / hopping_number_s
        if TAT.random.uniform_real(0, 1)() < p:
            self.configuration = configuration_s
            self.ws = wss

    def __call__(self):
        for positions in self._sweep_order:
            hamiltonian = self._hopping_hamiltonians[positions]
            self._single_term(positions, hamiltonian)
        for positions in reversed(self._sweep_order):
            hamiltonian = self._hopping_hamiltonians[positions]
            self._single_term(positions, hamiltonian)
        return self.ws**2, self.copy_configuration(self.configuration)


class ErgodicSampling(Sampling):
    """
    Ergodic sampling.
    """

    __slots__ = ["total_step", "_edges", "configuration"]

    def __init__(self, owner, restrict_subspace):
        """
        Create sampling object.

        Parameters
        ----------
        owner : AnsatzProductState
            The owner of this sampling object
        restrict_subspace
            A function return bool to restrict sampling subspace.
        """
        super().__init__(owner, restrict_subspace)

        self.configuration = [
            [{orbit: None for orbit in owner.physics_edges[l1, l2]} for l2 in range(owner.L2)] for l1 in range(owner.L1)
        ]

        self._edges = [[{orbit: edge
                         for orbit, edge in self._owner.physics_edges[l1, l2].items()}
                        for l2 in range(self._owner.L2)]
                       for l1 in range(self._owner.L1)]

        self.total_step = 1
        for l1 in range(self._owner.L1):
            for l2 in range(self._owner.L2):
                for orbit, edge in self._edges[l1][l2].items():
                    self.total_step *= edge.dimension

        self._zero_configuration()
        for t in range(mpi_rank):
            self._next_configuration()

    def _zero_configuration(self):
        owner = self._owner
        for l1 in range(owner.L1):
            for l2 in range(owner.L2):
                for orbit, edge in self._edges[l1][l2].items():
                    self.configuration[l1][l2][orbit] = edge.get_point_from_index(0)

    def _next_configuration(self):
        owner = self._owner
        for l1 in range(owner.L1):
            for l2 in range(owner.L2):
                for orbit, edge in self._edges[l1][l2].items():
                    index = edge.get_index_from_point(self.configuration[l1][l2][orbit])
                    index += 1
                    if index == edge.dimension:
                        self.configuration[l1][l2][orbit] = edge.get_point_from_index(0)
                    else:
                        self.configuration[l1][l2][orbit] = edge.get_point_from_index(index)
                        return

    def __call__(self):
        for t in range(mpi_size):
            self._next_configuration()
        possibility = 1.
        if self._restrict_subspace is not None:
            if not self._restrict_subspace(self.configuration):
                possibility = float("+inf")
        return possibility, self.copy_configuration(self.configuration)
