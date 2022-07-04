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


class Sampling:
    """
    Metropois sampling object for multiple product state.
    """

    __slots__ = ["_owner", "configuration", "_hopping_hamiltonians", "_restrict_subspace", "ws"]

    def __init__(self, owner, configuration, hopping_hamiltonians, restrict_subspace):
        """
        Create sampling object.

        Parameters
        ----------
        owner : MultipleProductState
            The owner of this sampling object
        configuration : list[list[dict[int, EdgePoint]]]
            The initial configuration.
        hopping_hamiltonian : None | dict[tuple[tuple[int, int, int], ...], Tensor]
            The hamiltonian used in hopping, using the state hamiltonian if this is None.
        restrict_subspace
            A function return bool to restrict sampling subspace.
        """
        self._owner = owner
        self.configuration = [[{orbit: configuration[l1][l2][orbit]
                                for orbit in owner.physics_edges[l1, l2]}
                               for l2 in range(owner.L2)]
                              for l1 in range(owner.L1)]
        [self.ws], _ = self._owner.weight_and_delta([self.configuration], set())
        if hopping_hamiltonians is not None:
            self._hopping_hamiltonians = hopping_hamiltonians
        else:
            self._hopping_hamiltonians = self._owner._hamiltonians
        self._hopping_hamiltonians = list(self._hopping_hamiltonians.items())
        self._restrict_subspace = restrict_subspace

    def __call__(self):
        """
        Get the next configuration.
        """
        owner = self._owner
        hamiltonian_number = len(self._hopping_hamiltonians)
        positions, hamiltonian = self._hopping_hamiltonians[TAT.random.uniform_int(0, hamiltonian_number - 1)()]
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
        configuration_s = [[{orbit: self.configuration[l1][l2][orbit]
                             for orbit in owner.physics_edges[l1, l2]}
                            for l2 in range(owner.L2)]
                           for l1 in range(owner.L1)]
        for i, [l1, l2, orbit] in enumerate(positions):
            configuration_s[l1][l2][orbit] = current_configuration_s[i]
        [wss], _ = self._owner.weight_and_delta([configuration_s], set())
        p = (np.linalg.norm(wss)**2) / (np.linalg.norm(self.ws)**2) * hopping_number / hopping_number_s
        if TAT.random.uniform_real(0, 1)() < p:
            self.configuration = configuration_s
            self.ws = wss
