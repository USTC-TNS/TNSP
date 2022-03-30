#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2020-2022 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
from ..auxiliaries import DoubleLayerAuxiliaries, ThreeLineAuxiliaries
from ..sampling_lattice import Configuration
from ..common_variable import mpi_rank, mpi_size
from .tensor_element import tensor_element


class Sampling:
    """
    Helper type for run sampling for sampling lattice.
    """

    __slots__ = ["_owner", "_cut_dimension", "_restrict_subspace"]

    def __init__(self, owner, cut_dimension, restrict_subspace):
        """
        Create sampling object for the given sampling lattice.

        Parameters
        ----------
        owner : SamplingLattice
            The owner of this sampling object.
        cut_dimension : int
            The cut dimension in single layer auxiliaries.
        restrict_subspace
            A function return bool to restrict sampling subspace.
        """
        self._owner = owner
        self._cut_dimension = cut_dimension
        self._restrict_subspace = restrict_subspace

    def refresh_all(self):
        """
        Refresh the sampling system, need to be called after lattice tensor changed.
        """
        raise NotImplementedError("Not implement in abstract sampling")

    def __call__(self):
        """
        Get the next sampling configuration

        Returns
        -------
        tuple[float, Configuration]
            The sampled weight in importance sampling, and the result configuration system.
        """
        raise NotImplementedError("Not implement in abstract sampling")


class SweepSampling(Sampling):
    """
    Sweep sampling.
    """

    __slots__ = ["_sweep_order", "configuration", "_hopping_hamiltonians"]

    def __init__(self, owner, cut_dimension, restrict_subspace, hopping_hamiltonians):
        super().__init__(owner, cut_dimension, restrict_subspace)
        self.configuration = Configuration(self._owner, self._cut_dimension)
        if hopping_hamiltonians is not None:
            self._hopping_hamiltonians = hopping_hamiltonians
        else:
            self._hopping_hamiltonians = self._owner._hamiltonians
        # list[tuple[tuple[int, int, int], ...]]
        self._sweep_order = self._get_proper_position_order()

    def _single_term(self, positions, hamiltonian, ws):
        body = hamiltonian.rank // 2
        current_configuration = tuple(self.configuration[l1l2o] for l1l2o in positions)  # tuple[EdgePoint, ...]
        element_pool = tensor_element(hamiltonian)
        if current_configuration not in element_pool:
            return ws
        possible_hopping = element_pool[current_configuration]
        if possible_hopping:
            hopping_number = len(possible_hopping)
            configuration_new, element = list(possible_hopping.items())[TAT.random.uniform_int(0, hopping_number - 1)()]
            hopping_number_s = len(element_pool[configuration_new])
            replacement = {positions[i]: configuration_new[i] for i in range(body)}
            if self._restrict_subspace is not None:
                if not self._restrict_subspace(self.configuration, replacement):
                    return ws
            wss = self.configuration.replace(replacement)  # which return a tensor, we only need its norm
            p = (wss.norm_2()**2) / (ws.norm_2()**2) * hopping_number / hopping_number_s
            if TAT.random.uniform_real(0, 1)() < p:
                ws = wss
                for i in range(body):
                    self.configuration[positions[i]] = configuration_new[i]
        return ws

    def __call__(self):
        self.configuration = self.configuration.copy()
        if not self.configuration.valid():
            raise RuntimeError("Configuration not initialized")
        ws = self.configuration.hole(())
        for positions in self._sweep_order:
            hamiltonian = self._hopping_hamiltonians[positions]
            ws = self._single_term(positions, hamiltonian, ws)
        for positions in reversed(self._sweep_order):
            hamiltonian = self._hopping_hamiltonians[positions]
            ws = self._single_term(positions, hamiltonian, ws)
        return ws.norm_2()**2, self.configuration

    def _get_proper_position_order(self):
        L1 = self._owner.L1
        L2 = self._owner.L2
        positions = set(self._hopping_hamiltonians.keys())
        result = []
        # Single site auxiliary use horizontal style contract by default
        for l1 in range(L1):
            for l2 in range(L2):
                # Single site first, if not one useless right auxiliary tensor will be calculated.
                new_positions = set()
                for ps in positions:
                    if len([p for p in (p[:2] for p in ps) if p not in ((l1, l2))]) == 0:
                        result.append(ps)
                    else:
                        new_positions.add(ps)
                positions = new_positions
                new_positions = set()
                for ps in positions:
                    if len([p for p in (p[:2] for p in ps) if p not in ((l1, l2), (l1, l2 + 1))]) == 0:
                        result.append(ps)
                    else:
                        new_positions.add(ps)
                positions = new_positions
        for l2 in range(L2):
            for l1 in range(L1):
                new_positions = set()
                for ps in positions:
                    if len([p for p in (p[:2] for p in ps) if p not in ((l1, l2), (l1 + 1, l2))]) == 0:
                        result.append(ps)
                    else:
                        new_positions.add(ps)
                positions = new_positions
        if len(positions) != 0:
            raise NotImplementedError("Not implemented hamiltonian")
        return result

    def refresh_all(self):
        self.configuration.refresh_all()


class ErgodicSampling(Sampling):
    """
    Ergodic sampling.
    """

    __slots__ = ["total_step", "_edges", "configuration"]

    def __init__(self, owner, cut_dimension, restrict_subspace):
        super().__init__(owner, cut_dimension, restrict_subspace)

        self.configuration = Configuration(self._owner, self._cut_dimension)

        self._edges = [[{
            orbit: self._owner[l1, l2].edges(f"P{orbit}") for orbit, edge in self._owner.physics_edges[l1, l2].items()
        } for l2 in range(self._owner.L2)] for l1 in range(self._owner.L1)]

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
                    self.configuration[l1, l2, orbit] = edge.get_point_from_index(0)

    def _next_configuration(self):
        owner = self._owner
        for l1 in range(owner.L1):
            for l2 in range(owner.L2):
                for orbit, edge in self._edges[l1][l2].items():
                    index = edge.get_index_from_point(self.configuration[l1, l2, orbit])
                    index += 1
                    if index == edge.dimension:
                        self.configuration[l1, l2, orbit] = edge.get_point_from_index(0)
                    else:
                        self.configuration[l1, l2, orbit] = edge.get_point_from_index(index)
                        return

    def refresh_all(self):
        self.configuration.refresh_all()

    def __call__(self):
        self.configuration = self.configuration.copy()
        for t in range(mpi_size):
            self._next_configuration()
        possibility = 1.
        if self._restrict_subspace is not None:
            if not self._restrict_subspace(self.configuration):
                possibility = float("+inf")
        return possibility, self.configuration


class DirectSampling(Sampling):
    """
    Direct sampling.
    """

    __slots__ = ["_double_layer_cut_dimension", "_double_layer_auxiliaries"]

    def __init__(self, owner, cut_dimension, restrict_subspace, double_layer_cut_dimension):
        super().__init__(owner, cut_dimension, restrict_subspace)
        self._double_layer_cut_dimension = double_layer_cut_dimension
        self.refresh_all()

    def refresh_all(self):
        owner = self._owner
        self._double_layer_auxiliaries = DoubleLayerAuxiliaries(owner.L1, owner.L2, self._double_layer_cut_dimension,
                                                                True, owner.Tensor)
        for l1 in range(owner.L1):
            for l2 in range(owner.L2):
                this = owner[l1, l2].copy()
                self._double_layer_auxiliaries[l1, l2, "n"] = this
                self._double_layer_auxiliaries[l1, l2, "c"] = this.conjugate()

    def __call__(self):
        owner = self._owner
        configuration = Configuration(owner, self._cut_dimension)
        random = TAT.random.uniform_real(0, 1)
        for l1 in range(owner.L1):
            for l2 in range(owner.L2):
                for orbit in owner.physics_edges[l1, l2]:
                    configuration[l1, l2, orbit] = None
        possibility = 1.
        for l1 in range(owner.L1):

            three_line_auxiliaries = ThreeLineAuxiliaries(owner.L2, owner.Tensor, self._cut_dimension)
            for l2 in range(owner.L2):
                tensor_1 = configuration._up_to_down_site[l1 - 1, l2]()
                three_line_auxiliaries[0, l2, "n"] = tensor_1
                three_line_auxiliaries[0, l2, "c"] = tensor_1.conjugate()
                tensor_2 = owner[l1, l2]
                three_line_auxiliaries[1, l2, "n"] = tensor_2
                three_line_auxiliaries[1, l2, "c"] = tensor_2.conjugate()
                three_line_auxiliaries[2, l2] = self._double_layer_auxiliaries._down_to_up_site[l1 + 1, l2]()

            for l2 in range(owner.L2):
                shrinked_site_tensor = owner[l1, l2]
                config = {}
                shrinkers = configuration._get_shrinker((l1, l2), config)
                for orbit in owner.physics_edges[l1, l2]:
                    hole = three_line_auxiliaries.hole(l2, orbit).transpose(["I0", "O0"])
                    hole_edge = hole.edges("O0")
                    rho = np.array([])
                    for seg in hole_edge.segment:
                        symmetry, _ = seg
                        block_rho = hole.blocks[[("I0", -symmetry), ("O0", symmetry)]]
                        diag_rho = np.diagonal(block_rho)
                        rho = np.array([*rho, *diag_rho])
                    rho = rho.real
                    if len(rho) == 0:
                        return self()
                    rho = rho / np.sum(rho)
                    choice = self._choice(random(), rho)
                    possibility *= rho[choice]
                    config[orbit] = configuration[l1, l2, orbit] = hole_edge.get_point_from_index(choice)
                    _, shrinker = next(shrinkers)
                    shrinked_site_tensor = shrinked_site_tensor.contract(shrinker.edge_rename({"P": f"P{orbit}"}),
                                                                         {(f"P{orbit}", "Q")})
                    three_line_auxiliaries[1, l2, "n"] = shrinked_site_tensor
                    three_line_auxiliaries[1, l2, "c"] = shrinked_site_tensor.conjugate()

        if self._restrict_subspace is not None:
            if not self._restrict_subspace(configuration):
                return self()
        return possibility, configuration

    @staticmethod
    def _choice(p, rho):
        for i, r in enumerate(rho):
            p -= r
            if p < 0:
                return i
        return i
