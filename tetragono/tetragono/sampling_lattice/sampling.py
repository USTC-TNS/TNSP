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
from ..common_toolkit import mpi_rank, mpi_size
from ..tensor_element import tensor_element
from .lattice import Configuration, SamplingLattice


class Sampling:
    """
    Helper type for run sampling for sampling lattice.
    """

    __slots__ = ["owner", "_cut_dimension", "_restrict_subspace"]

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
        self.owner: SamplingLattice = owner

        # This is cut dimension used by Configuration object, since for PEPS ansatz, approximation is needed when
        # calculate w(s), which is controled by a cut dimension parameter, which is often called Dc.
        self._cut_dimension = cut_dimension

        # Restrict subspace when doing sampling.
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
        # Sweep sampling need to store a configuration to generate the next one.
        self.configuration = Configuration(self.owner, self._cut_dimension)
        if hopping_hamiltonians is not None:
            self._hopping_hamiltonians = hopping_hamiltonians
        else:
            self._hopping_hamiltonians = self.owner._hamiltonians
        # list[tuple[tuple[int, int, int], ...]]
        # Sweep sampling needs a proper sweep order to trigger less auxiliary refresh.
        self._sweep_order = self._get_proper_position_order()

    def _single_term(self, positions, hamiltonian, ws):
        body = len(positions)
        # tuple[EdgePoint, ...]
        positions_configuration = tuple(self.configuration[l1l2o] for l1l2o in positions)
        element_pool = tensor_element(hamiltonian)
        if positions_configuration not in element_pool:
            return ws
        possible_hopping = element_pool[positions_configuration]
        if possible_hopping:
            hopping_number = len(possible_hopping)
            positions_configuration_s = list(possible_hopping)[TAT.random.uniform_int(0, hopping_number - 1)()]
            hopping_number_s = len(element_pool[positions_configuration_s])
            replacement = {positions[i]: positions_configuration_s[i] for i in range(body)}
            if self._restrict_subspace is not None:
                if not self._restrict_subspace(self.configuration, replacement):
                    # Then wss is zero forcely, hopping possibility is zero definitely, so hopping failed.
                    return ws
            wss = self.configuration.replace(replacement)  # which return a tensor, we only need its norm
            p = (wss / ws).norm_2()**2 * hopping_number / hopping_number_s
            if TAT.random.uniform_real(0, 1)() < p:
                # Hopping success, update ws and configuration
                ws = wss
                for i in range(body):
                    self.configuration[positions[i]] = positions_configuration_s[i]
        return ws

    def __call__(self):
        if not self.configuration.valid():
            raise RuntimeError("Configuration not initialized")
        ws = self.configuration.hole(())
        # Hopping twice from different direction to keep detailed balance.
        for positions in self._sweep_order:
            hamiltonian = self._hopping_hamiltonians[positions]
            ws = self._single_term(positions, hamiltonian, ws)
        for positions in reversed(self._sweep_order):
            hamiltonian = self._hopping_hamiltonians[positions]
            ws = self._single_term(positions, hamiltonian, ws)
        return ws.norm_2()**2, self.configuration.copy()

    def _get_proper_position_order(self):
        L1 = self.owner.L1
        L2 = self.owner.L2
        positions = set(self._hopping_hamiltonians.keys())
        result = []
        # Single site auxiliary use horizontal style contract by default
        for l1 in range(L1):
            for l2 in range(L2):
                # Single site first, if not, one useless right auxiliary tensor will be calculated.
                remained_positions = set()
                for ps in positions:
                    if len([p for p in (p[:2] for p in ps) if p not in ((l1, l2))]) == 0:
                        result.append(ps)
                    else:
                        remained_positions.add(ps)
                positions = remained_positions
                remained_positions = set()
                for ps in positions:
                    if len([p for p in (p[:2] for p in ps) if p not in ((l1, l2), (l1, l2 + 1))]) == 0:
                        result.append(ps)
                    else:
                        remained_positions.add(ps)
                positions = remained_positions
        for l2 in range(L2):
            for l1 in range(L1):
                remained_positions = set()
                for ps in positions:
                    if len([p for p in (p[:2] for p in ps) if p not in ((l1, l2), (l1 + 1, l2))]) == 0:
                        result.append(ps)
                    else:
                        remained_positions.add(ps)
                positions = remained_positions
        if len(positions) != 0:
            raise NotImplementedError("Not implemented hamiltonian")
        return result

    def refresh_all(self):
        self.configuration.refresh_all()


class ErgodicSampling(Sampling):
    """
    Ergodic sampling.
    """

    __slots__ = ["total_step", "configuration"]

    def __init__(self, owner, cut_dimension, restrict_subspace):
        super().__init__(owner, cut_dimension, restrict_subspace)

        # The current configuration.
        self.configuration = Configuration(self.owner, self._cut_dimension)

        # Calculate total step count for outside usage.
        self.total_step = 1
        for l1 in range(self.owner.L1):
            for l2 in range(self.owner.L2):
                for orbit, edge in self.owner.physics_edges[l1, l2].items():
                    self.total_step *= edge.dimension

        # Initialize the current configuration
        self._zero_configuration()
        # And apply the offset because of mpi parallel
        for t in range(mpi_rank):
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

    def refresh_all(self):
        self.configuration.refresh_all()

    def __call__(self):
        for t in range(mpi_size):
            self._next_configuration()
        possibility = 1.
        if self._restrict_subspace is not None:
            if not self._restrict_subspace(self.configuration):
                # This configuration should be impossible, so set possibility to infinity, then it will get the correct
                # result when reweigting.
                possibility = np.inf
        return possibility, self.configuration.copy()


class DirectSampling(Sampling):
    """
    Direct sampling.
    """

    __slots__ = ["_double_layer_cut_dimension", "_double_layer_auxiliaries"]

    def __init__(self, owner, cut_dimension, restrict_subspace, double_layer_cut_dimension):
        super().__init__(owner, cut_dimension, restrict_subspace)
        # This is the cut dimension used in calculating unsampled part of lattice, which is a double layer auxiliaries
        # system.
        self._double_layer_cut_dimension = double_layer_cut_dimension
        # Refresh all will refresh the double layer auxiliaries.
        self.refresh_all()

    def refresh_all(self):
        self._double_layer_auxiliaries = DoubleLayerAuxiliaries(self.owner.L1, self.owner.L2,
                                                                self._double_layer_cut_dimension, True,
                                                                self.owner.Tensor)
        for l1 in range(self.owner.L1):
            for l2 in range(self.owner.L2):
                this = self.owner[l1, l2].copy()
                self._double_layer_auxiliaries[l1, l2, "n"] = this
                self._double_layer_auxiliaries[l1, l2, "c"] = this.conjugate()

    def __call__(self):
        configuration = Configuration(self.owner, self._cut_dimension)
        # A random generator
        random = TAT.random.uniform_real(0, 1)
        # The priori possibility
        possibility = 1.
        for l1 in range(self.owner.L1):

            # This three line auxiliaries contains one line for the sampled part, one line for the unsampled part and
            # one line for the sampling line part.
            three_line_auxiliaries = ThreeLineAuxiliaries(self.owner.L2, self.owner.Tensor, self._cut_dimension)
            for l2 in range(self.owner.L2):
                tensor_1 = configuration._up_to_down_site[l1 - 1, l2]()
                three_line_auxiliaries[0, l2, "n"] = tensor_1
                three_line_auxiliaries[0, l2, "c"] = tensor_1.conjugate()
                tensor_2 = self.owner[l1, l2]
                three_line_auxiliaries[1, l2, "n"] = tensor_2
                three_line_auxiliaries[1, l2, "c"] = tensor_2.conjugate()
                three_line_auxiliaries[2, l2] = self._double_layer_auxiliaries._down_to_up_site[l1 + 1, l2]()

            for l2 in range(self.owner.L2):
                # Choose configuration and calculate shrinked tensor orbit by orbit.
                shrinked_site_tensor = self.owner[l1, l2]
                config = {}
                # This iterator will read the config dict when yielding, so update the config and then calculate the
                # next item of this iterator.
                shrinkers = configuration._get_shrinker((l1, l2), config)
                # The hole of this site.
                # This hole style is like double layer auxiliaries hole.
                site_hole = three_line_auxiliaries.hole(l2)
                unsampled_orbits = set(self.owner.physics_edges[l1, l2])
                # The orbit order is important, because of iterator shrinkers.
                for orbit in sorted(self.owner.physics_edges[l1, l2]):
                    # Trace all unsampled orbits.
                    # The transpose ensure elements are all positive.
                    unsampled_orbits.remove(orbit)
                    hole = (
                        site_hole  #
                        .trace({(f"I{unsampled_orbit}", f"O{unsampled_orbit}") for unsampled_orbit in unsampled_orbits}
                              )  #
                        .edge_rename({
                            f"I{orbit}": "I",
                            f"O{orbit}": "O"
                        })  #
                        .transpose(["I", "O"]))
                    hole_edge = hole.edges("O")
                    # Calculate rho for all the segments of the physics edge of this orbit
                    rho = []
                    for seg in hole_edge.segment:
                        symmetry, _ = seg
                        block_rho = hole.blocks[[("I", -symmetry), ("O", symmetry)]]
                        diag_rho = np.diagonal(block_rho)
                        rho = [*rho, *diag_rho]
                    rho = np.array(rho).real
                    if len(rho) == 0:
                        # Block mismatch, redo a sampling.
                        return self()
                    rho = rho / np.sum(rho)
                    choice = self._choice(random(), rho)
                    # Choose the configuration of this orbit
                    possibility *= rho[choice]
                    config[orbit] = configuration[l1, l2, orbit] = hole_edge.get_point_from_index(choice)
                    # config updated for this orbit, now calculating the next item of iterator becomes valid.
                    _, shrinker = next(shrinkers)
                    # shrink the tensor and sampled orbit, and update the three line auxiliaries.
                    shrinked_site_tensor = (
                        shrinked_site_tensor  #
                        .contract(shrinker.edge_rename({"P": f"P{orbit}"}), {(f"P{orbit}", "Q")}))
                    site_hole = (
                        site_hole  #
                        .contract(shrinker.edge_rename({"P": f"O{orbit}"}), {(f"O{orbit}", "Q")})  #
                        .contract(shrinker.conjugate().edge_rename({"P": f"I{orbit}"}), {(f"I{orbit}", "Q")})  #
                        .trace({(f"I{orbit}", f"O{orbit}")}))
                    three_line_auxiliaries[1, l2, "n"] = shrinked_site_tensor
                    three_line_auxiliaries[1, l2, "c"] = shrinked_site_tensor.conjugate()

        if self._restrict_subspace is not None:
            if not self._restrict_subspace(configuration):
                # This configuration should not be selected, redo a sampling.
                return self()
        return possibility, configuration

    @staticmethod
    def _choice(p, rho):
        """
        Choose one element in a list with their possibility stored in rho, and the given random number p.
        """
        for i, r in enumerate(rho):
            p -= r
            if p < 0:
                return i
        # Maybe p is a small but positive number now, because of numeric error.
        return i
