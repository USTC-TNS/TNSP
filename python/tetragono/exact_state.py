#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2020-2021 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

from __future__ import annotations
from .common_variable import clear_line
from .abstract_state import AbstractState


class ExactState(AbstractState):
    """
    State for exact diagonalization.
    """

    __slots__ = ["vector"]

    def __init__(self, abstract):
        """
        Create an exact state from abstract state.

        Parameters
        ----------
        abstract : AbstractState
            The abstract state used to be converted to exact state.
        """
        super()._init_by_copy(abstract)
        self.vector = self._construct_vector()

    def _construct_vector(self):
        """
        Create state vector for this exact state.

        Returns
        -------
        Tensor
            The state vector
        """
        names = [
            f"P_{i}_{j}_{orbit}" for i in range(self.L1) for j in range(self.L2)
            for orbit, edge in self.physics_edges[i, j].items()
        ]
        edges = [
            edge for i in range(self.L1) for j in range(self.L2) for orbit, edge in self.physics_edges[i, j].items()
        ]
        names.append("T")
        edges.append(self._total_symmetry_edge)
        vector = self.Tensor(names, edges).randn()
        vector /= vector.norm_2()
        return vector

    def update(self, total_step, approximate_energy):
        """
        Exact update the state by power iteration.

        Parameters
        ----------
        total_step : int
            The step of power iteration.
        approximate_energy : float
            The approximate energy of this system, the matrix used by power iteration is $a I - H$, so the lowest energy
            become the largest, to avoid high energy also gets large absolute value, $a$ should be bigger than
            $(Eh + El)/2$, where $Eh$ and $El$ is the largest and lowest eigenenergy of $H$.

        Returns
        -------
        float
            The result energy per site calculated by power iteration.
        """
        total_approximate_energy = abs(approximate_energy) * self.site_number
        energy = 0
        if total_step <= 0:
            raise ValueError("Total iteration step should be a positive integer")
        for step in range(total_step):
            temporary_vector = self.vector.same_shape().zero()
            for positions, value in self._hamiltonians.items():
                temporary_vector += value.edge_rename({
                    f"O{t}": f"P_{i}_{j}_{orbit}" for t, [i, j, orbit] in enumerate(positions)
                }).contract(self.vector, {(f"I{t}", f"P_{i}_{j}_{orbit}") for t, [i, j, orbit] in enumerate(positions)})
            self.vector *= total_approximate_energy
            self.vector -= temporary_vector
            # v <- a v - H v = (a - H) v => E = a - |v'|/|v|
            norm = float(self.vector.norm_2())
            energy = total_approximate_energy - norm
            self.vector /= norm
            print(clear_line, f"Exact update, {total_step=}, {step=}, energy={energy / self.site_number}", end="\r")
        print(clear_line, f"Exact update done, {total_step=}, energy={energy / self.site_number}")
        return energy / self.site_number

    def observe(self, positions, observer):
        """
        Observe the state. If the oboserver is None and positions is an empty tuple, return normalization parameter
        $\langle\psi|\psi\rangle$.

        Parameters
        ----------
        positions : tuple[tuple[int, int, int], ...]
            The site and their orbits used to observe.
        observer : Tensor
            The operator used to observe.

        Returns
        -------
        float
            The observer result, which is not normalized.
        """
        if len(positions) == 0:
            result = self.vector
        else:
            result = self.vector.contract(
                observer.edge_rename({f"O{t}": f"P_{i}_{j}_{orbit}" for t, [i, j, orbit] in enumerate(positions)}),
                {(f"P_{i}_{j}_{orbit}", f"I{t}") for t, [i, j, orbit] in enumerate(positions)})
        result = result.contract(
            self.vector.conjugate(), {(f"P_{i}_{j}_{orbit}", f"P_{i}_{j}_{orbit}") for i in range(self.L1)
                                      for j in range(self.L2)
                                      for orbit, edge in self.physics_edges[i, j].items()} | {("T", "T")})
        return float(result)

    def observe_energy(self):
        """
        Observe the energy per site.

        Returns
        -------
        float
            The energy per site calculated by observing.
        """
        energy = 0
        for positions, observer in self._hamiltonians.items():
            energy += self.observe(positions, observer)
        return energy / self.observe(tuple(), None) / self.site_number
