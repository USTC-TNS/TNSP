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

from copyreg import _slotnames
from .common_toolkit import show, showln
from .abstract_state import AbstractState


class ExactState(AbstractState):
    """
    State for exact diagonalization.
    """

    __slots__ = ["vector"]

    def __setstate__(self, state):
        # before data_version mechanism, state is (None, state)
        if isinstance(state, tuple):
            state = state[1]
        # before data_version mechanism, there is no data_version field
        if "data_version" not in state:
            state["data_version"] = 0
        # version 0 to version 1
        if state["data_version"] == 0:
            state["data_version"] = 1
        # version 1 to version 2
        if state["data_version"] == 1:
            state["data_version"] = 2
        # version 2 to version 3
        if state["data_version"] == 2:
            self._v2_to_v3_rename(state)
            state["data_version"] = 3
        # version 3 to version 4
        if state["data_version"] == 3:
            state["data_version"] = 4
        # setstate
        for key, value in state.items():
            setattr(self, key, value)

    def __getstate__(self):
        # getstate
        state = {key: getattr(self, key) for key in _slotnames(self.__class__)}
        return state

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
            f"P_{l1}_{l2}_{orbit}" for l1 in range(self.L1) for l2 in range(self.L2)
            for orbit in sorted(self.physics_edges[l1, l2])
        ]
        edges = [
            self.physics_edges[l1, l2, orbit] for l1 in range(self.L1) for l2 in range(self.L2)
            for orbit in sorted(self.physics_edges[l1, l2])
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
        if total_step <= 0:
            raise ValueError("Total iteration step should be a positive integer")

        # Apprixmate the total energy
        total_approximate_energy = abs(approximate_energy) * self.site_number
        energy = 0

        for step in range(total_step):
            # let a be total_approximate_energy
            # v <- a v - H v = (a - H) v => E = a - |v'|/|v|
            # temporary_vector: H v
            temporary_vector = self.vector.same_shape().zero()
            for positions, value in self._hamiltonians.items():
                # H v = sum_i H_i v
                temporary_vector += (
                    value  #
                    .edge_rename({f"O{t}": f"P_{i}_{j}_{orbit}" for t, [i, j, orbit] in enumerate(positions)})  #
                    .contract(self.vector,
                              {(f"I{t}", f"P_{i}_{j}_{orbit}") for t, [i, j, orbit] in enumerate(positions)}))
            # To calculate a v - H v => v *= a; v -= H v
            self.vector *= total_approximate_energy
            self.vector -= temporary_vector

            # Get the new norm, original norm should be set as 1.
            # If it is initialize by exact state, it is 1,
            # If not, after first step, it is also 1.
            # So the norm is a - |v'|/|v|, when converging, |v'|/|v| -> H
            # Then getting energy by a - norm
            norm = float(self.vector.norm_2())
            energy = total_approximate_energy - norm

            # Normalize state to ensure norm is 1 in the next iteration.
            self.vector /= norm

            show(f"Exact update, {total_step=}, {step=}, energy={energy / self.site_number}")

        showln(f"Exact update done, {total_step=}, energy={energy / self.site_number}")
        return energy / self.site_number

    def observe(self, positions, observer):
        """
        Observe the state. If the oboserver is None and positions is an empty tuple, return normalization parameter
        $\langle\psi|\psi\rangle$.

        Parameters
        ----------
        positions : tuple[tuple[int, int, int] | tuple[int, int], ...]
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
            positions = [position if len(position) == 3 else (position[0], position[1], 0) for position in positions]
            result = (
                self.vector  #
                .contract(
                    observer.edge_rename(
                        {f"O{t}": f"P_{l1}_{l2}_{orbit}" for t, [l1, l2, orbit] in enumerate(positions)}),
                    {(f"P_{l1}_{l2}_{orbit}", f"I{t}") for t, [l1, l2, orbit] in enumerate(positions)}))
        result = (
            result  #
            .contract(
                self.vector.conjugate(), {(f"P_{l1}_{l2}_{orbit}", f"P_{l1}_{l2}_{orbit}") for l1 in range(self.L1)
                                          for l2 in range(self.L2)
                                          for orbit in self.physics_edges[l1, l2]} | {("T", "T")}))
        return complex(result).real

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
