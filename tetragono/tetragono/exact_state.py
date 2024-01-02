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

from copyreg import _slotnames
from scipy.linalg import eigh_tridiagonal
from .utility import show, showln
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
        # version 4 to version 5
        if state["data_version"] == 4:
            state["data_version"] = 5
        # version 5 to version 6
        if state["data_version"] == 5:
            self._v5_to_v6_attribute(state)
            state["data_version"] = 6
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
        names = [f"P_{l1}_{l2}_{orbit}" for [l1, l2, orbit], edge in self.physics_edges]
        edges = [edge for [l1, l2, orbit], edge in self.physics_edges]
        names.append("T")
        edges.append(self._total_symmetry_edge)
        vector = self.Tensor(names, edges).randn_()
        vector /= vector.norm_2()
        return vector

    def update(self, total_step):
        """
        Exact update the state by Lanczos algorithm.

        Parameters
        ----------
        total_step : int
            The step of power iteration.

        Returns
        -------
        float
            The result energy per site calculated by Lanczos algorithm.
        """
        if total_step <= 1:
            raise ValueError("Total iteration step should larger than 1")

        v = []
        alpha = []
        beta = []

        v.append(self.vector)

        for step in range(total_step):
            w = v[step].same_shape().zero_()
            for positions, value in self.hamiltonians:
                w += (
                    value  #
                    .edge_rename({
                        f"O{t}": f"P_{i}_{j}_{orbit}" for t, [i, j, orbit] in enumerate(positions)
                    })  #
                    .contract(
                        v[step],
                        {(f"I{t}", f"P_{i}_{j}_{orbit}") for t, [i, j, orbit] in enumerate(positions)},
                    ))
            alpha.append(complex(v[step].contract(w.conjugate(), {(name, name) for name in w.names})).real)
            if step == 0:
                w = w - alpha[step] * v[step]
            else:
                w = w - alpha[step] * v[step] - beta[step - 1] * v[step - 1]

                vals, vecs = eigh_tridiagonal(alpha, beta, lapack_driver="stemr", select="i", select_range=[0, 0])
                # Both stemr and stebz works, based on some test, it seems stemr is faster

                energy = vals[0]
                self.vector = (vecs.T @ v)[0]

                show(f"Exact update, {total_step=}, {step=}, energy={energy / self.site_number}")

            beta.append(w.norm_2())
            if beta[step] != 0:
                v.append(w / beta[step])
            else:
                showln("Exact update break since converged")
                total_step = step + 1
                break

        showln(f"Exact update done, {total_step=}, energy={energy / self.site_number}")
        return energy / self.site_number

    def observe(self, positions, observer):
        """
        Observe the state. If the oboserver is None and positions is an empty tuple, return normalization parameter
        $\\langle\\psi|\\psi\\rangle$.

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
                    observer.edge_rename({
                        f"O{t}": f"P_{l1}_{l2}_{orbit}" for t, [l1, l2, orbit] in enumerate(positions)
                    }),
                    {(f"P_{l1}_{l2}_{orbit}", f"I{t}") for t, [l1, l2, orbit] in enumerate(positions)},
                ))
        result = (
            result  #
            .contract(
                self.vector.conjugate(),
                {(f"P_{l1}_{l2}_{orbit}", f"P_{l1}_{l2}_{orbit}") for [l1, l2, orbit], edge in self.physics_edges} |
                {("T", "T")},
            ))
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
        for positions, observer in self.hamiltonians:
            energy += self.observe(positions, observer)
        return energy / self.observe(tuple(), None) / self.site_number
