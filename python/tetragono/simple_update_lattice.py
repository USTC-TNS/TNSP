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
from copyreg import _slotnames
from .double_layer_auxiliaries import DoubleLayerAuxiliaries
from .abstract_lattice import AbstractLattice
from .common_variable import clear_line


class SimpleUpdateLatticeEnvironment:
    """
    Environment handler for simple update lattice.
    """

    __slots__ = ["_owner"]

    def __init__(self, owner):
        """
        Create environment handler

        Parameters
        ----------
        owner : SimpleUpdateLattice
            The owner of this handler.
        """
        self._owner = owner

    def __getitem__(self, where):
        """
        Get the environment by coordinate of site and direction.

        Parameters
        ----------
        where : tuple[int, int, str]
            The coordinate of site and direction to find environment.

        Returns
        -------
        Tensor | None
            Returns the environment. If the environment is missing, or there is no environment here, it returns None.
        """
        owner = self._owner
        l1, l2, direction = where
        if direction == "R":
            if 0 <= l1 < owner.L1 and 0 <= l2 < owner.L2 - 1:
                return owner._environment_h[l1][l2]
        elif direction == "L":
            l2 -= 1
            if 0 <= l1 < owner.L1 and 0 <= l2 < owner.L2 - 1:
                return owner._environment_h[l1][l2]
        elif direction == "D":
            if 0 <= l1 < owner.L1 - 1 and 0 <= l2 < owner.L2:
                return owner._environment_v[l1][l2]
        elif direction == "U":
            l1 -= 1
            if 0 <= l1 < owner.L1 - 1 and 0 <= l2 < owner.L2:
                return owner._environment_v[l1][l2]
        else:
            raise ValueError("Invalid direction")
        # out of lattice
        return None

    def __setitem__(self, where, value):
        """
        Set the environment by coordinate of site and direction.

        Parameters
        ----------
        where : tuple[int, int, str]
            The coordinate of site and direction to find environment.
        value : Tensor | None
            The environment tensor to set.
        """
        owner = self._owner
        l1, l2, direction = where
        if direction == "R":
            if 0 <= l1 < owner.L1 and 0 <= l2 < owner.L2 - 1:
                owner._environment_h[l1][l2] = value
                return
        elif direction == "L":
            l2 -= 1
            if 0 <= l1 < owner.L1 and 0 <= l2 < owner.L2 - 1:
                owner._environment_h[l1][l2] = value
                return
        elif direction == "D":
            if 0 <= l1 < owner.L1 - 1 and 0 <= l2 < owner.L2:
                owner._environment_v[l1][l2] = value
                return
        elif direction == "U":
            l1 -= 1
            if 0 <= l1 < owner.L1 - 1 and 0 <= l2 < owner.L2:
                owner._environment_v[l1][l2] = value
                return
        else:
            raise ValueError("Invalid direction")
        raise ValueError("Environment out of lattice")


class SimpleUpdateLattice(AbstractLattice):
    """
    The lattice used to do simple update.
    """

    __slots__ = ["_lattice", "_environment_v", "_environment_h", "_auxiliaries"]

    def __setstate__(self, state):
        for key, value in state.items():
            setattr(self, key, value)
        self._auxiliaries = None

    def __getstate__(self):
        state = {key: getattr(self, key) for key in _slotnames(self.__class__) if key != "_auxiliaries"}
        return state

    def __init__(self, abstract):
        """
        Create a simple update lattice from abstract lattice.

        Parameters
        ----------
        abstract : AbstractLattice
            The abstract lattice used to create simple update lattice.
        """
        super()._init_by_copy(abstract)

        self._lattice = [[self._construct_tensor(l1, l2) for l2 in range(self.L2)] for l1 in range(self.L1)]
        self._environment_h = [[None for l2 in range(self.L2 - 1)] for l1 in range(self.L1)]
        self._environment_v = [[None for l2 in range(self.L2)] for l1 in range(self.L1 - 1)]
        self._auxiliaries = None

    def __getitem__(self, l1l2):
        """
        Get the tensor at the given coordinate.

        Parameters
        ----------
        l1l2 : tuple[int, int]
            The coordinate.

        Returns
        -------
        Tensor
            The tensor at the given coordinate.
        """
        l1, l2 = l1l2
        return self._lattice[l1][l2]

    def __setitem__(self, l1l2, value):
        """
        Set the tensor at the given coordinate.

        Parameters
        ----------
        l1l2 : tuple[int, int]
            The coordinate.
        value : Tensor
            The tensor used to set.
        """
        l1, l2 = l1l2
        self._lattice[l1][l2] = value

    @property
    def environment(self):
        """
        Get the environment handler of this simple update lattice.

        Returns
        -------
        SimpleUpdateLatticeEnvironment
            The environment handler of this simple update lattice.
        """
        return SimpleUpdateLatticeEnvironment(self)

    def update(self, total_step, delta_tau, new_dimension):
        """
        Do simple update on the lattice.

        Parameters
        ----------
        total_step : int
            The total step number of the simple update.
        delta_tau : float
            The delta tau in the evolution operator.
        new_dimension : int
            The dimension cut used in svd of simple update.
        """
        updaters = []
        for positions, hamiltonian_term in self._hamiltonians.items():
            coordinates = []
            index_and_orbit = []
            for l1, l2, orbit in positions:
                if (l1, l2) not in coordinates:
                    coordinates.append((l1, l2))
                index = coordinates.index((l1, l2))
                index_and_orbit.append((index, orbit))

            site_number = len(positions)
            evolution_operator = (-delta_tau * hamiltonian_term).exponential(
                {(f"I{i}", f"O{i}") for i in range(site_number)}, step=8)

            updaters.append((coordinates, index_and_orbit, evolution_operator))
        for step in range(total_step):
            print(clear_line, f"Simple update, {total_step=}, {delta_tau=}, {new_dimension=}, {step=}", end="\r")
            for coordinates, index_and_orbit, evolution_operator in updaters:
                self._single_term_simple_update(coordinates, index_and_orbit, evolution_operator, new_dimension)
            for coordinates, index_and_orbit, evolution_operator in reversed(updaters):
                self._single_term_simple_update(coordinates, index_and_orbit, evolution_operator, new_dimension)
        print(clear_line, f"Simple update done, {total_step=}, {delta_tau=}, {new_dimension=}")
        self._update_virtual_bond()

    def _update_virtual_bond(self):
        """
        Update virtual bond after simple update is applied.
        """
        for l1 in range(self.L1):
            for l2 in range(self.L2):
                if l1 != self.L1 - 1:
                    self.virtual_bond[l1, l2, "D"] = self[l1, l2].edges("D")
                if l2 != self.L2 - 1:
                    self.virtual_bond[l1, l2, "R"] = self[l1, l2].edges("R")

    def _single_term_simple_update(self, coordinates, index_and_orbit, evolution_operator, new_dimension):
        """
        Dispatcher to do simple step single term simple update.

        Parameters
        ----------
        coordinates : list[tuple[int, int]]
            The coordinates of sites which need to be updated, there is no duplicated element in this list.
        index_and_orbit : list[tuple[int, int]]
            The list of the coordinates index and the orbit index of every hamiltonian edge.
        evolution_operator : Tensor
            $\exp^{-\Delta\tau H}$, used to update the state.
        new_dimension : int
            The dimension cut used in svd of simple update.
        """
        if len(coordinates) == 1:
            return self._single_term_simple_update_single_site(
                coordinates,
                index_and_orbit,
                evolution_operator,
                new_dimension,
            )
        if len(coordinates) == 2:
            coordinate_1, coordinate_2 = coordinates
            if coordinate_1[0] == coordinate_2[0] and abs(coordinate_1[1] - coordinate_2[1]) == 1:
                return self._single_term_simple_update_double_site_nearest_horizontal(
                    coordinates,
                    index_and_orbit,
                    evolution_operator,
                    new_dimension,
                )
            if coordinate_1[1] == coordinate_2[1] and abs(coordinate_1[0] - coordinate_2[0]) == 1:
                return self._single_term_simple_update_double_site_nearest_vertical(
                    coordinates,
                    index_and_orbit,
                    evolution_operator,
                    new_dimension,
                )
        raise NotImplementedError("Unsupported simple update style")

    def _single_term_simple_update_single_site(self, coordinates, index_and_orbit, evolution_operator, new_dimension):
        """
        See Also
        --------
        _single_term_simple_update
        """
        coordinate = coordinates[0]
        orbits = [orbit for index, orbit in index_and_orbit]
        self[coordinate] = self[coordinate].contract(evolution_operator, {
            (f"P{orbit}", f"I{body_index}") for body_index, orbit in enumerate(orbits)
        }).edge_rename({f"O{body_index}": f"P{orbit}" for body_index, orbit in enumerate(orbits)})

    def _single_term_simple_update_double_site_nearest_horizontal(self, coordinates, index_and_orbit,
                                                                  evolution_operator, new_dimension):
        """
        See Also
        --------
        _single_term_simple_update
        """
        coordinate_1, coordinate_2 = coordinates
        if coordinate_1[0] != coordinate_2[0]:
            raise RuntimeError("Wrong simple update dispatch")
        if coordinate_1[1] == coordinate_2[1] + 1:
            # Exchange two coordinate
            left_index_and_orbit = [
                (body_index, orbit) for body_index, [index, orbit] in enumerate(index_and_orbit) if index == 1
            ]
            right_index_and_orbit = [
                (body_index, orbit) for body_index, [index, orbit] in enumerate(index_and_orbit) if index == 0
            ]
            i, j = coordinate_2
        elif coordinate_1[1] + 1 == coordinate_2[1]:
            # Normal order
            left_index_and_orbit = [
                (body_index, orbit) for body_index, [index, orbit] in enumerate(index_and_orbit) if index == 0
            ]
            right_index_and_orbit = [
                (body_index, orbit) for body_index, [index, orbit] in enumerate(index_and_orbit) if index == 1
            ]
            i, j = coordinate_1
        else:
            raise RuntimeError("Wrong simple update dispatch")
        body = len(index_and_orbit)
        left = self[i, j]
        left = self._try_multiple(left, i, j, "L")
        left = self._try_multiple(left, i, j, "U")
        left = self._try_multiple(left, i, j, "D")
        left = self._try_multiple(left, i, j, "R")
        right = self[i, j + 1]
        right = self._try_multiple(right, i, j + 1, "U")
        right = self._try_multiple(right, i, j + 1, "D")
        right = self._try_multiple(right, i, j + 1, "R")
        left_q, left_r = left.qr("r", {*(f"P{orbit}" for body_index, orbit in left_index_and_orbit), "R"}, "R", "L")
        right_q, right_r = right.qr("r", {*(f"P{orbit}" for body_index, orbit in right_index_and_orbit), "L"}, "L", "R")
        u, s, v = left_r.edge_rename({
            f"P{orbit}": f"P{body_index}" for body_index, orbit in left_index_and_orbit
        }).contract(right_r.edge_rename({f"P{orbit}": f"P{body_index}" for body_index, orbit in right_index_and_orbit}),
                    {("R", "L")}).contract(evolution_operator,
                                           {(f"P{body_index}", f"I{body_index}") for body_index in range(body)}).svd(
                                               {*(f"O{body_index}" for body_index, orbit in left_index_and_orbit), "L"},
                                               "R", "L", "L", "R", new_dimension)
        s /= s.norm_2()
        self.environment[i, j, "R"] = s
        left_q = self._try_multiple(left_q, i, j, "L", True)
        left_q = self._try_multiple(left_q, i, j, "U", True)
        left_q = self._try_multiple(left_q, i, j, "D", True)
        u = u.contract(left_q, {("L", "R")}).edge_rename(
            {f"O{body_index}": f"P{orbit}" for body_index, orbit in left_index_and_orbit})
        self[i, j] = u
        right_q = self._try_multiple(right_q, i, j + 1, "U", True)
        right_q = self._try_multiple(right_q, i, j + 1, "D", True)
        right_q = self._try_multiple(right_q, i, j + 1, "R", True)
        v = v.contract(right_q, {("R", "L")}).edge_rename(
            {f"O{body_index}": f"P{orbit}" for body_index, orbit in right_index_and_orbit})
        self[i, j + 1] = v

    def _single_term_simple_update_double_site_nearest_vertical(self, coordinates, index_and_orbit, evolution_operator,
                                                                new_dimension):
        """
        See Also
        --------
        _single_term_simple_update
        """
        coordinate_1, coordinate_2 = coordinates
        if coordinate_1[1] != coordinate_2[1]:
            raise RuntimeError("Wrong simple update dispatch")
        if coordinate_1[0] == coordinate_2[0] + 1:
            # Exchange two coordinate
            up_index_and_orbit = [
                (body_index, orbit) for body_index, [index, orbit] in enumerate(index_and_orbit) if index == 1
            ]
            down_index_and_orbit = [
                (body_index, orbit) for body_index, [index, orbit] in enumerate(index_and_orbit) if index == 0
            ]
            i, j = coordinate_2
        elif coordinate_1[0] + 1 == coordinate_2[0]:
            # Normal order
            up_index_and_orbit = [
                (body_index, orbit) for body_index, [index, orbit] in enumerate(index_and_orbit) if index == 0
            ]
            down_index_and_orbit = [
                (body_index, orbit) for body_index, [index, orbit] in enumerate(index_and_orbit) if index == 1
            ]
            i, j = coordinate_1
        else:
            raise RuntimeError("Wrong simple update dispatch")
        body = len(index_and_orbit)
        up = self[i, j]
        up = self._try_multiple(up, i, j, "L")
        up = self._try_multiple(up, i, j, "U")
        up = self._try_multiple(up, i, j, "D")
        up = self._try_multiple(up, i, j, "R")
        down = self[i + 1, j]
        down = self._try_multiple(down, i + 1, j, "L")
        down = self._try_multiple(down, i + 1, j, "D")
        down = self._try_multiple(down, i + 1, j, "R")
        up_q, up_r = up.qr("r", {*(f"P{orbit}" for body_index, orbit in up_index_and_orbit), "D"}, "D", "U")
        down_q, down_r = down.qr("r", {*(f"P{orbit}" for body_index, orbit in down_index_and_orbit), "U"}, "U", "D")
        u, s, v = up_r.edge_rename({
            f"P{orbit}": f"P{body_index}" for body_index, orbit in up_index_and_orbit
        }).contract(down_r.edge_rename({f"P{orbit}": f"P{body_index}" for body_index, orbit in down_index_and_orbit}),
                    {("D", "U")}).contract(evolution_operator,
                                           {(f"P{body_index}", f"I{body_index}") for body_index in range(body)}).svd(
                                               {*(f"O{body_index}" for body_index, orbit in up_index_and_orbit), "U"},
                                               "D", "U", "U", "D", new_dimension)
        s /= s.norm_2()
        self.environment[i, j, "D"] = s
        up_q = self._try_multiple(up_q, i, j, "L", True)
        up_q = self._try_multiple(up_q, i, j, "U", True)
        up_q = self._try_multiple(up_q, i, j, "R", True)
        u = u.contract(up_q, {("U", "D")}).edge_rename(
            {f"O{body_index}": f"P{orbit}" for body_index, orbit in up_index_and_orbit})
        self[i, j] = u
        down_q = self._try_multiple(down_q, i + 1, j, "L", True)
        down_q = self._try_multiple(down_q, i + 1, j, "D", True)
        down_q = self._try_multiple(down_q, i + 1, j, "R", True)
        v = v.contract(down_q, {("D", "U")}).edge_rename(
            {f"O{body_index}": f"P{orbit}" for body_index, orbit in down_index_and_orbit})
        self[i + 1, j] = v

    def _try_multiple(self, tensor, i, j, direction, division=False):
        """
        Try to multiple environment to a given tensor.

        Parameters
        ----------
        tensor : Tensor
            The input tensor.
        i, j : int
            The site coordinate to find environment.
        direction : str
            The direction of the environment from the site.
        division : bool, default=False
            Divide the environment instead of multiple it.

        Returns
        -------
        Tensor
            The result tensor multipled with environment, if the environment does not exist, return the origin tensor.
        """
        environment_tensor = self.environment[i, j, direction]
        if environment_tensor is not None:
            if division:
                environment_tensor = environment_tensor.map(lambda x: 0 if x == 0 else 1. / x)
            if direction == "L":
                tensor = tensor.contract(environment_tensor, {("L", "R")})
            if direction == "R":
                tensor = tensor.contract(environment_tensor, {("R", "L")})
            if direction == "U":
                tensor = tensor.contract(environment_tensor, {("U", "D")})
            if direction == "D":
                tensor = tensor.contract(environment_tensor, {("D", "U")})
        return tensor

    def initialize_auxiliaries(self, cut_dimension):
        """
        Initialize auxiliares.

        Parameters
        ----------
        cut_dimension : int
            The cut dimension when calculating auxiliary tensors.
        """
        self._auxiliaries = DoubleLayerAuxiliaries(self.L1, self.L2, cut_dimension, True, self.Tensor)
        for l1 in range(self.L1):
            for l2 in range(self.L2):
                this_site = self[l1, l2]
                this_site = self._try_multiple(this_site, l1, l2, "L")
                this_site = self._try_multiple(this_site, l1, l2, "U")
                self._auxiliaries[l1, l2, "N"] = this_site
                self._auxiliaries[l1, l2, "C"] = this_site.conjugate()

    def observe(self, positions, observer):
        """
        Observe the state. Need to initialize auxiliaries before calling `observe`.

        Parameters
        ----------
        positions : tuple[tuple[int, int, int], ...]
            The site and their orbits used to observe.
        observer : Tensor
            The operator used to observe.

        Returns
        -------
        float
            The observer result, which is normalized.
        """
        if self._auxiliaries is None:
            raise RuntimeError("Need to initialize auxiliary before call observe")
        body = observer.rank // 2
        if body == 0:
            return float(1)
        rho = self._auxiliaries.hole(positions)
        psipsi = rho.trace({(f"O{i}", f"I{i}") for i in range(body)})
        psiHpsi = rho.contract(observer,
                               {*((f"O{i}", f"I{i}") for i in range(body)), *((f"I{i}", f"O{i}") for i in range(body))})
        return float(psiHpsi) / float(psipsi)

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
        return energy / self.site_number
