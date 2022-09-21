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
import numpy as np
from .auxiliaries import DoubleLayerAuxiliaries
from .abstract_lattice import AbstractLattice
from .common_toolkit import show, showln, mpi_comm, mpi_rank, mpi_size


class SimpleUpdateLatticeEnvironment:
    """
    Environment handler for simple update lattice.
    """

    __slots__ = ["owner"]

    def __init__(self, owner):
        """
        Create environment handler

        Parameters
        ----------
        owner : SimpleUpdateLattice
            The owner of this handler.
        """
        self.owner: SimpleUpdateLattice = owner

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
        l1, l2, direction = where
        if direction == "R":
            if 0 <= l1 < self.owner.L1 and 0 <= l2 < self.owner.L2 - 1:
                return self.owner._environment_h[l1][l2]
        elif direction == "L":
            l2 -= 1
            if 0 <= l1 < self.owner.L1 and 0 <= l2 < self.owner.L2 - 1:
                return self.owner._environment_h[l1][l2]
        elif direction == "D":
            if 0 <= l1 < self.owner.L1 - 1 and 0 <= l2 < self.owner.L2:
                return self.owner._environment_v[l1][l2]
        elif direction == "U":
            l1 -= 1
            if 0 <= l1 < self.owner.L1 - 1 and 0 <= l2 < self.owner.L2:
                return self.owner._environment_v[l1][l2]
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
        l1, l2, direction = where
        if direction == "R":
            if 0 <= l1 < self.owner.L1 and 0 <= l2 < self.owner.L2 - 1:
                self.owner._environment_h[l1][l2] = value
                return
        elif direction == "L":
            l2 -= 1
            if 0 <= l1 < self.owner.L1 and 0 <= l2 < self.owner.L2 - 1:
                self.owner._environment_h[l1][l2] = value
                return
        elif direction == "D":
            if 0 <= l1 < self.owner.L1 - 1 and 0 <= l2 < self.owner.L2:
                self.owner._environment_v[l1][l2] = value
                return
        elif direction == "U":
            l1 -= 1
            if 0 <= l1 < self.owner.L1 - 1 and 0 <= l2 < self.owner.L2:
                self.owner._environment_v[l1][l2] = value
                return
        else:
            raise ValueError("Invalid direction")
        raise ValueError("Environment out of lattice")


class SimpleUpdateLattice(AbstractLattice):
    """
    The lattice used to do simple update.
    """

    __slots__ = ["_lattice", "_environment_v", "_environment_h", "_auxiliaries"]

    def _v1_to_v2_multiple(self):
        """
        Migrate data from version 1 to version 2.

        In version 1, environment and site tensor is what is be like.
        But in version 2, site tensor store the product of the original site tensor and the 4 environment tensors
        surrounding it, for better performance.

        So multiple the 4 environment tensors into the site tensor here.
        """
        for l1 in range(self.L1):
            for l2 in range(self.L2):
                this = self[l1, l2]
                this = self._try_multiple(this, l1, l2, "L")
                this = self._try_multiple(this, l1, l2, "R")
                this = self._try_multiple(this, l1, l2, "U")
                this = self._try_multiple(this, l1, l2, "D")
                self[l1, l2] = this

    def __setstate__(self, state):
        call_at_last = []
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
            call_at_last.append(self._v1_to_v2_multiple)
        # version 2 to version 3
        if state["data_version"] == 2:
            self._v2_to_v3_rename(state)
            state["data_version"] = 3
        # version 3 to version 4
        if state["data_version"] == 3:
            state["data_version"] = 4
        # setstate
        state["_auxiliaries"] = None
        for key, value in state.items():
            setattr(self, key, value)
        for call in call_at_last:
            call()

    def __getstate__(self):
        # getstate
        state = {key: getattr(self, key) for key in _slotnames(self.__class__)}
        del state["_auxiliaries"]  # aux is lazy graph, should not be dumped.
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

        # The data storage of site tensor, access it directly by lattice[l1, l2] instead
        self._lattice = [[self._construct_tensor(l1, l2) for l2 in range(self.L2)] for l1 in range(self.L1)]
        # The data storage of environment tensor, access it by lattice.environment[l1, l2, direction] instead.
        self._environment_h = [[None for l2 in range(self.L2 - 1)] for l1 in range(self.L1)]
        self._environment_v = [[None for l2 in range(self.L2)] for l1 in range(self.L1 - 1)]
        # The double layer auxiliaries, only used by internal method when observing.
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
        new_dimension : int | float
            The dimension cut used in svd of simple update, or the amplitude of dimension expandance.
        """

        # Create updater first
        # updater U_i = exp(- delta_tau H_i)
        # At this step, get the coordinates of every hamiltonian term instead of original knowning specific orbit only.
        updaters = []
        for positions, hamiltonian_term in self._hamiltonians.items():
            # coordinates is the site list of what this hamiltonian term effects on.
            # it may be less than hamiltonian rank
            coordinates = []
            # it store each rank of hamiltonian effect on which coordinates by recording its index in the coordinates
            # list and its orbit.
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

        # Split updaters into bundles, to scatter jobs into every mpi process.
        updaters_bundles = []
        # To get the max parallel size for simple update
        max_index = 0
        while len(updaters) != 0:
            this_bundle = []  # A bundle contains independent hamiltonian, which will be added into updaters_bundles.
            coordinates_pool = set()  # The coordinates set of hamiltonians which have already insert into this_bundle.
            coordinates_map = {}  # Record which mpi rank will modified the site tensor at some coordinate.
            remained_updaters = []  # The remained updaters to be added into bundles at later iterations.
            index = 0  # Record the mpi rank to run this updater in this bundle.
            for coordinates, index_and_orbit, evolution_operator in updaters:
                if len([coordinate for coordinate in coordinates if coordinate in coordinates_pool]) == 0:
                    # This hamiltonian term is independent to and term in this_bundle, insert it into this bundle.
                    this_bundle.append((index, coordinates, index_and_orbit, evolution_operator))
                    # Add coordinate of this term into coordinates_pool
                    # And associate this coordinates to specific mpi rank
                    for coordinate in coordinates:
                        coordinates_pool.add(coordinate)
                        coordinates_map[coordinate] = index
                    index += 1
                else:
                    # Cannot insert it into this bundle, add it into remained_updaters for later iterations.
                    remained_updaters.append((coordinates, index_and_orbit, evolution_operator))
            # The max mpi rank is the max parallel size.
            if index > max_index:
                max_index = index
            # Update updaters list for the next iterations
            updaters = remained_updaters
            # Append this bundle to updaters_bundles.
            updaters_bundles.append((this_bundle, coordinates_map))
        showln(f"Simple update max parallel size is {max_index}")

        # Run simple update
        for step in range(total_step):
            show(f"Simple update, {total_step=}, {delta_tau=}, {new_dimension=}, {step=}")
            # Trotter expansion
            # run updater bundle by bundle.
            for bundle, coordinates_map in updaters_bundles:
                # In a single bundle, put each updater to its mpi rank process.
                for index, coordinates, index_and_orbit, evolution_operator in bundle:
                    if index % mpi_size == mpi_rank:
                        self._single_term_simple_update(coordinates, index_and_orbit, evolution_operator, new_dimension)
                # Bcast what modified
                self._bcast_by_map(coordinates_map)
            for bundle, coordinates_map in reversed(updaters_bundles):
                # In a single bundle, put each updater to its mpi rank process.
                for index, coordinates, index_and_orbit, evolution_operator in bundle:
                    if index % mpi_size == mpi_rank:
                        self._single_term_simple_update(coordinates, index_and_orbit, evolution_operator, new_dimension)
                # Bcast what modified
                self._bcast_by_map(coordinates_map)
        showln(f"Simple update done, {total_step=}, {delta_tau=}, {new_dimension=}")
        # After simple update, virtual bond changed, so update it.
        self._update_virtual_bond()

    def _bcast_by_map(self, coordinates_map):
        """
        Bcast tensors according to a map recording which rank changed which tensor. The environment between two tensor
        changed by the same rank is considered changed by that rank too.

        Parameters
        ----------
        coordinates_map : dict[tuple[int, int], int]
            A map recording which rank changed which tensor.
        """
        for l1 in range(self.L1):
            for l2 in range(self.L2):
                if (l1, l2) in coordinates_map:
                    owner_rank = coordinates_map[l1, l2] % mpi_size
                    self[l1, l2] = mpi_comm.bcast(self[l1, l2], root=owner_rank)
                    # If the nearest site is also belong to this rank, then the environment between them is also belong to it.
                    if (l1 - 1, l2) in coordinates_map and coordinates_map[l1 - 1, l2] % mpi_size == owner_rank:
                        self.environment[l1, l2, "U"] = mpi_comm.bcast(self.environment[l1, l2, "U"], root=owner_rank)
                    if (l1, l2 - 1) in coordinates_map and coordinates_map[l1, l2 - 1] % mpi_size == owner_rank:
                        self.environment[l1, l2, "L"] = mpi_comm.bcast(self.environment[l1, l2, "L"], root=owner_rank)

    def _update_virtual_bond(self):
        """
        Update virtual bond after simple update is applied.
        """
        for l1 in range(self.L1):
            for l2 in range(self.L2):
                # Update half of virtual bond, another part will be updated automatically.
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
        new_dimension : int | float
            The dimension cut used in svd of simple update, or the amplitude of dimension expandance.
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
        self[coordinate] = (
            self[coordinate]  #
            .contract(evolution_operator,
                      {(f"P{orbit}", f"I{body_index}") for body_index, orbit in enumerate(orbits)})  #
            .edge_rename({f"O{body_index}": f"P{orbit}" for body_index, orbit in enumerate(orbits)}))

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
        right = self[i, j + 1]
        right = self._try_multiple(right, i, j + 1, "L", division=True)
        original_dimension = left.edges("R").dimension
        if isinstance(new_dimension, float):
            new_dimension = round(original_dimension * new_dimension)
        left_q, left_r = left.qr("r", {*(f"P{orbit}" for body_index, orbit in left_index_and_orbit), "R"}, "R", "L")
        right_q, right_r = right.qr("r", {*(f"P{orbit}" for body_index, orbit in right_index_and_orbit), "L"}, "L", "R")
        u, s, v = (
            left_r  #
            .edge_rename({f"P{orbit}": f"P{body_index}" for body_index, orbit in left_index_and_orbit})  #
            .contract(
                right_r.edge_rename({f"P{orbit}": f"P{body_index}" for body_index, orbit in right_index_and_orbit}),
                {("R", "L")})  #
            .contract(evolution_operator, {(f"P{body_index}", f"I{body_index}") for body_index in range(body)})  #
            .svd({*(f"O{body_index}" for body_index, orbit in left_index_and_orbit), "L"}, "R", "L", "L", "R",
                 new_dimension))
        s /= s.norm_2()
        self.environment[i, j, "R"] = s
        u = self._try_multiple(u, i, j, "R")
        u = (
            u  #
            .contract(left_q, {("L", "R")})  #
            .edge_rename({f"O{body_index}": f"P{orbit}" for body_index, orbit in left_index_and_orbit}))
        self[i, j] = u
        v = self._try_multiple(v, i, j + 1, "L")
        v = (
            v  #
            .contract(right_q, {("R", "L")})  #
            .edge_rename({f"O{body_index}": f"P{orbit}" for body_index, orbit in right_index_and_orbit}))
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
        down = self[i + 1, j]
        down = self._try_multiple(down, i + 1, j, "U", division=True)
        original_dimension = up.edges("D").dimension
        if isinstance(new_dimension, float):
            new_dimension = round(original_dimension * new_dimension)
        up_q, up_r = up.qr("r", {*(f"P{orbit}" for body_index, orbit in up_index_and_orbit), "D"}, "D", "U")
        down_q, down_r = down.qr("r", {*(f"P{orbit}" for body_index, orbit in down_index_and_orbit), "U"}, "U", "D")
        u, s, v = (
            up_r  #
            .edge_rename({f"P{orbit}": f"P{body_index}" for body_index, orbit in up_index_and_orbit})  #
            .contract(down_r.edge_rename({f"P{orbit}": f"P{body_index}" for body_index, orbit in down_index_and_orbit}),
                      {("D", "U")})  #
            .contract(evolution_operator, {(f"P{body_index}", f"I{body_index}") for body_index in range(body)})  #
            .svd({*(f"O{body_index}" for body_index, orbit in up_index_and_orbit), "U"}, "D", "U", "U", "D",
                 new_dimension))
        s /= s.norm_2()
        self.environment[i, j, "D"] = s
        u = self._try_multiple(u, i, j, "D")
        u = (
            u  #
            .contract(up_q, {("U", "D")})  #
            .edge_rename({f"O{body_index}": f"P{orbit}" for body_index, orbit in up_index_and_orbit}))
        self[i, j] = u
        v = self._try_multiple(v, i + 1, j, "U")
        v = (
            v  #
            .contract(down_q, {("D", "U")})  #
            .edge_rename({f"O{body_index}": f"P{orbit}" for body_index, orbit in down_index_and_orbit}))
        self[i + 1, j] = v

    def _try_multiple(self, tensor, i, j, direction, *, division=False, square_root=False):
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
        square_root : bool, default=False
            Multiple or divide the square root instead of itself.

        Returns
        -------
        Tensor
            The result tensor multipled with environment, if the environment does not exist, return the origin tensor.
        """
        environment_tensor = self.environment[i, j, direction]
        if environment_tensor is not None:
            if division:
                environment_tensor = environment_tensor.map(lambda x: 0 if x == 0 else 1. / x)
            if square_root:
                # Cannot calculate sqrt trivially, prepare an identity matrix and get the square root of absolute value
                # first. And then calculate time it or devide it with idenity
                environment_tensor = environment_tensor.copy()  # copy it, since there is inplace operator later.
                identity = environment_tensor.same_shape().identity({tuple(environment_tensor.names)})
                delta = np.sqrt(np.abs(environment_tensor.storage))
                delta[delta == 0] = 1
                environment_tensor.storage /= delta
                identity.storage *= delta
                # Delivery former identity and former environment tensor to four direction.
                if direction in ("D", "R"):
                    environment_tensor = identity
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
                # Every site tensor also contains its environment, so divide half of its environment.
                this_site = self._try_multiple(this_site, l1, l2, "L", division=True)
                this_site = self._try_multiple(this_site, l1, l2, "U", division=True)
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
        # Cannot calculate psipsi together at last, because auxiliaries will normalize the tensor.
        # That is why it is different to observe function for exact state.
        psipsi = rho.trace({(f"O{i}", f"I{i}") for i in range(body)})
        psiHpsi = (
            rho  #
            .contract(observer,
                      {*((f"O{i}", f"I{i}") for i in range(body)), *((f"I{i}", f"O{i}") for i in range(body))}))
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
