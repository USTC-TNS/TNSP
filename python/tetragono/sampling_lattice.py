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
from .auxiliaries import SingleLayerAuxiliaries
from .abstract_lattice import AbstractLattice


class Configuration(SingleLayerAuxiliaries):
    """
    Configuration system for square sampling lattice.
    """

    __slots__ = ["_owner", "_configuration", "_holes"]

    def copy(self, cp=None):
        """
        Copy the configuration system.

        Parameters
        ----------
        cp : Copy, default=None
            The copy object used to copy the internal lazy node graph.

        Returns
        -------
        Configuration
            The new configuration system.
        """
        result = super().copy(cp=cp)

        result._owner = self._owner
        result._configuration = [[{
            orbit: self._configuration[l1][l2][orbit] for orbit, edge in self._owner.physics_edges[l1, l2].items()
        } for l2 in range(self._owner.L2)] for l1 in range(self._owner.L1)]
        result._holes = self._holes
        return result

    def __init__(self, owner, cut_dimension):
        """
        Create configuration system for the given sampling lattice.

        Parameters
        ----------
        owner : SamplingLattice
            The sampling lattice owning this configuration system.
        cut_dimension : int
            The cut dimension in single layer auxiliaries.
        """
        super().__init__(owner.L1, owner.L2, cut_dimension, False, owner.Tensor)
        self._owner = owner
        # EdgePoint = tuple[self.Symmetry, int]
        self._configuration = [[{orbit: None
                                 for orbit, edge in self._owner.physics_edges[l1, l2].items()}
                                for l2 in range(self._owner.L2)]
                               for l1 in range(self._owner.L1)]
        for l1 in range(owner.L1):
            for l2 in range(owner.L2):
                if len(self._owner.physics_edges[l1, l2]) == 0:
                    super().__setitem__((l1, l2), self._owner[l1, l2])
        self._holes = None

    def site_valid(self, l1, l2):
        """
        Check if specific site have valid configuration

        Parameters
        ----------
        l1, l2 : int
            The coordinate of the specific site.

        Returns
        -------
        bool
            The validity of this single site configuration.
        """
        for orbit in self._owner.physics_edges[l1, l2]:
            if self._configuration[l1][l2][orbit] is None:
                return False
        return True

    def valid(self):
        """
        Check if all site have valid configuration.

        Returns
        -------
        bool
            The validity of this configuration system.
        """
        for l1 in range(self._owner.L1):
            for l2 in range(self._owner.L2):
                if not self.site_valid(l1, l2):
                    return False
        return True

    def __getitem__(self, l1l2o):
        """
        Get the configuration of the specific site.

        Parameters
        ----------
        l1l2o : tuple[int, int, int]
            The coordinate and orbit index of the site.

        Returns
        -------
        EdgePoint | None
            The configuration of the specific site.
        """
        l1, l2, orbit = l1l2o
        return self._configuration[l1][l2][orbit]

    def __setitem__(self, l1l2o, value):
        """
        Set the configuration of the specific site.

        Parameters
        ----------
        l1l2o : tuple[int, int, int]
            The coordinate and orbit index of the site.
        value : ?EdgePoint
            The configuration of this site.
        """
        l1, l2, orbit = l1l2o
        if value is None:
            self._configuration[l1][l2][orbit] = None
            super().__setitem__((l1, l2), None)
            self._holes = None
            return
        this_configuration = self._construct_edge_point(value)
        if this_configuration == self._configuration[l1][l2][orbit]:
            changed = False
        else:
            self._configuration[l1][l2][orbit] = this_configuration
            changed = True
        if self._lattice[l1][l2]() is None or changed:
            if self.site_valid(l1, l2):
                shrinked_site = self._shrink_configuration((l1, l2), self._configuration[l1][l2])
                super().__setitem__((l1, l2), shrinked_site)
                self._holes = None

    def __delitem__(self, l1l2o):
        """
        Clear the configuration of the specific site.

        Parameters
        ----------
        l1l2o : tuple[int, int, int]
            The coordinate and orbit index of the site.
        """
        self.__setitem__(l1l2o, None)

    def replace(self, replacement, *, hint=None):
        """
        Calculate $\langle s\psi\rangle$ with several $s$ replaced.

        Parameters
        ----------
        replacement : dict[tuple[int, int, int], ?EdgePoint]
            Replacement plan to modify $s$.
        hint : Any, default=None
            Hint passed to base class replace

        Returns
        -------
        Tensor | None
            $\langle s\psi\rangle$ with several $s$ replaced. If replace style is not implemented yet, None will be
        returned.
        """
        grouped_replacement = {}  # dict[tuple[int, int], dict[int, EdgePoint]]
        for [l1, l2, orbit], edge_point in replacement.items():
            l1l2 = l1, l2
            if l1l2 not in grouped_replacement:
                grouped_replacement[l1l2] = {}
            grouped_replacement[l1l2][orbit] = edge_point

        base_replacement = {}  # dict[tuple[int, int], Tensor]
        for l1l2, site_replacement in grouped_replacement.items():
            l1, l2 = l1l2
            changed = False
            for orbit, configuration in self._configuration[l1][l2].items():
                if orbit not in site_replacement:
                    site_replacement[orbit] = configuration
                else:
                    if site_replacement[orbit] != configuration:
                        changed = True
            if changed:
                base_replacement[l1l2] = self._shrink_configuration(l1l2, site_replacement)
        return super().replace(base_replacement, hint=hint)

    def _construct_edge_point(self, value):
        """
        Construct edge point from something that can be used to construct an edge point.

        Parameters
        ----------
        value : ?EdgePoint
            Edge point or something that can be used to construct a edge point.

        Returns
        -------
        EdgePoint
            The result edge point object.
        """
        if not isinstance(value, tuple):
            symmetry = self._owner.Symmetry()  # work for NoSymmetry
            index = value
        else:
            symmetry, index = value
        symmetry = self._owner._construct_symmetry(symmetry)
        return (symmetry, index)

    def _get_shrinker(self, l1l2, configuration):
        """
        Get shrinker tensor for the given coordinate site, using the given configuration map.

        Parameters
        ----------
        l1l2 : tuple[int, int]
            The coordinate of the site.
        configuration : dict[int, EdgePoint]
            The given configuration for this site, mapping orbit to edge point.

        Yields
        ------
        tuple[int, Tensor]
            The orbit index and shrinker tensor, shrinker tensor name is "P" and "Q", where edge "P" is wider one.
        """
        l1, l2 = l1l2
        for orbit, edge in self._owner.physics_edges[l1, l2].items():
            symmetry, index = configuration[orbit]
            # P side is dimension - 1 edge
            # Q side is connected to lattice
            shrinker = self.Tensor(["P", "Q"], [[(symmetry, 1)], edge.conjugated()]).zero()
            shrinker[{"Q": (-symmetry, index), "P": (symmetry, 0)}] = 1
            yield orbit, shrinker

    def _shrink_configuration(self, l1l2, configuration):
        """
        Shrink all configuration of given coordinate point.

        Parameters
        ----------
        l1l2 : tuple[int, int]
            The coordinate of the site.
        configuration : dict[int, EdgePoint]
            The given configuration for this site, mapping orbit to edge point.

        Returns
        -------
        Tensor
            The shrinked result tensor
        """
        l1, l2 = l1l2
        tensor = self._owner[l1l2]
        for orbit, shrinker in self._get_shrinker(l1l2, configuration):
            tensor = tensor.contract(shrinker.edge_rename({"P": f"P_{l1}_{l2}_{orbit}"}), {(f"P{orbit}", "Q")})
        return tensor

    def refresh_site(self, l1l2o):
        """
        Refresh the single site configuration, need to be called after lattice tensor changed.

        Parameters
        ----------
        l1l2o : tuple[int, int, int]
            The coordinate and orbit index of the site.
        """
        configuration = self[l1l2o]
        del self[l1l2o]
        self[l1l2o] = configuration

    def refresh_all(self):
        """
        Refresh the configuration of all sites, need to be called after lattice tensor changed.
        """
        for l1 in range(self._owner.L1):
            for l2 in range(self._owner.L2):
                for orbit in self._owner.physics_edges[l1, l2]:
                    self.refresh_site((l1, l2, orbit))

    def holes(self):
        """
        Get the lattice holes of this configuration.

        Returns
        -------
        list[list[Tensor]]
            The holes of this configuration.
        """
        if self._holes is None:
            # Prepare
            ws = self.hole(())
            inv_ws_conj = ws / (ws.norm_2()**2)
            inv_ws = inv_ws_conj.conjugate()
            all_name = {("T", "T")} | {(f"P_{l1}_{l2}_{orbit}", f"P_{l1}_{l2}_{orbit}") for l1 in range(self._owner.L1)
                                       for l2 in range(self._owner.L2)
                                       for orbit, edge in self._owner.physics_edges[l1, l2].items()}

            # Calculate
            holes = [[None for l2 in range(self._owner.L2)] for l1 in range(self._owner.L1)]
            # \frac{\partial\langle s|\psi\rangle}{\partial x_i} / \langle s|\psi\rangle
            for l1 in range(self._owner.L1):
                for l2 in range(self._owner.L2):
                    contract_name = all_name.copy()
                    for orbit in self._owner.physics_edges[l1, l2]:
                        contract_name.remove((f"P_{l1}_{l2}_{orbit}", f"P_{l1}_{l2}_{orbit}"))
                    if l1 == l2 == 0:
                        contract_name.remove(("T", "T"))
                    hole = self.hole(((l1, l2),)).contract(inv_ws, contract_name)
                    hole = hole.edge_rename({
                        "L0": "R",
                        "R0": "L",
                        "U0": "D",
                        "D0": "U"
                    } | {
                        f"P_{l1}_{l2}_{orbit}": f"P{orbit}"
                        for orbit, edge in self._owner.physics_edges[l1, l2].items()
                    })

                    for orbit, shrinker in self._get_shrinker((l1, l2), self._configuration[l1][l2]):
                        hole = hole.contract(shrinker, {(f"P{orbit}", "P")}).edge_rename({"Q": f"P{orbit}"})

                    holes[l1][l2] = hole
            self._holes = holes
        return self._holes


class TailDictTree:
    __slots__ = ["_pool"]

    def __init__(self):
        self._pool = {}

    def __setitem__(self, key, value):
        key = list(key)
        current = self._pool
        while len(key) != 1:
            node = key.pop()
            if node not in current:
                current[node] = {}
            current = current[node]
        node = key.pop()
        current[node] = value

    def __getitem__(self, key):
        key = list(key)
        current = self._pool
        while len(key) != 0:
            node = key.pop()
            current = current[node]
        return current

    def nearest(self, key):
        key = list(key)
        current = self._pool
        while len(key) != 0:
            node = key.pop()
            if node in current:
                current = current[node]
            else:
                current = next(iter(current.values()))
        return current

    def __contains__(self, key):
        key = list(key)
        current = self._pool
        while len(key) != 0:
            node = key.pop()
            if node not in current:
                return False
            current = current[node]
        return True


class ConfigurationPool:
    """
    Configuration pool for one sampling lattice and multiple configuration.
    """

    __slots__ = ["_owner", "_pool"]

    def __init__(self, owner):
        """
        Create configuration system pool for the given sampling lattice.

        Parameters
        ----------
        owner : SamplingLattice
            The sampling lattice owning this configuration pool.
        """
        self._owner = owner
        self._pool = TailDictTree()

    def wss(self, configuration, replacement):
        """
        Calculate wss of the configuration with the given replacement.

        Parameters
        ----------
        configuration : Configuration
            The given configuration.
        replacement : dict[tuple[int, int, int], ?EdgePoint]
            Replacement plan to modify configuration.

        Tensor
            $\langle s\psi\rangle$ with several $s$ replaced.
        """
        wss = configuration.replace(replacement)
        if wss is not None:
            return wss

        config = self._get_config(configuration)
        config = self._replace_config(config, replacement)
        if config in self._pool:
            return self._pool[config].hole(())

        nearest = self._nearest_configuration(config)
        wss = nearest.replace(replacement)
        if wss is not None:
            return wss
        return self(config, hint=nearest).hole(())

    def _replace_config(self, config, replacement):
        """
        Get the replaced config.

        Parameters
        ----------
        config : tuple[EdgePoint, ...]
            The input config tuple.
        replacement : dict[tuple[int, int, int], ?EdgePoint]
            Replacement plan to modify configuration.

        Returns
        -------
        tuple[EdgePoint, ...]
            The result config tuple.
        """
        config = list(config)
        index = 0
        for l1 in range(self._owner.L1):
            for l2 in range(self._owner.L2):
                for orbit, _ in self._owner.physics_edges[l1, l2].items():
                    if (l1, l2, orbit) in replacement:
                        config[index] = replacement[l1, l2, orbit]
                    index += 1
        return tuple(config)

    def add(self, configuration):
        """
        Add the configuration into pool, the configuration is not copied, it is inserted into a dict directly.

        Parameters
        ----------
        configuration : Configuration
            The configuration to be added.

        Returns
        -------
        Configuration
            If the configuration exists already, return the former configuration, otherwise return the input
        configuration.
        """
        config = self._get_config(configuration)
        if config not in self._pool:
            self._pool[config] = configuration
        return self._pool[config]

    def _get_config(self, configuration):
        """
        Get the key used for hash from the given configuration.

        Parameters
        ----------
        configuration : Configuration
            the configuration object.

        Returns
        -------
        tuple[EdgePoint, ...]
            the config tuple.
        """
        config = []
        for l1 in range(self._owner.L1):
            for l2 in range(self._owner.L2):
                for orbit, _ in self._owner.physics_edges[l1, l2].items():
                    config.append(configuration[l1, l2, orbit])
        return tuple(config)

    def _nearest_configuration(self, config):
        """
        Get the nearest configuration to the given config.

        Parameters
        ----------
        config : tuple[EdgePoint, ...]
            the config tuple.

        Returns
        -------
        Configuration
            The configuration which is the nearest to the given config tuple.
        """
        return self._pool.nearest(config)

    def _calculate_configuration(self, config, *, hint=None):
        """
        Calculate the configuration from the given config.

        Parameters
        ----------
        config : tuple[EdgePoint, ...]
            the config tuple.
        hint : Configuration, optional
            A similar configuration used to copy

        Returns
        -------
            The result configuration of the given config tuple.
        """
        if hint is not None:
            configuration = hint.copy()
        else:
            configuration = self._nearest_configuration(config).copy()
        index = 0
        for l1 in range(self._owner.L1):
            for l2 in range(self._owner.L2):
                for orbit, _ in self._owner.physics_edges[l1, l2].items():
                    # If they are the same, auxiliaries tensor will not be refreshed
                    configuration[l1, l2, orbit] = config[index]
                    index += 1
        return configuration

    def __call__(self, config, *, hint=None):
        """
        Get the configuration from the given config tuple.

        Parameters
        ----------
        config : tuple[EdgePoint, ...]
            The config tuple.
        hint : Configuration, optional
            A similar configuration used to copy if the given config is not calculated yet.

        Returns
        -------
            The result configuration.
        """
        if config not in self._pool:
            self._pool[config] = self._calculate_configuration(config, hint=hint)
        return self._pool[config]


class SamplingLattice(AbstractLattice):
    """
    Square lattice used for sampling.
    """

    __slots__ = ["_lattice"]

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
            # version 0 MAY have useless _cut_dimension
            if "_cut_dimension" in state:
                del state["_cut_dimension"]
        # version 1 to version 2
        if state["data_version"] == 1:
            state["data_version"] = 2
        # setstate
        for key, value in state.items():
            setattr(self, key, value)

    def __getstate__(self):
        # getstate
        state = {key: getattr(self, key) for key in _slotnames(self.__class__)}
        return state

    def __init__(self, abstract):
        """
        Create a sampling lattice from abstract lattice.

        Parameters
        ----------
        abstract : AbstractLattice
            The abstract lattice used to create sampling lattice.
        """
        super()._init_by_copy(abstract)

        self._lattice = [[self._construct_tensor(l1, l2) for l2 in range(self.L2)] for l1 in range(self.L1)]

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
