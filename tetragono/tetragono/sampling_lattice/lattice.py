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
from ..auxiliaries import SingleLayerAuxiliaries
from ..abstract_lattice import AbstractLattice
from ..common_toolkit import lattice_prod_sum, lattice_conjugate, showln, bcast_lattice_buffer


class Configuration(SingleLayerAuxiliaries):
    """
    Configuration system for square sampling lattice.
    """

    __slots__ = ["owner", "_configuration", "_holes"]

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

        result.owner = self.owner
        result._configuration = self.export_configuration()
        result._set_site_without_orbit()
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
        self.owner: SamplingLattice = owner

        # EdgePoint = tuple[self.Symmetry, int]
        # The data storage for spin configuration, access it by configuration[l1, l2, orbit] instead.
        self._configuration = [[{orbit: None
                                 for orbit in self.owner.physics_edges[l1, l2]}
                                for l2 in range(self.owner.L2)]
                               for l1 in range(self.owner.L1)]
        self._set_site_without_orbit()
        # The holes cache for this configuration, access it by configuration.holes()
        self._holes = None

    def _set_site_without_orbit(self):
        """
        Set super() item for sites without orbit. It is needed to be called when initializing a configuration.

        If some site has no orbit, it will never be set, so it is needed to set tensor of auxiliaries here.
        """
        for l1 in range(self.owner.L1):
            for l2 in range(self.owner.L2):
                if len(self.owner.physics_edges[l1, l2]) == 0:
                    super().__setitem__((l1, l2), self.owner[l1, l2])

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
        for orbit in self.owner.physics_edges[l1, l2]:
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
        for l1 in range(self.owner.L1):
            for l2 in range(self.owner.L2):
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

    def import_configuration(self, config):
        """
        Import the configuration of all the sites.

        Parameters
        ----------
        config : list[list[dict[int, ?EdgePoint]]]
            The configuration data of all the sites
        """
        for l1 in range(self.owner.L1):
            for l2 in range(self.owner.L2):
                for orbit, edge_point in config[l1][l2].items():
                    self[l1, l2, orbit] = edge_point

    def export_configuration(self):
        """
        Export the configuration of all the sites.

        Returns
        -------
        list[list[dict[int, EdgePoint]]]
            The configuration data of all the sites
        """
        return [[self._configuration[l1][l2].copy() for l2 in range(self.owner.L2)] for l1 in range(self.owner.L1)]

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
            symmetry = self.owner.Symmetry()  # work for NoSymmetry
            index = value
        else:
            symmetry, index = value
        symmetry = self.owner._construct_symmetry(symmetry)
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
            The orbit index and shrinker tensor, shrinker tensor name is "P" and "Q", where edge "P" is narrower one,
            edge "Q" is wider one, and edge "Q" should connect directly to physics edge, edge "P" is the same to physics
            edge.
        """
        l1, l2 = l1l2
        # An certain order is required.
        for orbit in sorted(self.owner.physics_edges[l1, l2]):
            edge = self.owner.physics_edges[l1, l2, orbit]
            symmetry, index = configuration[orbit]
            # P side is dimension one edge
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
        tensor = self.owner[l1l2]
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
        for l1 in range(self.owner.L1):
            for l2 in range(self.owner.L2):
                for orbit in self.owner.physics_edges[l1, l2]:
                    self.refresh_site((l1, l2, orbit))

    def holes(self):
        """
        Get the lattice holes of this configuration. so called holes is `<psi|s|partial_x psi> / <psi|s|psi>` where psi
        is the state and s is configuration, x is each tensor.

        Returns
        -------
        list[list[Tensor]]
            The holes of this configuration.
        """
        if self._holes is None:
            # Prepare
            ws = self.hole(())  # |s|psi>
            inv_ws_conj = ws / (ws.norm_2()**2)  # |s|psi> / <psi|s|psi>
            inv_ws = inv_ws_conj.conjugate()  # <psi|s| / <psi|s|psi>
            all_name = {("T", "T")} | {(f"P_{l1}_{l2}_{orbit}", f"P_{l1}_{l2}_{orbit}") for l1 in range(self.owner.L1)
                                       for l2 in range(self.owner.L2) for orbit in self.owner.physics_edges[l1, l2]}

            # Calculate
            holes = [[None for l2 in range(self.owner.L2)] for l1 in range(self.owner.L1)]
            # <psi|s|partial_x psi> / <psi|s|psi>
            for l1 in range(self.owner.L1):
                for l2 in range(self.owner.L2):
                    hole = self.hole(((l1, l2),))  # |s|partial_x psi>
                    # The right side is open in fact, because of partial_x, which is not Hilbert space, it is tensor space.
                    # So keep it a ket. But do not forget it is open in right side.
                    contract_name = all_name.copy()
                    for orbit in self.owner.physics_edges[l1, l2]:
                        contract_name.remove((f"P_{l1}_{l2}_{orbit}", f"P_{l1}_{l2}_{orbit}"))
                    if "T" not in hole.names:
                        contract_name.remove(("T", "T"))
                    # Contract and get <psi|s|partial_x psi> / <psi|s|psi>
                    hole = hole.contract(inv_ws, contract_name)
                    # Rename to the correct edge names.
                    hole = hole.edge_rename({
                        "L0": "R",
                        "R0": "L",
                        "U0": "D",
                        "D0": "U",
                        **{f"P_{l1}_{l2}_{orbit}": f"P{orbit}" for orbit in self.owner.physics_edges[l1, l2]},
                    })

                    # hole owns conjugated physics edge, because of partial_x. Expand it by connecting it with a physics
                    # edge but one dimension, and a conjugated edge but full dimension tensor, which is just what
                    # _get_shrinker returns.
                    for orbit, shrinker in self._get_shrinker((l1, l2), self._configuration[l1][l2]):
                        hole = hole.contract(shrinker, {(f"P{orbit}", "P")}).edge_rename({"Q": f"P{orbit}"})

                    holes[l1][l2] = hole
            self._holes = holes
        return self._holes


class TailDictTree:
    """
    A dict tree from the tail of list key.
    """
    __slots__ = ["_data"]

    def __init__(self):
        self._data = {}

    def __setitem__(self, key, value):
        key = list(key)
        current = self._data
        # The process is different for the last element, which will set key and value directly, but other will create a
        # dict first.
        while len(key) != 1:
            node = key.pop()
            if node not in current:
                current[node] = {}
            current = current[node]
        node = key.pop()
        current[node] = value

    def __getitem__(self, key):
        key = list(key)
        current = self._data
        while len(key) != 0:
            node = key.pop()
            current = current[node]
        return current

    def nearest(self, key):
        key = list(key)
        current = self._data
        while len(key) != 0:
            node = key.pop()
            if node in current:
                current = current[node]
            else:
                # If not found the same configuration in this site, find any config for this site. And then continue to
                # find the following site configuration.
                current = next(iter(current.values()))
        return current

    def __contains__(self, key):
        key = list(key)
        current = self._data
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

    __slots__ = ["owner", "tree"]

    # There are two kind of config format in this class
    # config list and config dict format
    # config list is often called as `config`
    # config dict or its container, aka, Configuration object, is often called as `configuration`

    def _config_dict_to_list(self, configuration):
        """
        Convert config dict to config list.

        Parameters
        ----------
        configuration : Configuration
            the configuration object.

        Returns
        -------
        tuple[EdgePoint, ...]
            the config list.
        """
        return tuple(configuration[l1, l2, orbit] for l1 in range(self.owner.L1) for l2 in range(self.owner.L2)
                     for orbit in self.owner.physics_edges[l1, l2])

    def _config_list_replace(self, config, replacement):
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
        for l1 in range(self.owner.L1):
            for l2 in range(self.owner.L2):
                for orbit, _ in self.owner.physics_edges[l1, l2].items():
                    if (l1, l2, orbit) in replacement:
                        config[index] = replacement[l1, l2, orbit]
                    index += 1
        return tuple(config)

    def _diff_two_config_dict(self, configuration_old, configuration_new):
        """
        Get the difference of two configurations.

        Parameters
        ----------
        configuration_old, configuration_new : Configuration
            The given two configurations.

        Returns
        -------
        dict[tuple[int, int, int], ?EdgePoint]
            The result replacement
        """
        replacement = {}
        for l1 in range(self.owner.L1):
            for l2 in range(self.owner.L2):
                for orbit in self.owner.physics_edges[l1, l2]:
                    if configuration_old[l1, l2, orbit] != configuration_new[l1, l2, orbit]:
                        replacement[l1, l2, orbit] = configuration_new[l1, l2, orbit]
        return replacement

    def __init__(self, owner):
        """
        Create configuration system pool for the given sampling lattice.

        Parameters
        ----------
        owner : SamplingLattice
            The sampling lattice owning this configuration pool.
        """
        self.owner: SamplingLattice = owner
        self.tree = TailDictTree()

    def wss(self, configuration, replacement):
        """
        Calculate wss of the configuration with the given replacement.

        Parameters
        ----------
        configuration : Configuration
            The given configuration.
        replacement : dict[tuple[int, int, int], ?EdgePoint]
            Replacement plan to modify configuration.

        Returns
        -------
        Tensor
            $\langle s\psi\rangle$ with several $s$ replaced.
        """
        # Try replace directly first
        wss = configuration.replace(replacement)
        if wss is not None:
            return wss

        # Try to find in the old config
        base_config = self._config_dict_to_list(configuration)
        config = self._config_list_replace(base_config, replacement)
        if config in self.tree:
            return self.tree[config].hole(())

        # Try to replace the nearest config
        nearest_configuration = self.tree.nearest(config)
        # This is replacement from nearest to the base configuration
        nearest_replacement = self._diff_two_config_dict(nearest_configuration, configuration)
        # Then update it with the input replacement, get the replacement from nearest to the result configuration.
        nearest_replacement.update(replacement)
        wss = nearest_configuration.replace(nearest_replacement)
        if wss is not None:
            return wss

        # Up till now, we may need to consider or create new configuration.
        # But do not create the result configuration directly.
        # Because usually, we measure only the correlation between to small region.
        # We hope to split the replacement to two parts, each of which is only in a small region.

        # Try to remove 2x2 area in replacement to find new near config
        replacement_1, replacement_2 = self._split_replacement(replacement)
        half_config = self._config_list_replace(base_config, replacement_1)
        # Ok, half replaced configuration has been found, try to find it in tree, otherwise create it.
        if half_config in self.tree:
            return self.tree[half_config].replace(replacement_2)

        # Cannot find it, so create the half replaced configuration, find its nearest again.
        half_nearest_configuration = self.tree.nearest(half_config)
        self.tree[half_config] = self._create_configuration(half_config, half_nearest_configuration)
        return self.tree[half_config].replace(replacement_2)

    def _create_configuration(self, config, base_configuration):
        """
        Calculate the configuration from the given config.

        Parameters
        ----------
        config : tuple[EdgePoint, ...]
            the config tuple.
        base_configuration : Configuration
            A similar configuration used to copy

        Returns
        -------
            The result configuration of the given config tuple.
        """
        configuration = base_configuration.copy()
        config = iter(config)
        for l1 in range(self.owner.L1):
            for l2 in range(self.owner.L2):
                for orbit in self.owner.physics_edges[l1, l2]:
                    # If they are the same, auxiliaries tensor in Configuration object will not be refreshed.:
                    configuration[l1, l2, orbit] = next(config)
        return configuration

    def _split_replacement(self, replacement):
        """
        Split a replacement into two part, the second of which can be calculated by auxiliary replace.

        Parameters
        ----------
        replacement : dict[tuple[int, int, int], ?EdgePoint]
            The input replacement

        Returns
        -------
        tuple[dict[tuple[int, int, int], ?EdgePoint], dict[tuple[int, int, int], ?EdgePoint]]
            The result two replacements
        """
        replacement_1 = {}
        replacement_2 = {}
        up = self.owner.L1
        down = -1
        left = self.owner.L2
        right = -1
        for l1 in range(self.owner.L1):
            for l2 in range(self.owner.L2):
                for orbit in self.owner.physics_edges[l1, l2]:
                    site = (l1, l2, orbit)
                    if site in replacement:
                        edge_point = replacement[site]
                        if l1 > down - 2 and l1 < up + 2 and l2 > right - 2 and l2 < left + 2:
                            replacement_2[site] = edge_point
                            if l1 < up:
                                up = l1
                            if l1 > down:
                                down = l1
                            if l2 < left:
                                left = l2
                            if l2 > right:
                                right = l2
                        else:
                            replacement_1[site] = edge_point
        return replacement_1, replacement_2

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
        config = self._config_dict_to_list(configuration)
        if config not in self.tree:
            self.tree[config] = configuration
        return self.tree[config]


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
        Create a sampling lattice from abstract lattice.

        Parameters
        ----------
        abstract : AbstractLattice
            The abstract lattice used to create sampling lattice.
        """
        super()._init_by_copy(abstract)

        # The data storage of the lattice tensor, access it by state[l1, l2] instead.
        self._lattice = np.array([[self._construct_tensor(l1, l2) for l2 in range(self.L2)] for l1 in range(self.L1)])

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

    def expand_dimension(self, new_dimension, epsilon):
        """
        Expand dimension of sampling lattice. If new_dimension equals to the origin dimension and epsilon is zero, this
        function will only fix the lattice gauge.

        Parameters
        ----------
        new_dimension : int | float
            The new dimension, or the amplitude of dimension expandance.
        epsilon : float
            The relative error added into tensor.
        """
        for l1 in range(self.L1):
            for l2 in range(self.L2):
                if l1 != 0 and l1 % 2 == 0:
                    self._expand_vertical(l1 - 1, l2, new_dimension, epsilon)
        for l1 in range(self.L1):
            for l2 in range(self.L2):
                if l1 != 0 and l1 % 2 == 1:
                    self._expand_vertical(l1 - 1, l2, new_dimension, epsilon)
        for l1 in range(self.L1):
            for l2 in range(self.L2):
                if l2 != 0 and l2 % 2 == 0:
                    self._expand_horizontal(l1, l2 - 1, new_dimension, epsilon)
        for l1 in range(self.L1):
            for l2 in range(self.L2):
                if l2 != 0 and l2 % 2 == 1:
                    self._expand_horizontal(l1, l2 - 1, new_dimension, epsilon)

    def _expand_horizontal(self, l1, l2, new_dimension, epsilon):
        left = self[l1, l2]
        right = self[l1, l2 + 1]
        original_dimension = left.edges("R").dimension
        if isinstance(new_dimension, float):
            new_dimension = round(original_dimension * new_dimension)
        if epsilon == 0:
            left_q, left_r = left.qr("r", {"R"}, "R", "L")
            right_q, right_r = right.qr("r", {"L"}, "L", "R")
        else:
            left_q, left_r = left.qr("r", {*(name for name in left.names if name.startswith("P")), "R"}, "R", "L")
            right_q, right_r = right.qr("r", {*(name for name in right.names if name.startswith("P")), "L"}, "L", "R")
        left_r = left_r.edge_rename({name: f"L_{name}" for name in left_r.names})
        right_r = right_r.edge_rename({name: f"R_{name}" for name in right_r.names})
        big = left_r.contract(right_r, {("L_R", "R_L")})
        norm = big.norm_max()
        big += big.same_shape().randn() * epsilon * norm
        u, s, v = big.svd({l_name for l_name in big.names if l_name.startswith("L_")}, "R", "L", "L", "R",
                          new_dimension)
        i = s.same_shape().identity({("L", "R")})
        delta = np.sqrt(np.abs(s.storage))
        delta[delta == 0] = 1
        s.storage /= delta
        i.storage *= delta
        left = left_q.contract(u, {("R", "L_L")}).contract(s, {("R", "L")})
        right = right_q.contract(v, {("L", "R_R")}).contract(i, {("L", "R")})
        self[l1, l2] = left.edge_rename({l_name: l_name[2:] for l_name in left.names if l_name.startswith("L_")})
        self[l1, l2 + 1] = right.edge_rename({r_name: r_name[2:] for r_name in right.names if r_name.startswith("R_")})

    def _expand_vertical(self, l1, l2, new_dimension, epsilon):
        up = self[l1, l2]
        down = self[l1 + 1, l2]
        original_dimension = up.edges("D").dimension
        if isinstance(new_dimension, float):
            new_dimension = round(original_dimension * new_dimension)
        if epsilon == 0:
            up_q, up_r = up.qr("r", {"D"}, "D", "U")
            down_q, down_r = down.qr("r", {"U"}, "U", "D")
        else:
            up_q, up_r = up.qr("r", {*(name for name in up.names if name.startswith("P")), "D"}, "D", "U")
            down_q, down_r = down.qr("r", {*(name for name in down.names if name.startswith("P")), "U"}, "U", "D")
        up_r = up_r.edge_rename({name: f"U_{name}" for name in up_r.names})
        down_r = down_r.edge_rename({name: f"D_{name}" for name in down_r.names})
        big = up_r.contract(down_r, {("U_D", "D_U")})
        norm = big.norm_max()
        big += big.same_shape().randn() * epsilon * norm
        u, s, v = big.svd({u_name for u_name in big.names if u_name.startswith("U_")}, "D", "U", "U", "D",
                          new_dimension)
        i = s.same_shape().identity({("U", "D")})
        delta = np.sqrt(np.abs(s.storage))
        delta[delta == 0] = 1
        s.storage /= delta
        i.storage *= delta
        up = up_q.contract(u, {("D", "U_U")}).contract(s, {("D", "U")})
        down = down_q.contract(v, {("U", "D_D")}).contract(i, {("U", "D")})
        self[l1, l2] = up.edge_rename({u_name: u_name[2:] for u_name in up.names if u_name.startswith("U_")})
        self[l1 + 1, l2] = down.edge_rename({d_name: d_name[2:] for d_name in down.names if d_name.startswith("D_")})

    def lattice_dot(self, a=None, b=None):
        """
        Calculate the dot of two lattice shape data. If None is given, its own lattice data will be used.

        Parameters
        ----------
        a, b : list[list[Tensor]] | None
            The lattice shape data.
        """
        if a is None:
            a = self._lattice
        if b is None:
            b = self._lattice
        return lattice_prod_sum(lattice_conjugate(a), b).real

    def apply_gradient(self, gradient, step_size):
        """
        Apply the gradient to the lattice.

        Parameters
        ----------
        gradient : list[list[Tensor]]
            The gradient calculated by observer object.
        step_size : float
            The gradient step size.
        """
        self._lattice -= step_size * gradient

    def bcast_lattice(self, root=0):
        """
        Bcast the lattice, to ensure the data keep the same across different process.
        """
        bcast_lattice_buffer(self._lattice, root=root)
