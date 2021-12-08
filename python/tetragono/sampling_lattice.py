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
import lazy
import TAT
from .auxiliaries import Auxiliaries
from .abstract_lattice import AbstractLattice
from .common_variable import clear_line
from .tensor_element import tensor_element


class Configuration(Auxiliaries):
    """
    Configuration system for square sampling lattice.
    """

    __slots__ = ["_owner"]

    def __init__(self, owner):
        """
        Create configuration system for the given sampling lattice.

        The configuration data is stored in sampling lattice, this system is only used for maintaining auxiliaries.

        Parameters
        ----------
        owner : SamplingLattice
            The sampling lattice owning this configuration system.
        """
        super().__init__(owner.L1, owner.L2, owner.cut_dimension, False, owner.Tensor)
        self._owner = owner
        # update exist configuration
        for l1 in range(owner.L1):
            for l2 in range(owner.L2):
                for orbit, edge in self._owner.physics_edges[l1, l2].items():
                    self[l1, l2, orbit] = self[l1, l2, orbit]

    def site_valid(self, l1, l2):
        """
        Check if specific site have valid configuration

        Parameters
        ----------
        l1, l2 : int
            The coordinate of the specific site.

        Returns
        -------
            The validity of this single site configuration.
        """
        for orbit, edge in self._owner.physics_edges[l1, l2].items():
            if self._owner._configuration[l1][l2][orbit] is None:
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
        return self._owner._configuration[l1][l2][orbit]

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
            self._owner._configuration[l1][l2][orbit] = None
            super().__setitem__((l1, l2), None)
            return
        this_configuration = self._construct_edge_point(value)
        if this_configuration == self._owner._configuration[l1][l2][orbit]:
            changed = False
        else:
            self._owner._configuration[l1][l2][orbit] = this_configuration
            changed = True
        if self._lattice[l1][l2]() is None or changed:
            if self.site_valid(l1, l2):
                shrinked_site = self._shrink_configuration((l1, l2), self._owner._configuration[l1][l2])
                super().__setitem__((l1, l2), shrinked_site)

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
        Tensor
            $\langle s\psi\rangle$ with several $s$ replaced.
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
            tensor = self._owner[l1l2]
            changed = False
            for orbit, configuration in self._owner._configuration[l1][l2].items():
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
                for orbit, edge in self._owner.physics_edges[l1, l2].items():
                    self.refresh_site((l1, l2, orbit))


class SamplingLattice(AbstractLattice):
    """
    Square lattice used for sampling.
    """

    __slots__ = ["_lattice", "_configuration", "_cut_dimension", "configuration"]

    def __setstate__(self, state):
        for key, value in state.items():
            setattr(self, key, value)
        self._create_auxiliaries()

    def __getstate__(self):
        state = {key: getattr(self, key) for key in _slotnames(self.__class__) if key != "configuration"}
        return state

    def __init__(self, abstract, cut_dimension):
        """
        Create a simple update lattice from abstract lattice.

        Parameters
        ----------
        abstract : AbstractLattice
            The abstract lattice used to create simple update lattice.
        cut_dimension : int
            The cut dimension when calculating auxiliary tensors.
        """
        super()._init_by_copy(abstract)

        self._lattice = [[self._construct_tensor(l1, l2) for l2 in range(self.L2)] for l1 in range(self.L1)]
        # EdgePoint = tuple[self.Symmetry, int]
        self._configuration = [[{orbit: None
                                 for orbit, edge in self.physics_edges[l1, l2].items()}
                                for l2 in range(self.L2)]
                               for l1 in range(self.L1)]
        self._cut_dimension = cut_dimension
        self._create_auxiliaries()

    @property
    def cut_dimension(self):
        """
        Get the cut dimension when calculating auxiliary tensor.

        Returns
        -------
        int
            The cut dimension when calculating auxiliary tensor.
        """
        return self._cut_dimension

    @cut_dimension.setter
    def cut_dimension(self, cut_dimension):
        """
        Get the cut dimension when calculating auxiliary tensor.

        Parameters
        ----------
        cut_dimension : int
            The cut dimension when calculating auxiliary tensor.
        """
        if self._cut_dimension != cut_dimension:
            self._cut_dimension = cut_dimension
            self._create_auxiliaries()

    def _create_auxiliaries(self):
        """
        Create new auxiliary system for the current given cut dimension.
        """
        self.configuration = Configuration(self)

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


class Observer():
    """
    Helper type for Observing the sampling lattice.
    """

    __slots__ = [
        "_owner", "_enable_hole", "_start", "_observer", "_result", "_count", "_total_weight", "_Delta", "_EDelta"
    ]

    def __init__(self, owner):
        """
        Create observer object for the given sampling lattice.

        Parameters
        ----------
        owner : SamplingLattice
            The owner of this obsever object.
        """
        self._owner = owner
        self._enable_hole = False
        self._start = False
        self._observer = {}  # dict[str, dict[tuple[tuple[int, int, int], ...], Tensor]]

        self._result = None  # dict[str, dict[tuple[tuple[int, int, int], ...], float]]
        self._count = None  # int
        self._total_weight = None  # float
        self._Delta = None  # list[list[Tensor]]
        self._EDelta = None  # list[list[Tensor]]

    def flush(self):
        """
        Flush all cached data in the observer object, need to be called every time a sampling sequence start.
        """
        self._result = {
            name: {positions: 0 for positions, observer in observers.items()
                  } for name, observers in self._observer.items()
        }
        self._count = 0
        self._total_weight = 0.0
        if self._enable_hole:
            self._Delta = [[self._owner[l1, l2].same_shape().zero()
                            for l2 in range(self._owner.L2)]
                           for l1 in range(self._owner.L1)]
            self._EDelta = [[self._owner[l1, l2].same_shape().zero()
                             for l2 in range(self._owner.L2)]
                            for l1 in range(self._owner.L1)]

    def add_observer(self, name, observers):
        """
        Add an observer set into this observer object, cannot add observer once observer started.

        Parameters
        ----------
        name : str
            This observer set name.
        observers : dict[tuple[tuple[int, int, int], ...], Tensor]
            The observer map.
        """
        if self._start:
            raise RuntimeError("Cannot enable hole after sampling start")
        self._observer[name] = observers

    def add_energy(self):
        """
        Add energy as an observer.
        """
        self.add_observer("energy", self._owner._hamiltonians)

    def enable_gradient(self):
        """
        Enable observing gradient.
        """
        if self._start:
            raise RuntimeError("Cannot enable gradient after sampling start")
        if "energy" not in self._observer:
            self.add_energy()
        self._enable_hole = True

    def __call__(self, reweight):
        """
        Collect observer value from current configuration, the sampling should have distribution based on
        $|\langle\psi s\rangle|^2$, If it is not, a reweight argument should be passed with a non-one float number.

        Parameters
        ----------
        reweight
            the weight for reweight in importance sampling.
        """
        self._start = True
        self._count += 1
        self._total_weight += reweight
        ws = self._owner.configuration.hole(())  # ws is a tensor
        inv_ws_conj = ws / (ws.norm_2()**2)
        inv_ws = inv_ws_conj.conjugate()
        all_name = {("T", "T")} | {(f"P_{l1}_{l2}_{orbit}", f"P_{l1}_{l2}_{orbit}") for l1 in range(self._owner.L1)
                                   for l2 in range(self._owner.L2)
                                   for orbit, edge in self._owner.physics_edges[l1, l2].items()}
        for name, observers in self._observer.items():
            if name == "energy" and self._enable_hole:
                calculating_gradient = True
                Es = 0
            else:
                calculating_gradient = False
            for positions, observer in observers.items():
                body = observer.rank // 2
                current_configuration = tuple(self._owner.configuration[positions[i]] for i in range(body))
                element_pool = tensor_element(observer)
                if current_configuration not in element_pool:
                    continue
                total_value = 0
                physics_names = [f"P_{positions[i][0]}_{positions[i][1]}_{positions[i][2]}" for i in range(body)]
                for other_configuration, observer_shrinked in element_pool[current_configuration].items():
                    wss = self._owner.configuration.replace({positions[i]: other_configuration[i] for i in range(body)
                                                            }).conjugate()
                    if wss.norm_num() == 0:
                        continue
                    value = inv_ws_conj.contract(observer_shrinked,
                                                 {(physics_names[i], f"I{i}") for i in range(body)}).edge_rename({
                                                     f"O{i}": physics_names[i] for i in range(body)
                                                 }).contract(wss, all_name)
                    total_value += float(value)
                self._result[name][positions] += total_value * reweight
                if calculating_gradient:
                    Es += total_value  # reweight will be multipled later
            if calculating_gradient:
                for l1 in range(self._owner.L1):
                    for l2 in range(self._owner.L2):
                        contract_name = all_name.copy()
                        for orbit, edge in self._owner.physics_edges[l1, l2].items():
                            contract_name.remove((f"P_{l1}_{l2}_{orbit}", f"P_{l1}_{l2}_{orbit}"))
                        if l1 == l2 == 0:
                            contract_name.remove(("T", "T"))
                        hole = self._owner.configuration.hole(((l1, l2),)).contract(inv_ws, contract_name)
                        hole = hole.edge_rename({
                            "L0": "R",
                            "R0": "L",
                            "U0": "D",
                            "D0": "U"
                        } | {
                            f"P_{l1}_{l2}_{orbit}": f"P{orbit}"
                            for orbit, edge in self._owner.physics_edges[l1, l2].items()
                        })

                        for orbit, shrinker in self._owner.configuration._get_shrinker(
                            (l1, l2), self._owner._configuration[l1][l2]):
                            hole = hole.contract(shrinker, {(f"P{orbit}", "P")}).edge_rename({"Q": f"P{orbit}"})

                        hole *= reweight
                        self._Delta[l1][l2] += hole
                        self._EDelta[l1][l2] += Es * hole

    @property
    def result(self):
        """
        Get the observer result.

        Returns
        -------
        dict[str, dict[tuple[tuple[int, int, int], ...], float]]
            The observer result of each observer set name and each site positions list.
        """
        return {
            name: {positions: value / self._total_weight for positions, value in data.items()
                  } for name, data in self._result.items()
        }

    @property
    def energy(self):
        """
        Get the observed energy per site.

        Returns
        -------
        float
            The energy per site.
        """
        return sum(self.result["energy"].values()) / self._owner.site_number

    @property
    def gradient(self):
        """
        Get the energy gradient for every tensor.

        Returns
        -------
        list[list[Tensor]]
            The gradient for every tensor.
        """
        return [[(self._EDelta[l1][l2] / self._total_weight) * 2 -
                 (self._Delta[l1][l2] / self._total_weight) * sum(self.result["energy"].values()) * 2
                 for l2 in range(self._owner.L2)]
                for l1 in range(self._owner.L1)]


class Sampling:
    """
    Helper type for run sampling for sampling lattice.
    """

    __slots__ = ["_owner"]

    def __init__(self, owner):
        """
        Create sampling object for the given sampling lattice.

        Parameters
        ----------
        owner : SamplingLattice
            The owner of this sampling object.
        """
        self._owner = owner

    def __call__(self):
        """
        Get the next sampling configuration

        Returns
        -------
        float
            The weight of reweight in importance sampling
        """
        raise NotImplementedError("Not implement in abstract sampling")


class SweepSampling(Sampling):
    """
    Sweep sampling.
    """

    __slots__ = ["_sweep_order"]

    def __init__(self, owner):
        super().__init__(owner)
        self._sweep_order = None  # list[tuple[tuple[int, int, int], ...]]

    def _single_term(self, positions, hamiltonian, ws):
        body = hamiltonian.rank // 2
        current_configuration = tuple(self._owner.configuration[l1l2o] for l1l2o in positions)  # tuple[EdgePoint, ...]
        element_pool = tensor_element(hamiltonian)
        if current_configuration not in element_pool:
            return ws
        possible_hopping = element_pool[current_configuration]
        if possible_hopping:
            hopping_number = len(possible_hopping)
            configuration_new, element = list(possible_hopping.items())[TAT.random.uniform_int(0, hopping_number - 1)()]
            hopping_number_s = len(element_pool[configuration_new])
            replacement = {positions[i]: configuration_new[i] for i in range(body)}
            wss = float(self._owner.configuration.replace(replacement))  # which return a tensor, we only need its norm
            p = (wss**2) / (ws**2) * hopping_number / hopping_number_s
            if TAT.random.uniform_real(0, 1)() < p:
                ws = wss
                for i in range(body):
                    self._owner.configuration[positions[i]] = configuration_new[i]
        return ws

    def __call__(self):
        owner = self._owner
        if not owner.configuration.valid():
            raise RuntimeError("Configuration not initialized")
        ws = float(owner.configuration.hole(()))
        if self._sweep_order is None:
            self._sweep_order = self._get_proper_position_order()
        for positions in self._sweep_order:
            hamiltonian = owner._hamiltonians[positions]
            ws = self._single_term(positions, hamiltonian, ws)
        self._sweep_order.reverse()
        for positions in self._sweep_order:
            hamiltonian = owner._hamiltonians[positions]
            ws = self._single_term(positions, hamiltonian, ws)
        return 1.

    def _get_proper_position_order(self):
        L1 = self._owner.L1
        L2 = self._owner.L2
        positions = set(self._owner._hamiltonians.keys())
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


class ErgodicSampling(Sampling):
    """
    Ergodic sampling.
    """

    __slots__ = ["total_step", "_edges"]

    def __init__(self, owner):
        super().__init__(owner)

        self._edges = [[{
            orbit: self._owner[l1, l2].edges(f"P{orbit}") for orbit, edge in self._owner.physics_edges[l1, l2].items()
        } for l2 in range(self._owner.L2)] for l1 in range(self._owner.L1)]

        self.total_step = 1
        for l1 in range(self._owner.L1):
            for l2 in range(self._owner.L2):
                for orbit, edge in self._edges[l1][l2].items():
                    self.total_step *= edge.dimension

    def __call__(self):
        owner = self._owner
        if not owner.configuration.valid():
            raise RuntimeError("Configuration not initialized")
        for l1 in range(owner.L1):
            for l2 in range(owner.L2):
                for orbit, edge in self._edges[l1][l2].items():
                    index = edge.get_index_from_point(owner.configuration[l1, l2, orbit])
                    index += 1
                    if index == edge.dimension:
                        owner.configuration[l1, l2, orbit] = edge.get_point_from_index(0)
                    else:
                        owner.configuration[l1, l2, orbit] = edge.get_point_from_index(index)
                        return owner.configuration.hole(()).norm_2()**2
        return owner.configuration.hole(()).norm_2()**2


class DirectSampling(Sampling):
    """
    Direct sampling.
    """

    __slots__ = []
