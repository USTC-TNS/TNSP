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
import lazy
import TAT
from .auxiliaries import Auxiliaries
from .double_layer_auxiliaries import DoubleLayerAuxiliaries
from .exact_state import ExactState
from .abstract_state import AbstractState
from .abstract_lattice import AbstractLattice
from .common_variable import clear_line
from .tensor_element import tensor_element


class Configuration(Auxiliaries):
    __slots__ = ["owner", "EdgePoint", "_configuration"]

    def __init__(self, owner: SamplingLattice, cut_dimension: int) -> None:
        super().__init__(owner.L1, owner.L2, cut_dimension, False, owner.Tensor)
        self.owner: SamplingLattice = owner
        self.EdgePoint = tuple[self.owner.Symmetry, int]
        self._configuration: list[list[None | self.EdgePoint]] = [[None for l2 in range(self.owner.L2)] for l1 in range(self.owner.L1)]

    def valid(self) -> bool:
        for l1 in range(self.owner.L1):
            for l2 in range(self.owner.L2):
                if self._configuration[l1][l2] is None:
                    return False
        return True

    def __getitem__(self, l1l2: tuple[int, int]) -> None | self.EdgePoint:
        l1, l2 = l1l2
        return self._configuration[l1][l2]

    def __setitem__(self, l1l2: tuple[int, int], value: "EdgePoint") -> None:
        l1, l2 = l1l2
        if value is None:
            self._configuration[l1][l2] = None
            super().__setitem__(l1l2, None)
            return
        this_configuration: self.EdgePoint = self._get_edge_point(value)
        if this_configuration != self._configuration[l1][l2]:
            self._configuration[l1][l2] = this_configuration
            super().__setitem__(l1l2, self._shrink_configuration(l1l2, this_configuration))

    def __delitem__(self, l1l2: tuple[int, int]) -> None:
        self.__setitem__(l1l2, None)

    def replace(self, replacement: dict[tuple[int, int], "EdgePoint"], *, hint=None) -> self.Tensor:
        base_replacement: dict[tuple[int, int], self.Tensor] = {}
        for k, v in replacement.items():
            t = self._shrink_configuration(k, self._get_edge_point(v))
            base_replacement[k] = t
        return super().replace(base_replacement, hint=hint)

    def _get_edge_point(self, value: "EdgePoint") -> self.EdgePoint:
        if not isinstance(value, tuple):
            symmetry = self.owner.Symmetry()  # work for NoSymmetry
            index = value
        else:
            symmetry, index = value
        if not isinstance(symmetry, self.owner.Symmetry):
            symmetry = self.owner.Symmetry(symmetry)
        return (symmetry, index)

    def _get_shrinker(self, l1l2: tuple[int, int], configuration: self.EdgePoint) -> self.Tensor:
        l1, l2 = l1l2
        tensor: self.Tensor = self.owner[l1l2]
        symmetry, index = configuration
        # P side is dimension - 1 edge
        # Q side is connected to lattice
        shrinker: self.Tensor = self.Tensor(["P", "Q"], [[(symmetry, 1)], tensor.edges("P").conjugated()]).zero()
        shrinker[{"Q": (-symmetry, index), "P": (symmetry, 0)}] = 1
        return shrinker

    def _shrink_configuration(self, l1l2: tuple[int, int], configuration: self.EdgePoint) -> self.Tensor:
        l1, l2 = l1l2
        tensor: self.Tensor = self.owner[l1l2]
        shrinker: self.Tensor = self._get_shrinker(l1l2, configuration).edge_rename({"P": f"P_{l1}_{l2}"})
        return tensor.contract(shrinker, {("P", "Q")})

    def refresh_site(self, l1l2: tuple[int, int]) -> None:
        conf = self[l1l2]
        del self[l1l2]
        self[l1l2] = conf

    def refresh_all(self) -> None:
        for l1 in range(self.owner.L1):
            for l2 in range(self.owner.L2):
                self.refresh_site((l1, l2))


class SamplingLattice(AbstractLattice):
    __slots__ = ["_lattice", "configuration", "_element_pool"]

    def __init__(self, abstract: AbstractLattice, cut_dimension: int) -> None:
        super()._init_by_copy(abstract)

        self._lattice: list[list[self.Tensor]] = [[self._construct_tensor(l1, l2) for l2 in range(self.L2)] for l1 in range(self.L1)]
        self.configuration: Configuration = Configuration(self, cut_dimension)

    def __getitem__(self, l1l2: tuple[int, int]) -> self.Tensor:
        l1, l2 = l1l2
        return self._lattice[l1][l2]

    def __setitem__(self, l1l2: tuple[int, int], value: self.Tensor) -> None:
        l1, l2 = l1l2
        self._lattice[l1][l2] = value

    def exact_state(self) -> ExactState:
        result: ExactState = ExactState(self)
        for l1 in range(self.L1):
            for l2 in range(self.L2):
                rename_map: dict[str, str] = {}
                rename_map["P"] = f"P_{l1}_{l2}"
                if l1 != self.L1 - 1:
                    rename_map["D"] = f"D_{l2}"
                this: self.Tensor = self[l1, l2].edge_rename(rename_map)
                if l1 == l2 == 0:
                    result.vector = this
                else:
                    contract_pair: set[tuple[int, int]] = set()
                    if l2 != 0:
                        contract_pair.add(("R", "L"))
                    if l1 != 0:
                        contract_pair.add((f"D_{l2}", "U"))
                    result.vector = result.vector.contract(this, contract_pair)
        return result


class Observer():
    __slots__ = ["state", "_enable_hole", "_start", "_observer", "_result", "_count", "_total_weight", "_Delta", "_EDelta"]

    def __init__(self, state: SamplingLattice) -> None:
        self.state: SamplingLattice = state
        self._enable_hole: bool = False
        self._start: bool = False
        self._observer: dict[str, dict[tuple[tuple[int, int], ...], self.state.Tensor]] = {}
        self._result: dict[str, dict[tuple[tuple[int, int], ...], float]] | None = None
        self._count: int | None = None
        self._total_weight: float | None = None
        self._Delta: list[list[self.state.Tensor]] | None = None
        self._EDelta: list[list[self.state.Tensor]] | None = None

    def flush(self) -> None:
        self._result = {name: {positions: 0 for positions, observer in observers.items()} for name, observers in self._observer.items()}
        self._count = 0
        self._total_weight = 0
        if self._enable_hole:
            self._Delta = [[self.state[l1, l2].same_shape().zero() for l2 in range(self.state.L2)] for l1 in range(self.state.L1)]
            self._EDelta = [[self.state[l1, l2].same_shape().zero() for l2 in range(self.state.L2)] for l1 in range(self.state.L1)]

    def add_observer(self, name: str, observers: dict[tuple[tuple[int, int], ...], self.state.Tensor]) -> None:
        if self._start:
            raise RuntimeError("Cannot enable hole after sampling start")
        self._observer[name] = observers

    def add_energy(self) -> None:
        self.add_observer("energy", self.state._hamiltonians)

    def enable_gradient(self) -> None:
        if self._start:
            raise RuntimeError("Cannot enable gradient after sampling start")
        if "energy" not in self._observer:
            self.add_energy()
        self._enable_hole = True

    def __call__(self, reweight: float) -> None:
        """
        Collect observer value from current configuration

        Parameters
        ----------
        reweight
            the weight for reweight in importance sampling
        """
        self._start = True
        self._count += 1
        self._total_weight += reweight
        configuration_t = tuple[self.state.configuration.EdgePoint, ...]
        ws: self.state.Tensor = self.state.configuration.hole(())
        inv_ws_conj: self.state.Tensor = ws / (ws.norm_2()**2)
        inv_ws: self.state.Tensor = inv_ws_conj.conjugate()
        all_name: set[tuple[str, str]] = {("T", "T")} | {(f"P_{l1}_{l2}", f"P_{l1}_{l2}") for l1 in range(self.state.L1) for l2 in range(self.state.L2)}
        for name, observers in self._observer.items():
            calculate_gradient: bool
            if name == "energy" and self._enable_hole:
                calculating_gradient = True
                Es: float = 0
            else:
                calculating_gradient = False
            for positions, observer in observers.items():
                body: int = observer.rank // 2
                current_configuration: configuration_t = tuple(self.state.configuration[positions[i]] for i in range(body))
                element_pool = tensor_element(observer)
                if current_configuration not in element_pool:
                    continue
                total_value: float = 0
                physics_names: list[str] = [f"P_{positions[i][0]}_{positions[i][1]}" for i in range(body)]
                for other_configuration, observer_shrinked in element_pool[current_configuration].items():
                    wss: self.state.Tensor = self.state.configuration.replace({positions[i]: other_configuration[i] for i in range(body)}).conjugate()
                    if wss.norm_num() == 0:
                        continue
                    value: self.state.Tensor = inv_ws_conj.contract(observer_shrinked, {(physics_names[i], f"I{i}") for i in range(body)}).edge_rename({f"O{i}": physics_names[i] for i in range(body)
                                                                                                                                                       }).contract(wss, all_name)
                    total_value += float(value)
                self._result[name][positions] += total_value * reweight
                if calculating_gradient:
                    Es += total_value
            if calculating_gradient:
                for l1 in range(self.state.L1):
                    for l2 in range(self.state.L2):
                        contract_name: set[tuple[str, str]] = all_name.copy()
                        contract_name.remove((f"P_{l1}_{l2}", f"P_{l1}_{l2}"))
                        if l1 == l2 == 0:
                            contract_name.remove(("T", "T"))
                        hole: self.state.Tensor = self.state.configuration.hole(((l1, l2),)).contract(inv_ws, contract_name)
                        hole = hole.edge_rename({"L0": "R", "R0": "L", "U0": "D", "D0": "U", f"P_{l1}_{l2}": "P"})

                        shrinker: self.state.Tensor = self.state.configuration._get_shrinker((l1, l2), self.state.configuration[l1, l2])
                        grad: self.state.Tensor = hole.contract(shrinker, {("P", "P")}).edge_rename({"Q": "P"})

                        grad *= reweight
                        self._Delta[l1][l2] += grad
                        self._EDelta[l1][l2] += Es * grad

    @property
    def result(self) -> dict[str, dict[tuple[tuple[int, int], ...], float]]:
        return {name: {positions: value / self._total_weight for positions, value in data.items()} for name, data in self._result.items()}

    @property
    def energy(self) -> float:
        return sum(self.result["energy"].values())

    @property
    def gradient(self) -> list[list[self.state.Tensor]]:
        return [[(self._EDelta[l1][l2] / self._total_weight) - (self._Delta[l1][l2] / self._total_weight) * self.energy for l2 in range(self.state.L2)] for l1 in range(self.state.L1)]


class Sampling:
    __slots__ = ["state"]

    def __init__(self, state: SamplingLattice) -> None:
        self.state: SamplingLattice = state

    def __call__(self) -> float:
        """
        Get the next sampling configuration

        Returns
        -------
        float
            The weight of reweight in importance sampling
        """
        raise NotImplementedError("Not implement in abstract sampling")


class SweepSampling(Sampling):
    __slots__ = ["sweep_order"]

    def __init__(self, state: SamplingLattice) -> None:
        super().__init__(state)
        self.sweep_order: list[tuple[tuple[int, int], ...]] | None = None

    def _single_term(self, positions: tuple[tuple[int, int], ...], hamiltonian: self.state.Tensor, ws: float) -> float:
        body: int = hamiltonian.rank // 2
        configuration_t = tuple[self.state.configuration.EdgePoint, ...]
        current_configuration: configuration_t = tuple(self.state.configuration[l1l2] for l1l2 in positions)
        element_pool = tensor_element(hamiltonian)
        if current_configuration not in element_pool:
            return ws
        possible_hopping: dict[configuration_t, self.state.Tensor] = element_pool[current_configuration]
        if possible_hopping:
            hopping_number: int = len(possible_hopping)
            configuration_new, element = list(possible_hopping.items())[TAT.random.uniform_int(0, hopping_number - 1)()]
            hopping_number_s: int = len(element_pool[configuration_new])
            replacement = {positions[i]: configuration_new[i] for i in range(body)}
            wss: float = float(self.state.configuration.replace(replacement))  # which return a tensor, we only need its norm
            p: float = (wss**2) / (ws**2) * hopping_number / hopping_number_s
            if TAT.random.uniform_real(0, 1)() < p:
                ws = wss
                for i in range(body):
                    self.state.configuration[positions[i]] = configuration_new[i]
        return ws

    def __call__(self) -> float:
        state: SamplingLattice = self.state
        if not state.configuration.valid():
            raise RuntimeError("Configuration not initialized")
        ws: float = float(state.configuration.hole(()))
        if self.sweep_order is None:
            self.sweep_order: list[tuple[tuple[int, int], ...]] = self._get_proper_position_order()
        for positions in self.sweep_order:
            hamiltonian: state.Tensor = state._hamiltonians[positions]
            ws = self._single_term(positions, hamiltonian, ws)
        self.sweep_order.reverse()
        for positions in self.sweep_order:
            hamiltonian: state.Tensor = state._hamiltonians[positions]
            ws = self._single_term(positions, hamiltonian, ws)
        return 1.

    def _get_proper_position_order(self) -> list[tuple[tuple[int, int], ...]]:
        L1: int = self.state.L1
        L2: int = self.state.L2
        positions: set[tuple[tuple[int, int], ...]] = set(self.state._hamiltonians.keys())
        result = []
        for l1 in range(L1):
            for l2 in range(L2):
                p = ((l1, l2),)
                if p in positions:
                    positions.remove(p)
                    result.append(p)
                p = ((l1, l2), (l1, l2 + 1))
                if p in positions:
                    positions.remove(p)
                    result.append(p)
        for l2 in range(L2):
            for l1 in range(L1):
                p = ((l1, l2), (l1 + 1, l2))
                if p in positions:
                    positions.remove(p)
                    result.append(p)
        if len(positions) != 0:
            raise NotImplementedError("Not implemented hamiltonian")
        return result


class ErgodicSampling(Sampling):
    __slots__ = ["total_step", "edges"]

    def __init__(self, state: SamplingLattice) -> None:
        super().__init__(state)
        self.edges: list[list[self.state.Edge]] = [[self.state[l1, l2].edges("P") for l2 in range(self.state.L2)] for l1 in range(self.state.L1)]
        self.total_step: int = 1
        for l1 in range(self.state.L1):
            for l2 in range(self.state.L2):
                self.total_step *= self.edges[l1][l2].dimension

    def __call__(self) -> float:
        state: SamplingLattice = self.state
        if not state.configuration.valid():
            raise RuntimeError("Configuration not initialized")
        for l1 in range(state.L1):
            for l2 in range(state.L2):
                index: int = self.edges[l1][l2].get_index_from_point(self.state.configuration[l1, l2])
                index += 1
                if index == self.edges[l1][l2].dimension:
                    self.state.configuration[l1, l2] = self.edges[l1][l2].get_point_from_index(0)
                else:
                    self.state.configuration[l1, l2] = self.edges[l1][l2].get_point_from_index(index)
                    return self.state.configuration.hole(()).norm_2()**2
        return self.state.configuration.hole(()).norm_2()**2


class DirectSampling(Sampling):
    __slots__ = []
