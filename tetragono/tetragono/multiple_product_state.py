#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2022 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
from .abstract_state import AbstractState
from .common_toolkit import allreduce_buffer
from .sampling_tools.tensor_element import tensor_element
from .multiple_product_ansatz import *


class MultipleProductState(AbstractState):
    """
    The multiple product state, which is product of several subansatz.
    """

    __slots__ = ["ansatzes"]

    def __init__(self, abstract):
        """
        Create multiple product state from a given abstract state.

        Parameters
        ----------
        abstract : AbstractState
            The abstract state used to create multiple product state.
        """
        super()._init_by_copy(abstract)
        self.ansatzes = {}

    def add_ansatz(self, ansatz, name=None):
        """
        Add an ansatz.

        Parameters
        ----------
        ansatz : Ansatz
            The ansatz to be made.
        name : str, optional
            The name of the new ansatz.
        """
        if name is None:
            name = str(len(self.ansatzes))
        self.ansatzes[name] = ansatz

    def weight_and_delta(self, configurations, calculate_delta):
        """
        Calculate weight and delta of all ansatz.

        Parameters
        ----------
        configuration : dict[tuple[int, int, int], int]
            The given configuration to calculate weight and delta
        calculate_delta : set[str]
            The iterator of name of ansatz to calculate delta.

        Returns
        -------
        tuple[list[complex | float], list[dict[str, ansatz]]]
            The weight and the delta ansatz.
        """
        number = len(configurations)
        weight = [1. for _ in range(number)]
        delta = [{} for _ in range(number)]
        for name, ansatz in self.ansatzes.items():
            sub_weight, sub_delta = ansatz.weight_and_delta(configurations, name in calculate_delta)
            for i in range(number):
                weight[i] *= sub_weight[i]
            if sub_delta is not None:
                for i in range(number):
                    delta[i][name] = sub_delta[i] / sub_weight[i]
        for i in range(number):
            this_weight = weight[i]
            this_delta = delta[i]
            for name in this_delta:
                this_delta[name] *= this_weight
        return weight, delta

    def apply_gradient(self, gradient, step_size, relative):
        """
        Apply the gradient to the state.

        Parameters
        ----------
        gradient : dict[str, Delta]
            The gradient calculated by observer object.
        step_size : float
            The gradient step size.
        relative : bool
            Use relative step size or not.
        """
        for name in gradient:
            self.ansatzes[name].apply_gradient(gradient[name], step_size, relative)


class Sampling:
    """
    Metropois sampling object for multiple product state.
    """

    __slots__ = ["_owner", "configuration", "_hopping_hamiltonians", "ws"]

    def __init__(self, owner, configuration, hopping_hamiltonians):
        """
        Create sampling object.

        Parameters
        ----------
        owner : MultipleProductState
            The owner of this sampling object
        configuration : dict[tuple[int, int, int], int]
            The initial configuration.
        hopping_hamiltonian : None | dict[tuple[tuple[int, int, int], ...], Tensor]
            The hamiltonian used in hopping, using the state hamiltonian if this is None.
        """
        self._owner = owner
        self.configuration = configuration
        if hopping_hamiltonians is not None:
            self._hopping_hamiltonians = hopping_hamiltonians
        else:
            self._hopping_hamiltonians = self._owner._hamiltonians
        self._hopping_hamiltonians = list(self._hopping_hamiltonians.items())
        self.refresh()

    def __call__(self):
        """
        Get the next configuration.
        """
        hamiltonian_number = len(self._hopping_hamiltonians)
        positions, hamiltonian = self._hopping_hamiltonians[TAT.random.uniform_int(0, hamiltonian_number - 1)()]
        body = hamiltonian.rank // 2
        current_configuration = tuple((TAT.No.Symmetry(), self.configuration[l1l2o]) for l1l2o in positions)
        element_pool = tensor_element(hamiltonian)
        if current_configuration not in element_pool:
            return
        possible_hopping = element_pool[current_configuration]
        if len(possible_hopping) == 0:
            return
        hopping_number = len(possible_hopping)
        current_configuration_s, _ = list(possible_hopping.items())[TAT.random.uniform_int(0, hopping_number - 1)()]
        hopping_number_s = len(element_pool[current_configuration_s])
        configuration_s = self.configuration.copy()
        for i in range(body):
            _, configuration_s[positions[i]] = current_configuration_s[i]
        [wss], _ = self._owner.weight_and_delta([configuration_s], set())
        p = (np.linalg.norm(wss)**2) / (np.linalg.norm(self.ws)**2) * hopping_number / hopping_number_s
        if TAT.random.uniform_real(0, 1)() < p:
            self.configuration = configuration_s
            self.ws = wss

    def refresh(self):
        """
        Refresh ws, call it after state updated.
        """
        [self.ws], _ = self._owner.weight_and_delta([self.configuration], set())


class Observer:

    __slots__ = ["_owner", "_enable_gradient", "_observer", "_start", "_count", "_result", "_Delta", "_EDelta"]

    def __init__(self, owner):
        """
        Create observer object for the given multiple product state.

        Parameters
        ----------
        owner : MultipleProductState
            The owner of this obsever object.
        """
        self._owner = owner

        self._enable_gradient = set()
        self._observer = {}

        self._start = False
        self._count = None
        self._result = None
        self._Delta = None
        self._EDelta = None

    def __enter__(self):
        """
        Enter sampling loop, flush all cached data in the observer object.
        """
        self._start = True
        self._count = 0
        self._result = {
            name: {positions: 0.0 for positions, observer in observers.items()
                  } for name, observers in self._observer.items()
        }
        self._Delta = {name: None for name in self._enable_gradient}
        self._EDelta = {name: None for name in self._enable_gradient}

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit sampling loop, reduce observed values, used when running with multiple processes.
        """
        if exc_type is not None:
            return False
        buffer = []
        for name, observers in self._observer.items():
            for positions in observers:
                buffer.append(self._result[name][positions])
        buffer.append(self._count)

        buffer = np.array(buffer)
        allreduce_buffer(buffer)
        buffer = buffer.tolist()

        self._count = buffer.pop()
        for name, observer in reversed(self._observer.items()):
            for positions in reversed(observers):
                self._result[name][positions] = buffer.pop()

        for name in self._enable_gradient:
            self._owner.ansatzes[name].allreduce_delta(self._Delta[name])
            self._owner.ansatzes[name].allreduce_delta(self._EDelta[name])

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
            name: {positions: self._result[name][positions] / self._count for positions, _ in data.items()
                  } for name, data in self._observer.items()
        }

    @property
    def total_energy(self):
        """
        Get the observed energy.

        Returns
        -------
        float
            The total energy.
        """
        return sum(self._result["energy"][positions] for positions, _ in self._observer["energy"].items()) / self._count

    @property
    def energy(self):
        """
        Get the observed energy per site.

        Returns
        -------
        float
            The energy per site.
        """
        return self.total_energy / self._owner.site_number

    @property
    def gradient(self):
        """
        Get the energy gradient for every subansatz.

        Returns
        -------
        dict[str, Delta]
            The gradient for every subansatz.
        """
        energy = self.total_energy
        return {
            name: 2 * self._EDelta[name] / self._count - 2 * energy * self._Delta[name] / self._count
            for name in self._enable_gradient
        }

    def enable_gradient(self, ansatz_name=None):
        """
        Enable observing gradient for specified ansatz.

        Parameters
        ----------
        ansatz_name : str | list[str] | None
            The ansatzes of which the gradient should be calculated.
        """
        if self._start:
            raise RuntimeError("Cannot enable gradient after sampling start")
        if "energy" not in self._observer:
            self.add_energy()
        if ansatz_name is None:
            ansatz_name = self._owner.ansatzes.keys()
        if isinstance(ansatz_name, str):
            ansatz_name = [ansatz_name]
        for name in ansatz_name:
            self._enable_gradient.add(name)

    def add_observer(self, name, observer):
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
            raise RuntimeError("Canot add observer after sampling start")
        self._observer[name] = observer

    def add_energy(self):
        """
        Add energy as an observer.
        """
        self.add_observer("energy", self._owner._hamiltonians)

    def __call__(self, configuration):
        """
        Collect observer value from current configuration.

        Parameters
        ----------
        configuration : dict[tuple[int, int, int], int]
            The current configuration.
        """
        owner = self._owner
        self._count += 1
        [ws], [delta] = owner.weight_and_delta([configuration], self._enable_gradient)
        # find wss
        configuration_list = []
        configuration_map = {}
        for name, observers in self._observer.items():
            configuration_map[name] = {}
            for positions, observer in observers.items():
                configuration_map[name][positions] = {}
                body = observer.rank // 2
                current_configuration = tuple((TAT.No.Symmetry(), configuration[positions[i]]) for i in range(body))
                element_pool = tensor_element(observer)
                if current_configuration not in element_pool:
                    continue
                for other_configuration, observer_shrinked in element_pool[current_configuration].items():
                    new_configuration = configuration.copy()
                    for i in range(body):
                        _, new_configuration[positions[i]] = other_configuration[i]
                    configuration_map[name][positions][other_configuration] = (len(configuration_list),
                                                                               observer_shrinked.storage[0])
                    configuration_list.append(new_configuration)
        wss_list, _ = owner.weight_and_delta(configuration_list, set())
        # measure
        for name, configuration_map_name in configuration_map.items():
            if name == "energy":
                Es = 0.0
            for positions, configuration_map_name_positions in configuration_map_name.items():
                total_value = 0
                for _, [index, hamiltonian_term] in configuration_map_name_positions.items():
                    wss = wss_list[index]
                    value = (wss / ws) * hamiltonian_term
                    total_value += complex(value)
                to_save = total_value.real
                self._result[name][positions] += to_save
                if name == "energy":
                    Es += total_value
            if name == "energy":
                if self._owner.Tensor.is_real:
                    Es = Es.real
                else:
                    Es = Es.conjugate()
                for ansatz_name in self._enable_gradient:
                    this_delta = delta[ansatz_name] / ws
                    if self._Delta[ansatz_name] is None:
                        self._Delta[ansatz_name] = this_delta
                    else:
                        self._Delta[ansatz_name] += this_delta
                    if self._EDelta[ansatz_name] is None:
                        self._EDelta[ansatz_name] = Es * this_delta
                    else:
                        self._EDelta[ansatz_name] += Es * this_delta
