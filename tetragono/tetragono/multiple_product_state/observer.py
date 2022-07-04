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

import os
import pickle
import numpy as np
from ..sampling_tools.tensor_element import tensor_element
from ..common_toolkit import allreduce_buffer, mpi_rank, show, MPI


class Observer:

    __slots__ = [
        "_owner", "_enable_gradient", "_enable_natural_gradient", "_cache_natural_delta", "_observer",
        "_restrict_subspace", "_start", "_count", "_result", "_result_square", "_total_energy", "_total_energy_square",
        "_Delta", "_EDelta", "_Deltas"
    ]

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
        self._enable_natural_gradient = False
        self._cache_natural_delta = None
        self._observer = {}
        self._restrict_subspace = None

        self._start = False

        self._count = None
        self._result = None
        self._result_square = None
        self._total_energy = None
        self._total_energy_square = None
        self._Delta = None
        self._EDelta = None
        self._Deltas = None

    def restrict_subspace(self, restrict_subspace):
        """
        Set restrict subspace for observers.

        Parameters
        ----------
        restrict_subspace
            A function return bool to restrict measure subspace.
        """
        if self._start:
            raise RuntimeError("Cannot set restrict subspace after sampling start")
        self._restrict_subspace = restrict_subspace

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
        self._result_square = {
            name: {positions: 0.0 for positions, observer in observers.items()
                  } for name, observers in self._observer.items()
        }
        self._total_energy = 0.0
        self._total_energy_square = 0.0
        self._Delta = {name: None for name in self._enable_gradient}
        self._EDelta = {name: None for name in self._enable_gradient}
        self._Deltas = []
        if self._cache_natural_delta is not None:
            os.makedirs(self._cache_natural_delta, exist_ok=True)
            with open(os.path.join(self._cache_natural_delta, str(mpi_rank)), "wb") as file:
                pass

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
                buffer.append(self._result_square[name][positions])
        buffer.append(self._total_energy)
        buffer.append(self._total_energy_square)
        buffer.append(self._count)

        buffer = np.array(buffer)
        allreduce_buffer(buffer)
        buffer = buffer.tolist()

        self._count = buffer.pop()
        self._total_energy_square = buffer.pop()
        self._total_energy = buffer.pop()
        for name, observer in reversed(self._observer.items()):
            for positions in reversed(observers):
                self._result_square[name][positions] = buffer.pop()
                self._result[name][positions] = buffer.pop()

        for name in sorted(self._enable_gradient):
            self._owner.ansatzes[name].allreduce_delta(self._Delta[name])
            self._owner.ansatzes[name].allreduce_delta(self._EDelta[name])

    def _expect_and_deviation(self, total, total_square):
        """
        Get the expect value and deviation.

        Parameters
        ----------
        total : float
            The summation of observed value.
        total_square : float
            The summation of observed value square.

        Returns
        -------
        tuple[float, float]
            The expect value and deviation.
        """
        if total == total_square == 0.0:
            return 0.0, 0.0

        N = self._count

        Eb = total / N
        E2b = total_square / N

        EV = E2b - Eb * Eb

        expect = Eb
        variance = EV / N

        if variance < 0.0:
            deviation = 0.0
        else:
            deviation = variance**0.5

        return expect, deviation

    @property
    def result(self):
        """
        Get the observer result.

        Returns
        -------
        dict[str, dict[tuple[tuple[int, int, int], ...], tuple[float, float]]]
            The observer result of each observer set name and each site positions list.
        """
        return {
            name: {
                positions: self._expect_and_deviation(self._result[name][positions],
                                                      self._result_square[name][positions])
                for positions, _ in data.items()
            } for name, data in self._observer.items()
        }

    @property
    def total_energy(self):
        """
        Get the observed energy.

        Returns
        -------
        tuple[float, float]
            The total energy.
        """
        return self._expect_and_deviation(self._total_energy, self._total_energy_square)

    @property
    def energy(self):
        """
        Get the observed energy per site.

        Returns
        -------
        tuple[float, float]
            The energy per site.
        """
        expect, deviation = self.total_energy
        site_number = self._owner.site_number
        return expect / site_number, deviation / site_number

    @property
    def gradient(self):
        """
        Get the energy gradient for every subansatz.

        Returns
        -------
        dict[str, Delta]
            The gradient for every subansatz.
        """
        energy, _ = self.total_energy
        return {
            name: 2 * self._EDelta[name] / self._count - 2 * energy * self._Delta[name] / self._count
            for name in self._enable_gradient
        }

    def delta_dot_sum(self, a, b):
        result = 0.0
        for name in self._enable_gradient:
            result += self._owner.ansatzes[name].delta_dot_sum(a[name], b[name])
        return result

    def delta_update(self, a, b):
        for name in self._enable_gradient:
            self._owner.ansatzes[name].delta_update(a[name], b[name])

    def allreduce_delta(self, a):
        requests = []
        for name in self._enable_gradient:
            requests += self._owner.ansatzes[name].iallreduce_delta(a[name])
        MPI.Request.Waitall(requests)

    def delta_scalar(self, func, *args):
        return {name: func(*(arg[name] for arg in args)) for name in self._enable_gradient}

    def _metric_mv(self, gradient, epsilon):
        """
        Product metric and delta, like matrix multiply vector. Metric is generated by Deltas and Delta.

        Parameters
        ----------
        gradient : dict[str, Delta]
            The hole tensors.
        epsilon : float
            The epsilon to avoid singularity of metric.

        Returns
        -------
        dict[str, Delta]
            The product result.
        """
        # Metric = |Deltas[s]> <Deltas[s]| reweight[s] / total_weight - |Delta> / total_weight <Delta| / total_weight
        result_1 = self.delta_scalar(lambda x1: x1 * 0, gradient)
        if self._cache_natural_delta:
            with open(os.path.join(self._cache_natural_delta, str(mpi_rank)), "rb") as file:
                for _ in self._Deltas:
                    deltas = pickle.load(file)
                    param = self.delta_dot_sum(deltas, gradient) / self._count
                    self.delta_update(result_1, self.delta_scalar(lambda x1: param * x1, deltas))
        else:
            for deltas in self._Deltas:
                param = self.delta_dot_sum(deltas, gradient) / self._count
                self.delta_update(result_1, self.delta_scalar(lambda x1: param * x1, deltas))
        self.allreduce_delta(result_1)

        delta = self.delta_scalar(lambda x1: x1 / self._count, self._Delta)
        param = self.delta_dot_sum(delta, gradient)
        result_2 = self.delta_scalar(lambda x1: x1 * param, delta)
        return self.delta_scalar(lambda x1, x2, x3: x1 - x2 + epsilon * x3, result_1, result_2, gradient)

    def natural_gradient(self, step, epsilon):
        """
        Get the energy natural gradient for every ansatz.

        Parameters
        ----------
        step : int
            conjugate gradient method step count.
        epsilon : float
            The epsilon to avoid singularity of metric.

        Returns
        -------
        dict[str, Delta]
            The gradient for every subansatz.
        """
        energy, _ = self.total_energy
        b = {
            name: 2 * self._EDelta[name] / self._count - 2 * energy * self._Delta[name] / self._count
            for name in self._enable_gradient
        }
        # A = metric
        # A x = b

        x = self.delta_scalar(lambda x1: x1 * 0, b)
        # r = b - A@x
        r = self.delta_scalar(lambda x1, x2: x1 - x2, b, self._metric_mv(x, epsilon))
        # p = r
        p = r
        for t in range(step):
            show(f"conjugate gradient step={t}")
            # alpha = (r @ r) / (p @ A @ p)
            alpha = self.delta_dot_sum(r, r) / self.delta_dot_sum(p, self._metric_mv(p, epsilon))
            # x = x + alpha * p
            x = self.delta_scalar(lambda x1, x2: x1 + alpha * x2, x, p)
            # new_r = r - alpha * A @ p
            new_r = self.delta_scalar(lambda x1, x2: x1 - alpha * x2, r, self._metric_mv(p, epsilon))
            # beta = (new_r @ new_r) / (r @ r)
            beta = self.delta_dot_sum(new_r, new_r) / self.delta_dot_sum(r, r)
            # r = new_r
            r = new_r
            # p = r + beta * p
            p = self.delta_scalar(lambda x1, x2: x1 + beta * x2, r, p)
        return x

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

    def enable_natural_gradient(self):
        """
        Enable observing natural gradient.
        """
        if self._start:
            raise RuntimeError("Cannot enable natural gradient after sampling start")
        self._enable_natural_gradient = True

    def cache_natural_delta(self, cache_natural_delta):
        """
        Set the cache folder to store deltas used in natural gradient.

        Parameters
        ----------

        cache_natural_delta : str | None
            The folder to store deltas.
        """
        if self._start:
            raise RuntimeError("Cannot set natural delta cache folder after sampling start")
        self._cache_natural_delta = cache_natural_delta

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
        configuration : list[list[dict[int, EdgePoint]]]
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
                current_configuration = tuple(configuration[l1][l2][orbit] for l1, l2, orbit in positions)
                element_pool = tensor_element(observer)
                if current_configuration not in element_pool:
                    continue
                for other_configuration, observer_shrinked in element_pool[current_configuration].items():
                    if self._restrict_subspace is not None:
                        replacement = {positions[i]: other_configuration[i] for i in range(body)}
                        if not self._restrict_subspace(configuration, replacement):
                            continue
                    new_configuration = [[{
                        orbit: configuration[l1][l2][orbit] for orbit in owner.physics_edges[l1, l2]
                    } for l2 in range(owner.L2)] for l1 in range(owner.L1)]
                    for i, [l1, l2, orbit] in enumerate(positions):
                        new_configuration[l1][l2][orbit] = other_configuration[i]
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
                self._result_square[name][positions] += to_save * to_save
                if name == "energy":
                    Es += total_value
            if name == "energy":
                to_save = Es.real
                self._total_energy += to_save
                self._total_energy_square += to_save * to_save
                if self._owner.Tensor.is_real:
                    Es = Es.real
                else:
                    Es = Es.conjugate()
                deltas = {}
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
                    if self._enable_natural_gradient:
                        deltas[ansatz_name] = this_delta
                if self._enable_natural_gradient:
                    if self._cache_natural_delta:
                        with open(os.path.join(self._cache_natural_delta, str(mpi_rank)), "ab") as file:
                            pickle.dump(deltas, file)
                        self._Deltas.append(None)
                    else:
                        self._Deltas.append(deltas)
