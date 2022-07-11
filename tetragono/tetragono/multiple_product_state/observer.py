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
import pandas as pd
from ..sampling_tools.tensor_element import tensor_element
from ..common_toolkit import allreduce_buffer, mpi_rank, mpi_comm, show, MPI


class Observer:

    __slots__ = [
        "_owner", "_observer", "_enable_gradient", "_enable_natural_gradient", "_cache_natural_delta",
        "_restrict_subspace", "_start", "_result", "_result_square", "_result_reweight", "_count", "_total_weight",
        "_total_weight_square", "_total_energy", "_total_energy_square", "_total_energy_reweight", "_Delta", "_EDelta",
        "_Deltas"
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

        self._observer = {}
        self._enable_gradient = set()
        self._enable_natural_gradient = False
        self._cache_natural_delta = None
        self._restrict_subspace = None

        self._start = False

        self._result = None
        self._result_square = None
        self._result_reweight = None
        self._count = None
        self._total_weight = None
        self._total_weight_square = None
        self._total_energy = None
        self._total_energy_square = None
        self._total_energy_reweight = None

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
        self._result = {
            name: {positions: 0.0 for positions, observer in observers.items()
                  } for name, observers in self._observer.items()
        }
        self._result_square = {
            name: {positions: 0.0 for positions, observer in observers.items()
                  } for name, observers in self._observer.items()
        }
        self._result_reweight = {
            name: {positions: 0.0 for positions, observer in observers.items()
                  } for name, observers in self._observer.items()
        }
        self._count = 0
        self._total_weight = 0.0
        self._total_weight_square = 0.0
        self._total_energy = 0.0
        self._total_energy_square = 0.0
        self._total_energy_reweight = 0.0
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
                buffer.append(self._result_reweight[name][positions])
        buffer.append(self._count)
        buffer.append(self._total_weight)
        buffer.append(self._total_weight_square)
        buffer.append(self._total_energy)
        buffer.append(self._total_energy_square)
        buffer.append(self._total_energy_reweight)

        buffer = np.array(buffer)
        allreduce_buffer(buffer)
        buffer = buffer.tolist()

        self._total_energy_reweight = buffer.pop()
        self._total_energy_square = buffer.pop()
        self._total_energy = buffer.pop()
        self._total_weight_square = buffer.pop()
        self._total_weight = buffer.pop()
        self._count = buffer.pop()
        for name, observer in reversed(self._observer.items()):
            for positions in reversed(observers):
                self._result_reweight[name][positions] = buffer.pop()
                self._result_square[name][positions] = buffer.pop()
                self._result[name][positions] = buffer.pop()

        for name in sorted(self._enable_gradient):
            self._owner.ansatzes[name].allreduce_delta(self._Delta[name])
            self._owner.ansatzes[name].allreduce_delta(self._EDelta[name])

    def _expect_and_deviation(self, total, total_square, total_reweight):
        """
        Get the expect value and deviation.

        Parameters
        ----------
        total : float
            The summation of observed value.
        total_square : float
            The summation of observed value square.
        total_reweight : float
            The summation of observed value with reweight.

        Returns
        -------
        tuple[float, float]
            The expect value and deviation.
        """
        if total == 0.0 or self._total_weight == 0.0:
            return 0.0, 0.0

        N = self._count

        Eb = total / N
        E2b = total_square / N
        EWb = total_reweight / N
        Wb = self._total_weight / N
        W2b = self._total_weight_square / N

        EV = E2b - Eb * Eb
        WV = W2b - Wb * Wb
        EWC = EWb - Eb * Wb

        expect = EWb / Wb
        # Derivation calculation
        # expect   = sumEW / sumW
        # variance = sum [W / sumW]^2 Var(E) +
        #            sum [E / sumW - expect / sumW]^2 Var(W) +
        #            sum [W / sumW][E / sumW - expect / sumW] Cov(E,W)
        #          = W2b / (Wb^2 N) Var(E) +
        #            (E2b + expect^2 - 2 expect Eb) / (Wb^2 N) Var(W) +
        #            (EWb - expect Wb) / (Wb^2 N) Cov(E,W)
        #          = [W2b EV + (E2b + expect^2 - 2 expect Eb) WV + (EWb - expect Wb) EWC] / (Wb^2 N)
        variance = (W2b * EV + (E2b + expect * expect - 2 * expect * Eb) * WV +
                    (EWb - expect * Wb) * EWC) / (Wb * Wb * N)
        if variance < 0.0:
            # When total summate several same values, numeric error will lead variance < 0
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
                positions:
                self._expect_and_deviation(self._result[name][positions], self._result_square[name][positions],
                                           self._result_reweight[name][pisitions]) for positions, _ in data.items()
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
        return self._expect_and_deviation(self._total_energy, self._total_energy_square, self._total_energy_reweight)

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
        b = 2 * (pd.Series(self._EDelta) / self._total_weight) - 2 * energy * (pd.Series(self._Delta) /
                                                                               self._total_weight)
        return b

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
        for name in sorted(self._enable_gradient):
            requests += self._owner.ansatzes[name].iallreduce_delta(a[name])
        MPI.Request.Waitall(requests)

    def _trace_metric(self):
        """
        Get the trace of metric used in natural gradient.

        Returns
        -------
        float
            The trace of metric.
        """
        # Metric = |Deltas[s]> <Deltas[s]| reweight[s] / total_weight - |Delta> / total_weight <Delta| / total_weight
        result = 0.0
        if self._cache_natural_delta:
            with open(os.path.join(self._cache_natural_delta, str(mpi_rank)), "rb") as file:
                for reweight, _ in self._Deltas:
                    deltas = pickle.load(file)
                    result += self.delta_dot_sum(deltas, deltas) * reweight / self._total_weight
        else:
            for reweight, deltas in self._Deltas:
                result += self.delta_dot_sum(deltas, deltas) * reweight / self._total_weight
        result = mpi_comm.allreduce(result)

        delta = pd.Series(self._Delta) / self._total_weight
        result -= self.delta_dot_sum(delta, delta)

        return result

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
        result_1 = gradient * 0
        if self._cache_natural_delta:
            with open(os.path.join(self._cache_natural_delta, str(mpi_rank)), "rb") as file:
                for reweight, _ in self._Deltas:
                    deltas = pickle.load(file)
                    param = self.delta_dot_sum(deltas, gradient) * reweight / self._total_weight
                    self.delta_update(result_1, param * pd.Series(deltas))
        else:
            for reweight, deltas in self._Deltas:
                param = self.delta_dot_sum(deltas, gradient) * reweight / self._total_weight
                self.delta_update(result_1, param * pd.Series(deltas))
        self.allreduce_delta(result_1)

        delta = pd.Series(self._Delta) / self._total_weight
        param = self.delta_dot_sum(delta, gradient)
        result_2 = delta * param
        return result_1 - result_2 + epsilon * gradient

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
        b = 2 * (pd.Series(self._EDelta) / self._total_weight) - 2 * energy * (pd.Series(self._Delta) /
                                                                               self._total_weight)
        # A = metric
        # A x = b

        tr = self._trace_metric()
        n = sum(self._owner.ansatzes[name].param_count(delta) for name, delta in b.items())
        relative_epsilon = epsilon * tr / n

        x = b * 0
        # r = b - A@x
        r = b - self._metric_mv(x, relative_epsilon)
        # p = r
        p = r
        for t in range(step):
            show(f"conjugate gradient step={t}")
            # alpha = (r @ r) / (p @ A @ p)
            alpha = self.delta_dot_sum(r, r) / self.delta_dot_sum(p, self._metric_mv(p, relative_epsilon))
            # x = x + alpha * p
            x = x + alpha * p
            # new_r = r - alpha * A @ p
            new_r = r - alpha * self._metric_mv(p, relative_epsilon)
            # beta = (new_r @ new_r) / (r @ r)
            beta = self.delta_dot_sum(new_r, new_r) / self.delta_dot_sum(r, r)
            # r = new_r
            r = new_r
            # p = r + beta * p
            p = r + beta * p
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

    def __call__(self, possibility, configuration):
        """
        Collect observer value from current configuration.

        Parameters
        ----------
        possibility : float
            the sampled weight used in importance sampling.
        configuration : list[list[dict[int, EdgePoint]]]
            The current configuration.
        """
        owner = self._owner
        self._count += 1
        [ws], [delta] = owner.weight_and_delta([configuration], self._enable_gradient)
        reweight = ws**2 / possibility
        self._total_weight += reweight
        self._total_weight_square += reweight * reweight
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
                self._result_reweight[name][positions] += to_save * reweight
                if name == "energy":
                    Es += total_value
            if name == "energy":
                to_save = Es.real
                self._total_energy += to_save
                self._total_energy_square += to_save * to_save
                self._total_energy_reweight += to_save * reweight
                if self._owner.Tensor.is_real:
                    Es = Es.real
                else:
                    Es = Es.conjugate()
                deltas = {}
                for ansatz_name in self._enable_gradient:
                    this_delta = reweight * delta[ansatz_name] / ws
                    if self._Delta[ansatz_name] is None:
                        self._Delta[ansatz_name] = this_delta
                    else:
                        self._Delta[ansatz_name] += this_delta
                    if self._EDelta[ansatz_name] is None:
                        self._EDelta[ansatz_name] = Es * this_delta
                    else:
                        self._EDelta[ansatz_name] += Es * this_delta
                    if self._enable_natural_gradient:
                        deltas[ansatz_name] = delta[ansatz_name] / ws
                if self._enable_natural_gradient:
                    if self._cache_natural_delta:
                        with open(os.path.join(self._cache_natural_delta, str(mpi_rank)), "ab") as file:
                            pickle.dump(deltas, file)
                        self._Deltas.append((reweight, None))
                    else:
                        self._Deltas.append((reweight, deltas))
