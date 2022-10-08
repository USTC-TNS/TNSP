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
import numpy as np
from ..common_toolkit import allreduce_buffer, allreduce_iterator_buffer, mpi_rank, mpi_comm, show, showln, pickle
from ..tensor_element import tensor_element
from .state import Configuration


class Observer:
    """
    Helper type for Observing the ansatz product state.
    """

    __slots__ = [
        "owner", "_observer", "_enable_gradient", "_enable_natural", "_cache_natural_delta", "_restrict_subspace",
        "_start", "_result", "_result_square", "_result_reweight", "_count", "_total_weight", "_total_weight_square",
        "_total_log_ws", "_total_energy", "_total_energy_square", "_total_energy_reweight", "_Delta", "_EDelta",
        "_Deltas", "_max_batch_size"
    ]

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
        self._total_log_ws = 0.0
        self._total_energy = 0.0
        self._total_energy_square = 0.0
        self._total_energy_reweight = 0.0
        if self._enable_gradient:
            self._Delta = None
            self._EDelta = None
            if self._enable_natural:
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
        buffer.append(self._total_log_ws)
        buffer.append(self._total_energy)
        buffer.append(self._total_energy_square)
        buffer.append(self._total_energy_reweight)

        buffer = np.array(buffer)
        allreduce_buffer(buffer)
        buffer = buffer.tolist()

        self._total_energy_reweight = buffer.pop()
        self._total_energy_square = buffer.pop()
        self._total_energy = buffer.pop()
        self._total_log_ws = buffer.pop()
        self._total_weight_square = buffer.pop()
        self._total_weight = buffer.pop()
        self._count = buffer.pop()
        for name, observers in reversed(self._observer.items()):
            for positions in reversed(observers):
                self._result_reweight[name][positions] = buffer.pop()
                self._result_square[name][positions] = buffer.pop()
                self._result[name][positions] = buffer.pop()

        if self._enable_gradient:
            allreduce_iterator_buffer(self.owner.ansatz.buffers(self._Delta))
            allreduce_iterator_buffer(self.owner.ansatz.buffers(self._EDelta))

    def __init__(
        self,
        owner,
        *,
        observer_set=None,
        enable_energy=False,
        enable_gradient=False,
        enable_natural_gradient=False,
        cache_natural_delta=None,
        restrict_subspace=None,
        max_batch_size=None,
    ):
        """
        Create observer object for the given ansatz product state.

        Parameters
        ----------
        owner : AnsatzProductState
            The owner of this obsever object.
        observer_set : dict[str, dict[tuple[tuple[int, int, int], ...], Tensor]], optional
            The given observers to observe.
        enable_energy : bool, optional
            Enable observing the energy.
        enable_gradient : bool, optional
            Enable calculating the gradient.
        enable_natural_gradient : bool, optional
            Enable calculating the natural gradient.
        cache_natural_delta : str, optional
            The folder name to cache deltas used in natural gradient.
        restrict_subspace : optional
            A function return bool to restrict sampling subspace.
        max_batch_size : int, optional
            The max batch size when calculating wss
        """
        self.owner = owner

        self._observer = {}
        self._enable_gradient = False
        self._enable_natural = False
        self._cache_natural_delta = None
        self._restrict_subspace = None

        self._start = False

        self._result = None
        self._result_square = None
        self._result_reweight = None
        self._count = None
        self._total_weight = None
        self._total_weight_square = None
        self._total_log_ws = None
        self._total_energy = None
        self._total_energy_square = None
        self._total_energy_reweight = None

        self._Delta = None
        self._EDelta = None
        self._Deltas = None

        self._max_batch_size = None

        if observer_set is not None:
            self._observer = observer_set
        if enable_energy:
            self.add_energy()
        if enable_gradient:
            self.enable_gradient()
        if enable_natural_gradient:
            self.enable_natural_gradient()
        self.cache_natural_delta(cache_natural_delta)
        self.restrict_subspace(restrict_subspace)
        self.set_max_batch_size(max_batch_size)

    def set_max_batch_size(self, max_batch_size):
        """
        Set the max batch size when calculating wss.

        Parameters
        ----------
        max_batch_size : int, optional
            The max batch size when calculating wss.
        """
        self._max_batch_size = max_batch_size

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
            raise RuntimeError("Canot add observer after sampling start")
        for positions, observer in observers.items():
            if not isinstance(observer, self.owner.Tensor):
                raise TypeError("Wrong observer type")
        self._observer[name] = observers

    def add_energy(self):
        """
        Add energy as an observer.
        """
        self.add_observer("energy", self.owner._hamiltonians)

    def enable_gradient(self):
        """
        Enable observing gradient.
        """
        if self._start:
            raise RuntimeError("Cannot enable gradient after sampling start")
        if "energy" not in self._observer:
            self.add_energy()
        self._enable_gradient = True

    def enable_natural_gradient(self):
        """
        Enable observing natural gradient.
        """
        if self._start:
            raise RuntimeError("Cannot enable natural gradient after sampling start")
        if not self._enable_gradient:
            self.enable_gradient()
        self._enable_natural = True

    def __call__(self, sampling_result):
        """
        Collect observer value from sampling result.

        Parameters
        ----------
        sampling_result : list[tuple[float, Configuration]]
            The sampling results, it contains the sampled weight used in importance sampling and the configuration.
        """
        self._count += len(sampling_result)
        # ws is |s|psi>
        # delta is |s|partial_x psi>
        ws, delta = self.owner.ansatz.weight_and_delta(
            [configuration for _, configuration in sampling_result],
            self._enable_gradient,
        )
        reweights = [
            np.linalg.norm(ws[chain])**2 / possibility for chain, [possibility, _] in enumerate(sampling_result)
        ]  # <psi|s|psi> / p(s)

        # find all wss
        # list[Configuration]
        configuration_list = []
        # chain -> name -> positions -> s' -> (index in configuration_list, H_{s,s'})
        configuration_data = [{} for _ in sampling_result]
        for chain, [possibility, configuration] in enumerate(sampling_result):
            self._total_weight += reweights[chain]
            self._total_weight_square += reweights[chain] * reweights[chain]
            self._total_log_ws += np.log(np.abs(complex(ws[chain])))

            for name, observers in self._observer.items():
                configuration_data_name = configuration_data[chain][name] = {}

                for positions, observer in observers.items():
                    configuration_data_name_positions = configuration_data_name[positions] = {}

                    body = len(positions)
                    positions_configuration = tuple(configuration[l1l2o] for l1l2o in positions)
                    element_pool = tensor_element(observer)
                    if positions_configuration not in element_pool:
                        continue

                    for positions_configuration_s, observer_shrinked in element_pool[positions_configuration].items():
                        # observer_shrinked is |s'|H|s|
                        if self._restrict_subspace is not None:
                            replacement = {positions[i]: positions_configuration_s[i] for i in range(body)}
                            if not self._restrict_subspace(configuration, replacement):
                                continue
                        new_configuration = configuration.copy()
                        for i, [l1, l2, orbit] in enumerate(positions):
                            new_configuration[l1, l2, orbit] = positions_configuration_s[i]

                        configuration_data_name_positions[positions_configuration_s] = (len(configuration_list),
                                                                                        observer_shrinked)
                        configuration_list.append(new_configuration)
        # measure
        if self._max_batch_size is None:
            wss_list, _ = self.owner.ansatz.weight_and_delta(configuration_list, False)
        else:
            wss_list = []
            for i in range((len(configuration_list) - 1) // self._max_batch_size + 1):
                wss_list_this, _ = self.owner.ansatz.weight_and_delta(
                    configuration_list[i * self._max_batch_size:(i + 1) * self._max_batch_size], False)
                wss_list += wss_list_this
        for chain, reweight in enumerate(reweights):
            for name, configuration_data_name in configuration_data[chain].items():
                if name == "energy":
                    Es = 0.0
                for positions, configuration_data_name_positions in configuration_data_name.items():
                    total_value = 0
                    for _, [index, hamiltonian_term] in configuration_data_name_positions.items():
                        # |s'|psi>
                        wss = wss_list[index]
                        # <psi|s'|H|s|psi> / <psi|s|psi>
                        value = (ws[chain] * hamiltonian_term * wss.conjugate()) / (ws[chain].conjugate() * ws[chain])
                        total_value += complex(value)
                    # total_value is sum_s' <psi|s'|H|s|psi> / <psi|s|psi>
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

                    if self._enable_gradient:
                        # delta is |s|partial_x psi>
                        # holes is <psi|s|partial_x psi> / <psi|s|psi>
                        holes = self.owner.ansatz.recovery_real(
                            (ws[chain].conjugate() * delta[chain]) / (ws[chain].conjugate() * ws[chain]))
                        if self.owner.Tensor.is_real:
                            Es = Es.real
                        # this_delta is r(s) <psi|s|partial_x psi> / <psi|s|psi>
                        this_delta = reweight * holes
                        this_edelta = Es * this_delta
                        if self._Delta is None:
                            self._Delta = this_delta
                        else:
                            self._Delta += this_delta
                        if self._EDelta is None:
                            self._EDelta = this_edelta
                        else:
                            self._EDelta += this_edelta
                        if self._enable_natural:
                            if self._cache_natural_delta:
                                with open(os.path.join(self._cache_natural_delta, str(mpi_rank)), "ab") as file:
                                    pickle.dump(holes, file)
                                self._Deltas.append((reweight, None))
                            else:
                                self._Deltas.append((reweight, holes))

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
                                           self._result_reweight[name][pisitions]) for positions in data
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
        site_number = self.owner.site_number
        return expect / site_number, deviation / site_number

    @property
    def gradient(self):
        """
        Get the energy gradient for the whole ansatz.

        Returns
        -------
        Tensors
            The gradient for the whole ansatz.
        """
        energy, _ = self.total_energy
        b = (2 * (np.array(self._EDelta, dtype=object) / self._total_weight) - 2 * energy *
             (np.array(self._Delta, dtype=object) / self._total_weight))
        return self.owner.state_conjugate(b)

    def _trace_metric(self):
        """
        Get the trace of metric used in natural gradient.

        Returns
        -------
        float
            The trace of metric.
        """
        result = 0.0
        for reweight, deltas in self._weights_and_deltas():
            result += self.owner.state_prod_sum(self.owner.state_conjugate(deltas),
                                                deltas) * reweight / self._total_weight
        result = mpi_comm.allreduce(result)

        result -= self.owner.state_prod_sum(self.owner.state_conjugate(self._Delta),
                                            self._Delta) / (self._total_weight * self._total_weight)

        return result

    def _weights_and_deltas(self):
        """
        Get the series of weights and deltas.

        Yields
        ------
        tuple[float, dict[str, Delta]]
            The weight and delta.
        """
        if self._cache_natural_delta:
            with open(os.path.join(self._cache_natural_delta, str(mpi_rank)), "rb") as file:
                for reweight, _ in self._Deltas:
                    deltas = pickle.load(file)
                    yield reweight, deltas
        else:
            for reweight, deltas in self._Deltas:
                yield reweight, deltas

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
        result_1 = gradient * 0
        for reweight, deltas in self._weights_and_deltas():
            param = self.owner.state_prod_sum(deltas, gradient) * reweight / self._total_weight
            result_1 += param * self.owner.state_conjugate(deltas)
        self.owner.allreduce_state(result_1)

        param = self.owner.state_prod_sum(self._Delta, gradient) / (self._total_weight * self._total_weight)
        result_2 = self.owner.state_conjugate(self._Delta) * param
        return result_1 - result_2 + epsilon * gradient

    def natural_gradient(self, step, error, epsilon):
        """
        Get the energy natural gradient for every ansatz.

        Parameters
        ----------
        step : int
            conjugate gradient method step count.
        error : float
            conjugate gradient method expected error.
        epsilon : float
            The epsilon to avoid singularity of metric.

        Returns
        -------
        Tensors
            The gradient for the whole ansatz.
        """
        show("calculating natural gradient")
        b = self.gradient
        b_square = self.owner.state_prod_sum(self.owner.state_conjugate(b), b).real
        # A = metric
        # A x = b

        tr = self._trace_metric()
        n = self.owner.ansatz.element_count(b)
        relative_epsilon = epsilon * tr / n

        x = b * 0
        # r = b - A@x
        r = b - self._metric_mv(x, relative_epsilon)
        r_square = self.owner.state_prod_sum(self.owner.state_conjugate(r), r).real
        # p = r
        p = r
        # loop
        t = 0
        while True:
            if t == step:
                break
            if error != 0.0:
                if error**2 > r_square / b_square:
                    break
            show(f"conjugate gradient step={t} r^2/b^2={r_square/b_square}")
            # alpha = (r @ r) / (p @ A @ p)
            alpha = (
                self.owner.state_prod_sum(self.owner.state_conjugate(r), r).real /
                self.owner.state_prod_sum(self.owner.state_conjugate(p), self._metric_mv(p, relative_epsilon)).real)
            # x = x + alpha * p
            x = x + alpha * p
            # new_r = r - alpha * A @ p
            new_r = r - alpha * self._metric_mv(p, relative_epsilon)
            new_r_square = self.owner.state_prod_sum(self.owner.state_conjugate(new_r), new_r).real
            # beta = (new_r @ new_r) / (r @ r)
            beta = new_r_square / r_square
            # r = new_r
            r = new_r
            r_square = new_r_square
            # p = r + beta * p
            p = r + beta * p
            t += 1
        showln(f"calculate natural gradient done step={t} r^2/b^2={r_square/b_square}")
        return x

    def normalize_state(self):
        """
        Normalize the state
        """
        mean_log_ws = self._total_log_ws / self._count
        self.owner.ansatz.normalize_ansatz(mean_log_ws)
