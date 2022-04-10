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

import os
import pickle
import numpy as np
from ..sampling_lattice import ConfigurationPool
from ..common_toolkit import (show, allreduce_lattice_buffer, allreduce_buffer, lattice_update, lattice_dot_sum,
                              lattice_conjugate, mpi_rank)
from .tensor_element import tensor_element


class Observer():
    """
    Helper type for Observing the sampling lattice.
    """

    __slots__ = [
        "_owner", "_observer", "_enable_gradient", "_enable_natural", "_cache_natural_delta", "_cache_configuration",
        "_restrict_subspace", "_start", "_result", "_result_square", "_result_reweight", "_count", "_total_weight",
        "_total_weight_square", "_total_log_ws", "_total_energy", "_total_energy_square", "_total_energy_reweight",
        "_Delta", "_EDelta", "_Deltas", "_pool"
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
            self._Delta = [[self._owner[l1, l2].same_shape().conjugate().zero()
                            for l2 in range(self._owner.L2)]
                           for l1 in range(self._owner.L1)]
            self._EDelta = [[self._owner[l1, l2].same_shape().conjugate().zero()
                             for l2 in range(self._owner.L2)]
                            for l1 in range(self._owner.L1)]
            if self._enable_natural:
                self._Deltas = []
        if self._cache_natural_delta is not None:
            os.makedirs(self._cache_natural_delta, exist_ok=True)
            with open(os.path.join(self._cache_natural_delta, str(mpi_rank)), "wb") as file:
                pass
        if self._cache_configuration:
            self._create_cache_configuration()

    def _create_cache_configuration(self):
        """
        Create or refresh configuration cache pool.
        """
        self._pool = ConfigurationPool(self._owner)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit sampling loop, reduce observed values, used when running with multiple processes. reduce all values
        collected except Deltas, which is handled specially.
        """
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
            allreduce_lattice_buffer(self._Delta)
            allreduce_lattice_buffer(self._EDelta)

    def __init__(
        self,
        owner,
        *,
        observer_set=None,
        enable_energy=False,
        enable_gradient=False,
        enable_natural_gradient=False,
        cache_natural_delta=None,
        cache_configuration=False,
        restrict_subspace=None,
    ):
        """
        Create observer object for the given sampling lattice.

        Parameters
        ----------
        owner : SamplingLattice
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
        cache_configuration : bool, optional
            Enable cache the configuration during observing.
        restrict_subspace, optional
            A function return bool to restrict sampling subspace.
        """
        self._owner = owner
        self._observer = {}  # dict[str, dict[tuple[tuple[int, int, int], ...], Tensor]]
        self._enable_gradient = False
        self._enable_natural = False
        self._cache_natural_delta = None
        self._cache_configuration = False
        self._restrict_subspace = None

        self._start = False

        self._result = None  # dict[str, dict[tuple[tuple[int, int, int], ...], float]]
        self._result_square = None
        self._result_reweight = None
        self._count = None  # int
        self._total_weight = None  # float
        self._total_weight_square = None
        self._total_log_ws = None
        self._total_energy = None
        self._total_energy_square = None
        self._total_energy_reweight = None

        self._Delta = None  # list[list[Tensor]]
        self._EDelta = None  # list[list[Tensor]]
        self._Deltas = None

        self._pool = None

        if observer_set is not None:
            self._observer = observer_set

        if enable_energy:
            self.add_energy()
        if enable_gradient:
            self.enable_gradient()
        if enable_natural_gradient:
            self.enable_natural_gradient()
        self.cache_natural_delta(cache_natural_delta)
        self.cache_configuration(cache_configuration)
        self._restrict_subspace = restrict_subspace

    def cache_natural_delta(self, cache_natural_delta):
        """
        Set the cache folder to store deltas used in natural gradient.

        Parameters
        cache_natural_delta : str | None
            The folder to store deltas.
        """
        if self._start:
            raise RuntimeError("Cannot set natural delta cache folder after sampling start")
        self._cache_natural_delta = cache_natural_delta

    def cache_configuration(self, cache_configuration):
        """
        Enable caching the configurations into one pool.

        Parameters
        ----------
        cache_configuration : bool | str
            The cache clean strategy of configuration cache.
        """
        if self._start:
            raise RuntimeError("Cannot enable caching after sampling start")
        self._cache_configuration = cache_configuration

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
        for positions, observer in observers.items():
            if not isinstance(observer, self._owner.Tensor):
                raise TypeError("Wrong observer type")
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

    def __call__(self, possibility, configuration):
        """
        Collect observer value from current configuration, the sampling should have distribution based on
        $|\langle\psi s\rangle|^2$, If it is not, a reweight argument should be passed with a non-one float number.

        Parameters
        ----------
        possibility : float
            the sampled weight used in importance sampling.
        configuration : Configuration
            The configuration system of the lattice.
        """
        if self._cache_configuration:
            if self._cache_configuration == "drop":
                self._create_cache_configuration()
            configuration = self._pool.add(configuration)
        self._count += 1
        ws = configuration.hole(())  # ws is a tensor
        if ws.norm_num() == 0:
            return
        reweight = ws.norm_2()**2 / possibility
        self._total_weight += reweight
        self._total_weight_square += reweight * reweight
        self._total_log_ws += np.log(np.abs(complex(ws)))
        inv_ws_conj = ws / (ws.norm_2()**2)
        inv_ws = inv_ws_conj.conjugate()
        all_name = {("T", "T")} | {(f"P_{l1}_{l2}_{orbit}", f"P_{l1}_{l2}_{orbit}") for l1 in range(self._owner.L1)
                                   for l2 in range(self._owner.L2)
                                   for orbit, edge in self._owner.physics_edges[l1, l2].items()}
        for name, observers in self._observer.items():
            if name == "energy":
                Es = 0.0
            calculating_gradient = name == "energy" and self._enable_gradient
            for positions, observer in observers.items():
                body = observer.rank // 2
                current_configuration = tuple(configuration[positions[i]] for i in range(body))
                element_pool = tensor_element(observer)
                if current_configuration not in element_pool:
                    continue
                total_value = 0
                physics_names = [f"P_{positions[i][0]}_{positions[i][1]}_{positions[i][2]}" for i in range(body)]
                for other_configuration, observer_shrinked in element_pool[current_configuration].items():
                    replacement = {positions[i]: other_configuration[i] for i in range(body)}
                    if self._restrict_subspace is not None:
                        if not self._restrict_subspace(configuration, replacement):
                            continue
                    if self._cache_configuration:
                        wss = self._pool.wss(configuration, replacement)
                    else:
                        wss = configuration.replace(replacement)
                        if wss is None:
                            raise NotImplementedError(
                                "not implemented replace style, set cache_configuration to True to calculate it")

                    if wss.norm_num() == 0:
                        continue
                    value = inv_ws.contract(observer_shrinked.conjugate(),
                                            {(physics_names[i], f"I{i}") for i in range(body)}).edge_rename({
                                                f"O{i}": physics_names[i] for i in range(body)
                                            }).contract(wss, all_name)
                    total_value += complex(value)
                to_save = total_value.real
                self._result[name][positions] += to_save
                self._result_square[name][positions] += to_save * to_save
                self._result_reweight[name][positions] += to_save * reweight
                if name == "energy":
                    Es += total_value  # Es maybe complex
            if name == "energy":
                to_save = Es.real
                self._total_energy += to_save
                self._total_energy_square += to_save * to_save
                self._total_energy_reweight += to_save * reweight
            if calculating_gradient:
                holes = configuration.holes()
                if self._owner.Tensor.is_real:
                    Es = Es.real
                else:
                    Es = Es.conjugate()
                for l1 in range(self._owner.L1):
                    for l2 in range(self._owner.L2):
                        hole = holes[l1][l2] * reweight
                        self._Delta[l1][l2] += hole
                        self._EDelta[l1][l2] += Es * hole
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
        if total == total_square == total_reweight == 0.0:
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
                                           self._result_reweight[name][positions]) for positions, _ in data.items()
            } for name, data in self._observer.items()
        }

    @property
    def total_energy(self):
        """
        Get the observed energy.

        Returns
        -------
        tuple[float, float]
            The energy per site.
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
        Get the energy gradient for every tensor.

        Returns
        -------
        list[list[Tensor]]
            The gradient for every tensor.
        """
        energy, _ = self.total_energy
        b = 2 * (np.array(self._EDelta) / self._total_weight) - 2 * energy * (np.array(self._Delta) /
                                                                              self._total_weight)
        return lattice_conjugate(b)

    def _metric_mv(self, gradient, epsilon):
        """
        Product metric tensors and hole tensors, like matrix multiply vector. Metric is generated by Deltas and Delta.

        Parameters
        ----------
        gradient : list[list[Tensor]]
            The hole tensors.
        epsilon : float
            The epsilon to avoid singularity of metric.

        Returns
        -------
        list[list[Tensor]]
            The product result.
        """
        # Metric = |Deltas[s]> <Deltas[s]| reweight[s] / total_weight - |Delta> / total_weight <Delta| / total_weight
        result_1 = np.array(
            [[self._Delta[l1][l2].same_shape().zero() for l2 in range(self._owner.L2)] for l1 in range(self._owner.L1)])
        if self._cache_natural_delta:
            with open(os.path.join(self._cache_natural_delta, str(mpi_rank)), "rb") as file:
                for reweight, _ in self._Deltas:
                    deltas = pickle.load(file)
                    param = lattice_dot_sum(deltas, gradient) * reweight / self._total_weight
                    lattice_update(result_1, param * np.array(deltas))
        else:
            for reweight, deltas in self._Deltas:
                param = lattice_dot_sum(deltas, gradient) * reweight / self._total_weight
                lattice_update(result_1, param * np.array(deltas))
        allreduce_lattice_buffer(result_1)

        delta = np.array(self._Delta) / self._total_weight
        param = lattice_dot_sum(delta, gradient)
        result_2 = delta * param
        return result_1 - result_2 + epsilon * gradient

    def natural_gradient(self, step, epsilon):
        """
        Get the energy natural gradient for every tensor.

        Parameters
        ----------
        step : int
            conjugate gradient method step count.
        epsilon : float
            The epsilon to avoid singularity of metric.

        Returns
        -------
        list[list[Tensor]]
            The gradient for every tensor.
        """
        energy, _ = self.total_energy
        b = 2 * (np.array(self._EDelta) / self._total_weight) - 2 * energy * (np.array(self._Delta) /
                                                                              self._total_weight)
        # A = metric
        # A x = b

        x = np.array([[t.same_shape().zero() for t in row] for row in b])
        # r = b - A@x
        r = b - self._metric_mv(x, epsilon)
        # p = r
        p = r
        for t in range(step):
            show(f"conjugate gradient step={t}")
            # alpha = (r @ r) / (p @ A @ p)
            alpha = lattice_dot_sum(r, r) / lattice_dot_sum(p, self._metric_mv(p, epsilon))
            # x = x + alpha * p
            x = x + alpha * p
            # new_r = r - alpha * A @ p
            new_r = r - alpha * self._metric_mv(p, epsilon)
            # beta = (new_r @ new_r) / (r @ r)
            beta = lattice_dot_sum(new_r, new_r) / lattice_dot_sum(r, r)
            # r = new_r
            r = new_r
            # p = r + beta * p
            p = r + beta * p
        return lattice_conjugate(x)

    def normalize_lattice(self):
        mean_log_ws = self._total_log_ws / self._count
        # Here it should use tensor number, not site number
        param = np.exp(mean_log_ws / (self._owner.L1 * self._owner.L2))
        self._owner._lattice /= param
