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

import numpy as np
from ..sampling_lattice import ConfigurationPool
from ..common_variable import show, allreduce_lattice_buffer, allreduce_buffer
from .tensor_element import tensor_element


class Observer():
    """
    Helper type for Observing the sampling lattice.
    """

    __slots__ = [
        "_owner", "_enable_gradient", "_enable_natural", "_start", "_observer", "_result", "_result_square",
        "_result_reweight", "_count", "_total_weight", "_total_weight_square", "_total_energy", "_total_energy_square",
        "_total_energy_reweight", "_Delta", "_EDelta", "_Deltas", "_cache_configuration", "_pool", "_restrict_subspace"
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
        for name, observers in reversed(self._observer.items()):
            for positions in reversed(observers):
                self._result_reweight[name][positions] = buffer.pop()
                self._result_square[name][positions] = buffer.pop()
                self._result[name][positions] = buffer.pop()

        if self._enable_gradient:
            allreduce_lattice_buffer(self._Delta)
            allreduce_lattice_buffer(self._EDelta)

    def __init__(self, owner, restrict_subspace):
        """
        Create observer object for the given sampling lattice.

        Parameters
        ----------
        owner : SamplingLattice
            The owner of this obsever object.
        restrict_subspace
            A function return bool to restrict sampling subspace.
        """
        self._owner = owner
        self._enable_gradient = False
        self._enable_natural = False
        self._start = False
        self._observer = {}  # dict[str, dict[tuple[tuple[int, int, int], ...], Tensor]]

        self._result = None  # dict[str, dict[tuple[tuple[int, int, int], ...], float]]
        self._result_square = None
        self._result_reweight = None
        self._count = None  # int
        self._total_weight = None  # float
        self._total_weight_square = None
        self._total_energy = None
        self._total_energy_square = None
        self._total_energy_reweight = None

        self._Delta = None  # list[list[Tensor]]
        self._EDelta = None  # list[list[Tensor]]
        self._Deltas = None

        self._cache_configuration = False
        self._pool = None

        self._restrict_subspace = restrict_subspace

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
        reweight = configuration.hole(()).norm_2()**2 / possibility
        self._count += 1
        self._total_weight += reweight
        self._total_weight_square += reweight * reweight
        ws = configuration.hole(())  # ws is a tensor
        if ws.norm_num() == 0:
            return
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
                    if self._cache_configuration:
                        self._Deltas.append((reweight, holes, configuration))
                    else:
                        self._Deltas.append((reweight, holes, None))

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
        return self._lattice_map(lambda x1, x2: 2 * (x1 / self._total_weight) - 2 * energy * (x2 / self._total_weight),
                                 self._EDelta, self._Delta)

    def _lattice_metric_mv(self, gradient, epsilon, *, sj_shift_per_site=None):
        """
        Product metric tensors and hole tensors, like matrix multiply vector. Metric is generated by Deltas and Delta.

        Parameters
        ----------
        gradient : list[list[Tensor]]
            The hole tensors.
        epsilon : float
            The epsilon to avoid singularity of metric.
        sj_shift_per_site : float, optional
            Energy shift per site used in Shaojun's methods.

        Returns
        -------
        list[list[Tensor]]
            The product result.
        """
        if sj_shift_per_site is not None:
            shift = sj_shift_per_site * self._owner.site_number

            result_1 = [[self._Delta[l1][l2].same_shape().zero()
                         for l2 in range(self._owner.L2)]
                        for l1 in range(self._owner.L1)]
            all_name = {("T", "T")} | {(f"P_{l1}_{l2}_{orbit}", f"P_{l1}_{l2}_{orbit}") for l1 in range(self._owner.L1)
                                       for l2 in range(self._owner.L2)
                                       for orbit, edge in self._owner.physics_edges[l1, l2].items()}
            for reweight, deltas, configuration in self._Deltas:
                ws = configuration.hole(())
                inv_ws_conj = ws / (ws.norm_2()**2)
                inv_ws = inv_ws_conj.conjugate()
                config = self._pool._get_config(configuration)
                param_pool = {}
                for positions, observer in self._observer["energy"].items():
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
                        other_config = self._pool._replace_config(config, replacement)
                        wss = self._pool(other_config).hole(())

                        if wss.norm_num() == 0:
                            continue
                        value = inv_ws.contract(observer_shrinked.conjugate(),
                                                {(physics_names[i], f"I{i}") for i in range(body)}).edge_rename({
                                                    f"O{i}": physics_names[i] for i in range(body)
                                                }).contract(wss, all_name)
                        total_value += complex(value).real
                        if other_config in param_pool:
                            param_pool[other_config] += total_value
                        else:
                            param_pool[other_config] = total_value
                param = 0
                for other_config, value in param_pool.items():
                    holes = self._pool(other_config).holes()
                    param += self._lattice_dot(holes, gradient) * value
                param += shift * self._lattice_dot(configuration.holes(), gradient)
                param *= reweight / self._total_weight
                to_sum = self._lattice_map(lambda x1: param * x1, deltas)
                self._lattice_sum(result_1, to_sum)
            allreduce_lattice_buffer(result_1)

            delta = self._lattice_map(lambda x1: x1 / self._total_weight, self._Delta)
            edelta = self._lattice_map(lambda x1: x1 / self._total_weight, self._EDelta)

            param = self._lattice_dot(delta, gradient) * (self.total_energy[0] + shift)
            result_2 = self._lattice_map(lambda x1: x1 * param, delta)

            param = self._lattice_dot(delta, gradient)
            result_3 = self._lattice_map(lambda x1: x1 * param, edelta)
            param = self._lattice_dot(edelta, gradient)
            result_4 = self._lattice_map(lambda x1: x1 * param, delta)

            return self._lattice_map(lambda x1, x2, x3, x4, x5: x1 + x2 - x3 - x4 + epsilon * x5, result_1, result_2,
                                     result_3, result_4, gradient)
        else:
            # Metric = |Deltas[s]> <Deltas[s]| reweight[s] / total_weight - |Delta> / total_weight <Delta| / total_weight
            result_1 = [[self._Delta[l1][l2].same_shape().zero()
                         for l2 in range(self._owner.L2)]
                        for l1 in range(self._owner.L1)]
            for reweight, deltas, _ in self._Deltas:
                param = self._lattice_dot(deltas, gradient) * reweight / self._total_weight
                to_sum = self._lattice_map(lambda x1: param * x1, deltas)
                self._lattice_sum(result_1, to_sum)
            allreduce_lattice_buffer(result_1)

            delta = self._lattice_map(lambda x1: x1 / self._total_weight, self._Delta)
            param = self._lattice_dot(delta, gradient)
            result_2 = self._lattice_map(lambda x1: x1 * param, delta)
            return self._lattice_map(lambda x1, x2, x3: x1 - x2 + epsilon * x3, result_1, result_2, gradient)

    def _lattice_sum(self, result, to_sum):
        """
        Summation lattice tensor like object into result.
        """
        for l1 in range(self._owner.L1):
            for l2 in range(self._owner.L2):
                result[l1][l2] += to_sum[l1][l2]

    def _lattice_dot(self, a, b):
        """
        Dot of two hole tensors, like vector dot product.

        Parameters
        ----------
        a, b : list[list[Tensor]]
            The hole tensors.

        Returns
        -------
        float
            The dot result.
        """
        result = 0.
        for l1 in range(self._owner.L1):
            for l2 in range(self._owner.L2):
                ta = a[l1][l2]
                tb = b[l1][l2]
                result += ta.conjugate(positive_contract=True).contract(tb, {(name, name) for name in ta.names})
        return complex(result).real

    def _lattice_map(self, func, *args):
        """
        Map function to several hole tensors.
        """
        return [[func(*(t[l1][l2] for t in args)) for l2 in range(self._owner.L2)] for l1 in range(self._owner.L1)]

    def natural_gradient(self, step, epsilon, *, sj_shift_per_site=None):
        """
        Get the energy natural gradient for every tensor.

        Parameters
        ----------
        step : int
            conjugate gradient method step count.
        epsilon : float
            The epsilon to avoid singularity of metric.
        sj_shift_per_site : float, optional
            Set sj's metric energy shift, if it is not None, use sj's method instead of standard natural gradient.

        Returns
        -------
        list[list[Tensor]]
            The gradient for every tensor.
        """
        b = self.gradient
        # A = metric
        # A x = b

        x = [[t.same_shape().zero() for t in row] for row in b]
        # r = b - A@x
        r = self._lattice_map(lambda x1, x2: x1 - x2, b,
                              self._lattice_metric_mv(x, epsilon, sj_shift_per_site=sj_shift_per_site))
        # p = r
        p = r
        for t in range(step):
            show(f"conjugate gradient step={t}")
            # alpha = (r @ r) / (p @ A @ p)
            alpha = self._lattice_dot(r, r) / self._lattice_dot(
                p, self._lattice_metric_mv(p, epsilon, sj_shift_per_site=sj_shift_per_site))
            # x = x + alpha * p
            x = self._lattice_map(lambda x1, x2: x1 + alpha * x2, x, p)
            # new_r = r - alpha * A @ p
            new_r = self._lattice_map(lambda x1, x2: x1 - alpha * x2, r,
                                      self._lattice_metric_mv(p, epsilon, sj_shift_per_site=sj_shift_per_site))
            # beta = (new_r @ new_r) / (r @ r)
            beta = self._lattice_dot(new_r, new_r) / self._lattice_dot(r, r)
            # r = new_r
            r = new_r
            # p = r + beta * p
            p = self._lattice_map(lambda x1, x2: x1 + beta * x2, r, p)
        return x
