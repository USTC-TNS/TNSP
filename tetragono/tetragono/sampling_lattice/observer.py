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
import numpy as np
from ..common_toolkit import (show, showln, allreduce_lattice_buffer, allreduce_buffer, lattice_update,
                              lattice_prod_sum, lattice_conjugate, mpi_rank, mpi_comm, pickle)
from ..tensor_element import tensor_element
from .lattice import ConfigurationPool


class Observer():
    """
    Helper type for Observing the sampling lattice.
    """

    __slots__ = [
        "owner", "_observer", "_enable_gradient", "_enable_natural", "_cache_natural_delta", "_cache_configuration",
        "_restrict_subspace", "_classical_energy", "_start", "_result", "_result_square", "_result_reweight", "_count",
        "_total_weight", "_total_weight_square", "_total_log_ws", "_total_energy", "_total_energy_square",
        "_total_energy_reweight", "_Delta", "_EDelta", "_Deltas", "_pool"
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
            self._Delta = [[self.owner[l1, l2].same_shape().conjugate().zero()
                            for l2 in range(self.owner.L2)]
                           for l1 in range(self.owner.L1)]
            self._EDelta = [[self.owner[l1, l2].same_shape().conjugate().zero()
                             for l2 in range(self.owner.L2)]
                            for l1 in range(self.owner.L1)]
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
        classical_energy=None,
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
        cache_configuration : bool | str, optional
            Enable cache the configuration during observing, if it is a string, it will describe the cache strategy for
            the configuration, currently only "drop" is allowed.
        restrict_subspace : optional
            A function return bool to restrict sampling subspace.
        classical_energy : optional
            A function for the classical term of energy.
        """
        self.owner = owner
        # The observables need to measure.
        # dict[str, dict[tuple[tuple[int, int, int], ...], Tensor]]
        self._observer = {}
        self._enable_gradient = False
        self._enable_natural = False
        self._cache_natural_delta = None
        self._cache_configuration = False
        self._restrict_subspace = None
        self._classical_energy = None

        self._start = False

        # Values collected during observing
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

        # Values about gradient collected during observing
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
        self.restrict_subspace(restrict_subspace)
        self.set_classical_energy(classical_energy)

    def set_classical_energy(self, classical_energy=None):
        """
        Set another classical energy term to total energy.

        Parameters
        ----------
        classical_energy
            A function return energy with Configuration as input.
        """
        self._classical_energy = classical_energy

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
            raise RuntimeError("Cannot enable hole after sampling start")
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

    def _create_cache_configuration(self):
        """
        Create or refresh configuration cache pool.
        """
        self._pool = ConfigurationPool(self.owner)

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
        # Update the cache configuration pool
        if self._cache_configuration:
            if self._count == 0:
                self._create_cache_configuration()
            if self._cache_configuration == "drop":
                self._create_cache_configuration()
            configuration = self._pool.add(configuration)

        self._count += 1
        ws = configuration.hole(())  # |s|psi>
        if ws.norm_max() == 0:
            # maybe block mismatch, so ws is 0, return directly, only count is updated, weight will not change.
            # maybe ws is just 0, also return directly
            return

        reweight = ws.norm_2()**2 / possibility  # <psi|s|psi> / p(s)
        self._total_weight += reweight
        self._total_weight_square += reweight * reweight
        self._total_log_ws += np.log(np.abs(complex(ws)))

        inv_ws_conj = ws / (ws.norm_2()**2)  # |s|psi> / <psi|s|psi>
        all_name = {("T", "T")} | {(f"P_{l1}_{l2}_{orbit}", f"P_{l1}_{l2}_{orbit}") for l1 in range(self.owner.L1)
                                   for l2 in range(self.owner.L2) for orbit in self.owner.physics_edges[l1, l2]}
        for name, observers in self._observer.items():
            if name == "energy":
                Es = 0.0
            for positions, observer in observers.items():
                body = len(positions)
                positions_configuration = tuple(configuration[l1l2o] for l1l2o in positions)
                element_pool = tensor_element(observer)
                if positions_configuration not in element_pool:
                    continue
                total_value = 0
                physics_names = [f"P_{l1}_{l2}_{orbit}" for l1, l2, orbit in positions]
                for positions_configuration_s, observer_shrinked in element_pool[positions_configuration].items():
                    # observer_shrinked is |s'|H|s|
                    replacement = {positions[i]: positions_configuration_s[i] for i in range(body)}
                    # Calculate wss: |s'|psi>
                    if self._restrict_subspace is not None:
                        if not self._restrict_subspace(configuration, replacement):
                            # wss should be zero, this term is zero, continue to next wss
                            continue
                    if self._cache_configuration:
                        wss = self._pool.wss(configuration, replacement)
                    else:
                        wss = configuration.replace(replacement)
                        if wss is None:
                            raise NotImplementedError(
                                "not implemented replace style, set cache_configuration to True to calculate it")
                    if wss.norm_max() == 0:
                        continue
                    # <psi|s'|H|s|psi> / <psi|s|psi>
                    value = (
                        inv_ws_conj  #
                        .contract(observer_shrinked, {(physics_names[i], f"I{i}") for i in range(body)})  #
                        .edge_rename({f"O{i}": physics_names[i] for i in range(body)})  #
                        .contract(wss.conjugate(), all_name))
                    total_value += complex(value)
                # total_value is sum_s' <psi|s'|H|s|psi> / <psi|s|psi>
                to_save = total_value.real
                self._result[name][positions] += to_save
                self._result_square[name][positions] += to_save * to_save
                self._result_reweight[name][positions] += to_save * reweight
                if name == "energy":
                    Es += total_value  # Es maybe complex
            if name == "energy":
                if self._classical_energy is not None:
                    Es += self._classical_energy(configuration)
                to_save = Es.real
                self._total_energy += to_save
                self._total_energy_square += to_save * to_save
                self._total_energy_reweight += to_save * reweight
                # Es should be complex here when calculating gradient

                if self._enable_gradient:
                    holes = configuration.holes()  # <psi|s|partial_x psi> / <psi|s|psi>
                    if self.owner.Tensor.is_real:
                        Es = Es.real
                    for l1 in range(self.owner.L1):
                        for l2 in range(self.owner.L2):
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
                                           self._result_reweight[name][positions]) for positions in data
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
        Get the energy gradient for every tensor.

        Returns
        -------
        list[list[Tensor]]
            The gradient for every tensor.
        """
        energy, _ = self.total_energy
        b = (2 * (np.array(self._EDelta) / self._total_weight) - 2 * energy *
             (np.array(self._Delta) / self._total_weight))
        return lattice_conjugate(b)

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
            result += lattice_prod_sum(lattice_conjugate(deltas), deltas) * reweight / self._total_weight
        result = mpi_comm.allreduce(result)

        result -= lattice_prod_sum(lattice_conjugate(self._Delta),
                                   self._Delta) / (self._total_weight * self._total_weight)

        return result

    def _weights_and_deltas(self):
        """
        Get the series of weights and deltas, where the weight is <psi|s|psi> / p(s) and deltas is
        <psi|s|partial_x psi> / <psi|s|psi>.

        Yields
        ------
        tuple[float, list[list[Tensor]]]
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
        result_1 = np.array(
            [[gradient[l1][l2].same_shape().zero() for l2 in range(self.owner.L2)] for l1 in range(self.owner.L1)])
        for reweight, deltas in self._weights_and_deltas():
            param = lattice_prod_sum(deltas, gradient) * reweight / self._total_weight
            lattice_update(result_1, param * lattice_conjugate(deltas))
        allreduce_lattice_buffer(result_1)

        param = lattice_prod_sum(self._Delta, gradient) / (self._total_weight * self._total_weight)
        result_2 = lattice_conjugate(self._Delta) * param
        return result_1 - result_2 + epsilon * gradient

    def natural_gradient(self, step, error, epsilon):
        """
        Get the energy natural gradient for every tensor.

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
        list[list[Tensor]]
            The gradient for every tensor.
        """
        show("calculating natural gradient")
        b = self.gradient
        b_square = lattice_prod_sum(lattice_conjugate(b), b).real
        # A = metric
        # A x = b

        tr = self._trace_metric()
        n = sum(t.norm_num() for row in b for t in row)
        relative_epsilon = epsilon * tr / n

        x = np.array([[t.same_shape().zero() for t in row] for row in b])
        # r = b - A@x
        r = b - self._metric_mv(x, relative_epsilon)
        r_square = lattice_prod_sum(lattice_conjugate(r), r).real
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
            alpha = (lattice_prod_sum(lattice_conjugate(r), r).real /
                     lattice_prod_sum(lattice_conjugate(r), self._metric_mv(p, relative_epsilon)).real)
            # x = x + alpha * p
            x = x + alpha * p
            # new_r = r - alpha * A @ p
            new_r = r - alpha * self._metric_mv(p, relative_epsilon)
            new_r_square = lattice_prod_sum(lattice_conjugate(new_r), new_r).real
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

    def normalize_lattice(self):
        """
        Normalize the owner sampling lattice by the total ws measured during observing.
        """
        # total_log_ws is sum log |ws|, here we do not want to normalize by <psi|psi>
        # We just want to let ws be a proper number, not to large, not to small.
        # Then, the mean log ws here represents the scaling of log|ws|
        mean_log_ws = self._total_log_ws / self._count
        # Here it should use tensor number, not site number
        # param represents scaling of |ws|^(1/L1L2)
        param = np.exp(mean_log_ws / (self.owner.L1 * self.owner.L2))
        self.owner._lattice /= param
