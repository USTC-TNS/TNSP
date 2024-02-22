#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2020-2024 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
import PyScalapack
from ..utility import (show, showln, allreduce_lattice_buffer, allreduce_buffer, allreduce_number, bcast_buffer,
                       lattice_update, lattice_prod_sum, lattice_conjugate, mpi_rank, mpi_size, mpi_comm, pickle)
from ..tensor_element import tensor_element
from .lattice import ConfigurationPool


class Observer():
    """
    Helper type for Observing the sampling lattice.
    """

    __slots__ = [
        "owner", "_observer", "_enable_gradient", "_enable_natural", "_cache_natural_delta", "_cache_configuration",
        "_restrict_subspace", "_classical_energy", "_start", "_result_reweight", "_result_reweight_square",
        "_result_square_reweight_square", "_count", "_total_weight", "_total_weight_square", "_total_log_ws",
        "_whole_result_reweight", "_whole_result_reweight_square", "_whole_result_square_reweight_square",
        "_total_imaginary_energy_reweight", "_Delta", "_EDelta", "_Deltas", "_pool"
    ]

    def __enter__(self):
        """
        Enter sampling loop, flush all cached data in the observer object.
        """
        self._start = True
        self._result_reweight = {
            name: {
                positions: 0.0 for positions, observer in observers.items()
            } for name, observers in self._observer.items()
        }
        self._result_reweight_square = {
            name: {
                positions: 0.0 for positions, observer in observers.items()
            } for name, observers in self._observer.items()
        }
        self._result_square_reweight_square = {
            name: {
                positions: 0.0 for positions, observer in observers.items()
            } for name, observers in self._observer.items()
        }
        self._count = 0
        self._total_weight = 0.0
        self._total_weight_square = 0.0
        self._total_log_ws = 0.0
        self._whole_result_reweight = {name: 0.0 for name in self._observer}
        self._whole_result_reweight_square = {name: 0.0 for name in self._observer}
        self._whole_result_square_reweight_square = {name: 0.0 for name in self._observer}
        self._total_imaginary_energy_reweight = 0.0
        if self._enable_gradient:
            self._Delta = [[self.owner[l1, l2].same_shape().conjugate().zero_()
                            for l2 in range(self.owner.L2)]
                           for l1 in range(self.owner.L1)]
            self._EDelta = [[self.owner[l1, l2].same_shape().conjugate().zero_()
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
                buffer.append(self._result_reweight[name][positions])
                buffer.append(self._result_reweight_square[name][positions])
                buffer.append(self._result_square_reweight_square[name][positions])
        buffer.append(self._count)
        buffer.append(self._total_weight)
        buffer.append(self._total_weight_square)
        buffer.append(self._total_log_ws)
        for name in self._observer:
            buffer.append(self._whole_result_reweight[name])
            buffer.append(self._whole_result_reweight_square[name])
            buffer.append(self._whole_result_square_reweight_square[name])
        buffer.append(self._total_imaginary_energy_reweight)

        buffer = np.array(buffer)
        allreduce_buffer(buffer)
        buffer = buffer.tolist()

        self._total_imaginary_energy_reweight = buffer.pop()
        for name in reversed(self._observer):
            self._whole_result_square_reweight_square[name] = buffer.pop()
            self._whole_result_reweight_square[name] = buffer.pop()
            self._whole_result_reweight[name] = buffer.pop()
        self._total_log_ws = buffer.pop()
        self._total_weight_square = buffer.pop()
        self._total_weight = buffer.pop()
        self._count = buffer.pop()
        for name, observers in reversed(self._observer.items()):
            for positions in reversed(observers):
                self._result_square_reweight_square[name][positions] = buffer.pop()
                self._result_reweight_square[name][positions] = buffer.pop()
                self._result_reweight[name][positions] = buffer.pop()

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
        # dict[str, dict[tuple[tuple[int, int, int], ...], float]]
        self._result_reweight = None
        self._result_reweight_square = None
        self._result_square_reweight_square = None
        self._count = None  # int
        self._total_weight = None  # float
        self._total_weight_square = None
        self._total_log_ws = None
        self._whole_result_reweight = None
        self._whole_result_reweight_square = None
        self._whole_result_square_reweight_square = None
        self._total_imaginary_energy_reweight = None

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
        $|\\langle\\psi s\\rangle|^2$, If it is not, a reweight argument should be passed with a non-one float number.

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
        all_name = {("T", "T")} | {(f"P_{l1}_{l2}_{orbit}", f"P_{l1}_{l2}_{orbit}")
                                   for l1 in range(self.owner.L1)
                                   for l2 in range(self.owner.L2)
                                   for orbit in self.owner.physics_edges[l1, l2]}
        for name, observers in self._observer.items():
            whole_name_value = 0
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
                self._result_reweight[name][positions] += to_save * reweight
                self._result_reweight_square[name][positions] += to_save * reweight**2
                self._result_square_reweight_square[name][positions] += to_save**2 * reweight**2
                whole_name_value += total_value  # maybe complex
            if name == "energy" and self._classical_energy is not None:
                whole_name_value += self._classical_energy(configuration)
            to_save = whole_name_value.real
            self._whole_result_reweight[name] += to_save * reweight
            self._whole_result_reweight_square[name] += to_save * reweight**2
            self._whole_result_square_reweight_square[name] += to_save**2 * reweight**2
            if name == "energy":
                self._total_imaginary_energy_reweight += whole_name_value.imag * reweight
                # Es should be complex here when calculating gradient
            if name == "energy" and self._enable_gradient:
                Es = whole_name_value
                if self.owner.Tensor.is_real:
                    Es = Es.real
                holes = configuration.holes()  # <psi|s|partial_x psi> / <psi|s|psi>
                for l1, l2 in self.owner.sites():
                    hole = holes[l1][l2] * reweight
                    self._Delta[l1][l2] += hole
                    self._EDelta[l1][l2] += Es * hole
                if self._enable_natural:
                    if self._cache_natural_delta:
                        with open(os.path.join(self._cache_natural_delta, str(mpi_rank)), "ab") as file:
                            pickle.dump(holes, file)
                        self._Deltas.append((reweight, Es, None))
                    else:
                        self._Deltas.append((reweight, Es, holes))

    @property
    def instability(self):
        """
        The instability of the sampling method.
        """
        N = self._count
        expect = self._total_weight / N
        expect_square = expect**2
        square_expect = self._total_weight_square / N
        variance = square_expect - expect_square
        if variance < 0.0:
            deviation = 0.0
        else:
            deviation = variance**0.5
        return deviation / expect

    def _expect_and_deviation(self, total_reweight, total_reweight_square, total_square_reweight_square):
        """
        Get the expect value and deviation.

        Parameters
        ----------
        total_reweight : float
            The summation of observed value with reweight.
        total_reweight_square : float
            The summation of observed value with reweight square.
        total_square_reweight_square : float
            The summation of observed value square with reweight square.

        Returns
        -------
        tuple[float, float]
            The expect value and deviation.
        """
        if total_reweight == 0.0 or self._total_weight == 0.0:
            return 0.0, 0.0

        R = self._total_weight
        ER = total_reweight

        RR = self._total_weight_square
        ERR = total_reweight_square
        EERR = total_square_reweight_square

        expect = ER / R
        variance = (EERR - 2 * ERR * expect + RR * expect**2) / R**2
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
                    self._expect_and_deviation(self._result_reweight[name][positions],
                                               self._result_reweight_square[name][positions],
                                               self._result_square_reweight_square[name][positions])
                for positions in data
            } for name, data in self._observer.items()
        }

    @property
    def whole_result(self):
        """
        Get the observer result of the whole observer set. It is useful if the deviation of the whole is wanted.

        Returns
        -------
        dict[str, tuple[float, float]]
            The observer result of each observer set name.
        """
        return {
            name:
                self._expect_and_deviation(self._whole_result_reweight[name], self._whole_result_reweight_square[name],
                                           self._whole_result_square_reweight_square[name]) for name in self._observer
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
        name = "energy"
        return self._expect_and_deviation(self._whole_result_reweight[name], self._whole_result_reweight_square[name],
                                          self._whole_result_square_reweight_square[name])

    def _total_energy_with_imaginary_part(self):
        name = "energy"
        if self.owner.Tensor.is_complex:
            return (self._whole_result_reweight[name] + self._total_imaginary_energy_reweight * 1j) / self._total_weight
        else:
            return self._whole_result_reweight[name] / self._total_weight

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
        energy = self._total_energy_with_imaginary_part()
        b = ((np.array(self._EDelta) / self._total_weight) - energy * (np.array(self._Delta) / self._total_weight))
        b *= 2
        return lattice_conjugate(b)

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
                for reweight, Es, _ in self._Deltas:
                    deltas = pickle.load(file)
                    yield reweight, Es, deltas
        else:
            for reweight, Es, deltas in self._Deltas:
                yield reweight, Es, deltas

    def natural_gradient_by_conjugate_gradient(self, step, error):
        """
        Get the energy natural gradient for every tensor.

        Parameters
        ----------
        step : int
            conjugate gradient method step count.
        error : float
            conjugate gradient method expected error.

        Returns
        -------
        list[list[Tensor]]
            The gradient for every tensor.
        """
        show("calculating natural gradient")
        energy = self._total_energy_with_imaginary_part()
        delta = self._delta_to_array(self._Delta) / self._total_weight

        Delta = []
        Energy = []
        for reweight_s, energy_s, delta_s in self._weights_and_deltas():
            param = (reweight_s / self._total_weight)**(1 / 2)
            Delta.append((self._delta_to_array(delta_s) - delta) * param)
            Energy.append((energy_s.conjugate() - energy) * param)
        Delta = np.asarray(Delta)
        Energy = np.asarray(Energy)

        # A x = b
        # DT r D x = DT r E

        # Delta, NsNp
        def D(v):
            # Ns Np * Np => Ns
            if Energy.size != 0:
                return Delta @ v
            else:
                return np.zeros_like(Energy)

        def DT(v):
            # Np Ns * Ns => Np
            if Energy.size != 0:
                result = np.conj(Delta.T) @ v
            else:
                result = np.zeros_like(delta)
            allreduce_buffer(result)
            return result

        def A(v):
            return DT(D(v))

        b = DT(Energy)
        b_square = np.dot(np.conj(b), b).real

        x = np.zeros_like(b)
        r = b - A(x)
        p = r
        r_square = np.dot(np.conj(r), r).real
        # loop
        t = 0
        while True:
            if t == step:
                reason = "max step count reached"
                break
            if error != 0.0:
                if error**2 > r_square / b_square:
                    reason = "r^2 is small enough"
                    break
            show(f"conjugate gradient step={t} r^2/b^2={r_square/b_square}")
            Dp = D(p)
            alpha = r_square / allreduce_number((np.conj(Dp) @ Dp).real)
            x = x + alpha * p
            r = r - alpha * DT(Dp)
            new_r_square = np.dot(np.conj(r), r).real
            beta = new_r_square / r_square
            r_square = new_r_square
            p = r + beta * p
            t += 1

        x = 2 * x
        showln(f"natural gradient calculated step={t} r^2/b^2={r_square/b_square} {reason}")
        return lattice_conjugate(self._array_to_delta(np.conj(x)))

    def _delta_to_array(self, delta):
        # Both delta and result array is in bra space
        result = []
        for l1, l2 in self.owner.sites():
            result.append(delta[l1][l2].transpose(self._Delta[l1][l2].names).storage)
        result = np.concatenate(result)
        return result

    def _array_to_delta(self, array):
        # Both array and result delta is in bra space
        result = np.array(
            [[self._Delta[l1][l2].same_shape() for l2 in range(self.owner.L2)] for l1 in range(self.owner.L1)])
        index = 0
        for row in result:
            for tensor in row:
                size = len(tensor.storage)
                tensor.storage = array[index:index + size]
                index += size
        return result

    def natural_gradient_by_direct_pseudo_inverse(self, r_pinv, a_pinv, libraries):
        """
        Get the energy natural gradient for every tensor.

        Parameters
        ----------
        r_pinv, a_pinv : float
            Parameters control how to calculate pseudo inverse.
        libraries : list[str]
            The dynamic link libraries containing scalapack functions.

        Returns
        -------
        list[list[Tensor]]
            The gradient for every tensor.
        """
        # shape of   rho              is   s (* s)
        # <X> = sum X_s rho_s / sum rho_s
        # let r = rho / sum rho
        # <X> = X r     # r is vector like
        # <XY> = X r Y  # r is matrix like
        # shape of   Delta - <Delta>  is   s * p, which is gradient, not derivative
        # shape of   E - <E>          is   s

        # The program record:
        # rho, E rho, Delta rho, Delta E rho
        # The program calc:
        # E = <E>
        # G = (<E Delta> - E <Delta>)^{+} = (<(E - <E>) (D - <D>)>)^{+} = <(D - <D>)^{+} (E - <E>)^{+}>
        # Old equation is:    (Delta - <Delta>)^{+} r (Delta - <Delta>) NG = (Delta - <Delta>)^{+} r (E - <E>)^{+}
        # That is             (Delta - <Delta>) NG = (E - <E>)^{+}
        # New equation is:    NG = (Delta - <Delta>)^{+} [(Delta - <Delta>) (Delta - <Delta>)^{+}]^{-1} (E - <E>)^{+}
        # where g is just:    (Delta - <Delta>)^{+} r (Delta - <Delta>)
        show("calculating natural gradient")
        energy = self._total_energy_with_imaginary_part()
        delta = self._delta_to_array(self._Delta) / self._total_weight

        dtype = np.dtype(self.owner.Tensor.dtype)
        btype = self.owner.Tensor.btype

        Delta = []
        Energy = []
        for _, energy_s, delta_s in self._weights_and_deltas():
            Delta.append(self._delta_to_array(delta_s) - delta)
            Energy.append(energy_s.conjugate() - energy)
        Delta = np.asfortranarray(Delta, dtype=dtype)
        Energy = np.asfortranarray(Energy, dtype=dtype)

        total_n_s = int(self._count)
        result_array = self._pseudo_inverse_kernel(Delta, Energy, r_pinv, a_pinv, total_n_s, dtype, btype, libraries)
        x = 2 * result_array
        showln("calculate natural gradient done")
        return lattice_conjugate(self._array_to_delta(np.conj(x)))

    @staticmethod
    def _pseudo_inverse_kernel(Delta, Energy, r_pinv, a_pinv, total_n_s, dtype, btype, libraries):
        scalapack = PyScalapack(*libraries)

        with scalapack(b'C', -1, 1) as context:
            n_s, n_p = Delta.shape

            rdtype = dtype.type().real.dtype
            is_complex = np.issubdtype(dtype, np.complexfloating)
            is_real = np.issubdtype(dtype, np.floating)

            Delta = context.array(total_n_s, n_p, 1, n_p, data=Delta)
            Energy = context.array(total_n_s, 1, 1, 1, data=Energy)
            if Delta.local_m != n_s:
                raise RuntimeError("local sampling number and global sampling number dismatch.")

            # Delta -> T
            T = context.array(total_n_s, total_n_s, 1, total_n_s, dtype=dtype)
            scalapack.pgemm[btype](
                b'N',
                b'C',
                *(total_n_s, total_n_s, n_p),
                scalapack.f_one[btype],
                *Delta.scalapack_params(),
                *Delta.scalapack_params(),
                scalapack.f_zero[btype],
                *T.scalapack_params(),
            )

            # T -> U
            sqrt_size = int(context.size.value**0.5)
            with scalapack(b'C', sqrt_size, sqrt_size) as context_square:
                T_square = context_square.array(total_n_s, total_n_s, 4, 4, dtype=dtype)
                scalapack.pgemr2d[btype](
                    *(total_n_s, total_n_s),
                    *T.scalapack_params(),
                    *T_square.scalapack_params(),
                    context.ictxt,
                )

                U_square = context_square.array(total_n_s, total_n_s, 4, 4, dtype=dtype)
                L = np.zeros([total_n_s], dtype=rdtype)
                if context_square:
                    c_info = scalapack.ctypes.c_int()
                    # maybe complex, always use double space of real
                    c_f_lwork = (np.ctypeslib.as_ctypes_type(rdtype) * 2)()
                    c_f_lrwork = np.ctypeslib.as_ctypes_type(rdtype)()
                    c_liwork = scalapack.ctypes.c_int()
                    scalapack.pheevd[btype](
                        b'V',
                        b'U',
                        total_n_s,
                        *T_square.scalapack_params(),
                        scalapack.numpy_ptr(L),
                        *U_square.scalapack_params(),
                        *(c_f_lwork, scalapack.neg_one),
                        *((c_f_lrwork, scalapack.neg_one) if is_complex else ()),
                        *(c_liwork, scalapack.neg_one),
                        c_info,
                    )
                    if c_info.value != 0:
                        raise RuntimeError(f"Error in p?syevd or p?heevd with info = {c_info.value}")
                    lwork = int(c_f_lwork[0])
                    lrwork = int(c_f_lrwork.value)
                    liwork = c_liwork.value
                    work = np.zeros([lwork], dtype=dtype)
                    rwork = np.zeros([lrwork], dtype=rdtype)
                    iwork = np.zeros([liwork], dtype=int)
                    scalapack.pheevd[btype](
                        b'V',
                        b'U',
                        total_n_s,
                        *T_square.scalapack_params(),
                        scalapack.numpy_ptr(L),
                        *U_square.scalapack_params(),
                        *(scalapack.numpy_ptr(work), lwork),
                        *((scalapack.numpy_ptr(rwork), lrwork) if is_complex else ()),
                        *(scalapack.numpy_ptr(iwork), liwork),
                        c_info,
                    )
                    if c_info.value != 0:
                        raise RuntimeError(f"Error in p?syevd or p?heevd with info = {c_info.value}")

                U = context.array(total_n_s, total_n_s, 1, total_n_s, dtype=dtype)
                scalapack.pgemr2d[btype](
                    *(total_n_s, total_n_s),
                    *U_square.scalapack_params(),
                    *U.scalapack_params(),
                    context.ictxt,
                )

            # U -> UE
            tmp1 = context.array(total_n_s, 1, 1, 1, dtype=dtype)
            scalapack.pgemv[btype](
                b'C',
                *(total_n_s, total_n_s),
                scalapack.f_one[btype],
                *U.scalapack_params(),
                *Energy.scalapack_params(),
                scalapack.one,
                scalapack.f_zero[btype],
                *tmp1.scalapack_params(),
                scalapack.one,
            )

            # UE -> LUE
            bcast_buffer(L)
            L_max = L[-1]
            num = r_pinv * L_max + a_pinv
            for i in range(n_s):
                l = L[context.size.value * i + context.rank.value]
                if l <= 0:
                    l_inv = 0
                else:
                    l_inv = 1 / (l * (1 + (num / l)**6))
                tmp1.data[i] *= l_inv

            # LUE -> ULUE
            tmp2 = context.array(total_n_s, 1, 1, 1, dtype=dtype)
            scalapack.pgemv[btype](
                b'N',
                *(total_n_s, total_n_s),
                scalapack.f_one[btype],
                *U.scalapack_params(),
                *tmp1.scalapack_params(),
                scalapack.one,
                scalapack.f_zero[btype],
                *tmp2.scalapack_params(),
                scalapack.one,
            )

            # ULUE -> Delta ULUE
            NG = np.zeros([n_p], dtype=dtype)
            scalapack.gemv[btype](
                b'C',
                *(n_s, n_p),
                scalapack.f_one[btype],
                *Delta.lapack_params(),
                scalapack.numpy_ptr(tmp2.data),
                scalapack.one,
                scalapack.f_zero[btype],
                scalapack.numpy_ptr(NG),
                scalapack.one,
            )

            allreduce_buffer(NG)
            return NG

        # The commented code is only for single process
        # T = Delta @ np.conj(Delta).T
        # L, U = np.linalg.eigh(T)
        # L_max = np.max(L)
        # L_inv = 1 / (L * (1 + ((r_pinv * L_max + a_pinv) / L)**6))
        # NG = np.einsum("ij,jk,k,kl,l->i", np.conj(Delta).T, U, L_inv, np.conj(U).T, Energy)
        # return NG

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
