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

from __future__ import annotations
from copyreg import _slotnames
import numpy as np
import lazy
import TAT
from .auxiliaries import Auxiliaries
from .double_layer_auxiliaries import DoubleLayerAuxiliaries
from .abstract_lattice import AbstractLattice
from .common_variable import show, showln, mpi_comm, mpi_rank, mpi_size, allreduce_lattice_buffer, allreduce_buffer
from .tensor_element import tensor_element


class Configuration(Auxiliaries):
    """
    Configuration system for square sampling lattice.
    """

    __slots__ = ["_owner", "_configuration", "_holes"]

    def copy(self, cp=None):
        """
        Copy the configuration system.

        Parameters
        ----------
        cp : Copy, default=None
            The copy object used to copy the internal lazy node graph.

        Returns
        -------
        The new configuration system.
        """
        result = Configuration.__new__(Configuration)
        result = super().copy(cp=cp, result=result)

        result._owner = self._owner
        result._configuration = [[{
            orbit: self._configuration[l1][l2][orbit] for orbit, edge in self._owner.physics_edges[l1, l2].items()
        } for l2 in range(self._owner.L2)] for l1 in range(self._owner.L1)]
        result._holes = self._holes
        return result

    def __init__(self, owner):
        """
        Create configuration system for the given sampling lattice.

        Parameters
        ----------
        owner : SamplingLattice
            The sampling lattice owning this configuration system.
        """
        super().__init__(owner.L1, owner.L2, owner.cut_dimension, False, owner.Tensor)
        self._owner = owner
        # EdgePoint = tuple[self.Symmetry, int]
        self._configuration = [[{orbit: None
                                 for orbit, edge in self._owner.physics_edges[l1, l2].items()}
                                for l2 in range(self._owner.L2)]
                               for l1 in range(self._owner.L1)]
        # update exist configuration
        for l1 in range(owner.L1):
            for l2 in range(owner.L2):
                for orbit, edge in self._owner.physics_edges[l1, l2].items():
                    self[l1, l2, orbit] = self[l1, l2, orbit]
        self._holes = None

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
            if self._configuration[l1][l2][orbit] is None:
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
        return self._configuration[l1][l2][orbit]

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
            self._configuration[l1][l2][orbit] = None
            super().__setitem__((l1, l2), None)
            return
        this_configuration = self._construct_edge_point(value)
        if this_configuration == self._configuration[l1][l2][orbit]:
            changed = False
        else:
            self._configuration[l1][l2][orbit] = this_configuration
            changed = True
        if self._lattice[l1][l2]() is None or changed:
            if self.site_valid(l1, l2):
                shrinked_site = self._shrink_configuration((l1, l2), self._configuration[l1][l2])
                super().__setitem__((l1, l2), shrinked_site)
                self._holes = None

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
            for orbit, configuration in self._configuration[l1][l2].items():
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

    def holes(self):
        """
        Get the lattice holes of this configuration.

        Returns
        -------
        list[list[Tensor]]
            The holes of this configuration.
        """
        if self._holes is None:
            # Prepare
            ws = self.hole(())
            inv_ws_conj = ws / (ws.norm_2()**2)
            inv_ws = inv_ws_conj.conjugate()
            all_name = {("T", "T")} | {(f"P_{l1}_{l2}_{orbit}", f"P_{l1}_{l2}_{orbit}") for l1 in range(self._owner.L1)
                                       for l2 in range(self._owner.L2)
                                       for orbit, edge in self._owner.physics_edges[l1, l2].items()}

            # Calculate
            holes = [[None for l2 in range(self._owner.L2)] for l1 in range(self._owner.L1)]
            # \frac{\partial\langle s|\psi\rangle}{\partial x_i} / \langle s|\psi\rangle
            for l1 in range(self._owner.L1):
                for l2 in range(self._owner.L2):
                    contract_name = all_name.copy()
                    for orbit, edge in self._owner.physics_edges[l1, l2].items():
                        contract_name.remove((f"P_{l1}_{l2}_{orbit}", f"P_{l1}_{l2}_{orbit}"))
                    if l1 == l2 == 0:
                        contract_name.remove(("T", "T"))
                    hole = self.hole(((l1, l2),)).contract(inv_ws, contract_name)
                    hole = hole.edge_rename({
                        "L0": "R",
                        "R0": "L",
                        "U0": "D",
                        "D0": "U"
                    } | {
                        f"P_{l1}_{l2}_{orbit}": f"P{orbit}"
                        for orbit, edge in self._owner.physics_edges[l1, l2].items()
                    })

                    for orbit, shrinker in self._get_shrinker((l1, l2), self._configuration[l1][l2]):
                        hole = hole.contract(shrinker, {(f"P{orbit}", "P")}).edge_rename({"Q": f"P{orbit}"})

                    holes[l1][l2] = hole
            self._holes = holes
        return self._holes


class SamplingLattice(AbstractLattice):
    """
    Square lattice used for sampling.
    """

    __slots__ = ["_lattice", "cut_dimension"]

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
        self.cut_dimension = cut_dimension

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
        "_owner", "_enable_gradient", "_enable_natural", "_start", "_observer", "_result", "_result_square", "_count",
        "_total_weight", "_total_weight_square", "_Delta", "_EDelta", "_Deltas"
    ]

    def __enter__(self):
        """
        Enter sampling loop, flush all cached data in the observer object.
        """
        self._result = {
            name: {positions: 0.0 for positions, observer in observers.items()
                  } for name, observers in self._observer.items()
        }
        self._result_square = {
            name: {positions: 0.0 for positions, observer in observers.items()
                  } for name, observers in self._observer.items()
        }
        self._count = 0
        self._total_weight = 0.0
        self._total_weight_square = 0.0
        if self._enable_gradient:
            self._Delta = [[self._owner[l1, l2].same_shape().conjugate().zero()
                            for l2 in range(self._owner.L2)]
                           for l1 in range(self._owner.L1)]
            self._EDelta = [[self._owner[l1, l2].same_shape().conjugate().zero()
                             for l2 in range(self._owner.L2)]
                            for l1 in range(self._owner.L1)]
            if self._enable_natural:
                self._Deltas = []
        self._start = True

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
        buffer.append(self._count)
        buffer.append(self._total_weight)
        buffer.append(self._total_weight_square)

        buffer = np.array(buffer)
        allreduce_buffer(buffer)
        buffer = buffer.tolist()

        self._total_weight_square = buffer.pop()
        self._total_weight = buffer.pop()
        self._count = buffer.pop()
        for name, observers in reversed(self._observer.items()):
            for positions in reversed(observers):
                self._result_square[name][positions] = buffer.pop()
                self._result[name][positions] = buffer.pop()

        if self._enable_gradient:
            allreduce_lattice_buffer(self._Delta)
            allreduce_lattice_buffer(self._EDelta)

    def __init__(self, owner):
        """
        Create observer object for the given sampling lattice.

        Parameters
        ----------
        owner : SamplingLattice
            The owner of this obsever object.
        """
        self._owner = owner
        self._enable_gradient = False
        self._enable_natural = False
        self._start = False
        self._observer = {}  # dict[str, dict[tuple[tuple[int, int, int], ...], Tensor]]

        self._result = None  # dict[str, dict[tuple[tuple[int, int, int], ...], float]]
        self._result_square = None
        self._count = None  # int
        self._total_weight = None  # float
        self._total_weight_square = None
        self._Delta = None  # list[list[Tensor]]
        self._EDelta = None  # list[list[Tensor]]
        self._Deltas = None

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
            if name == "energy" and self._enable_gradient:
                calculating_gradient = True
                Es = 0
            else:
                calculating_gradient = False
            for positions, observer in observers.items():
                body = observer.rank // 2
                current_configuration = tuple(configuration[positions[i]] for i in range(body))
                element_pool = tensor_element(observer)
                if current_configuration not in element_pool:
                    continue
                total_value = 0
                physics_names = [f"P_{positions[i][0]}_{positions[i][1]}_{positions[i][2]}" for i in range(body)]
                for other_configuration, observer_shrinked in element_pool[current_configuration].items():
                    wss = configuration.replace({positions[i]: other_configuration[i] for i in range(body)})
                    if wss.norm_num() == 0:
                        continue
                    value = inv_ws.contract(observer_shrinked.conjugate(),
                                            {(physics_names[i], f"I{i}") for i in range(body)}).edge_rename({
                                                f"O{i}": physics_names[i] for i in range(body)
                                            }).contract(wss, all_name)
                    total_value += complex(value)
                to_save = total_value.real * reweight
                self._result[name][positions] += to_save
                self._result_square[name][positions] += to_save * to_save
                if calculating_gradient:
                    Es += total_value  # reweight will be multipled later, Es maybe complex
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
                    self._Deltas.append((reweight, holes))

    def _expect_and_deviation_before_reweight(self, total, total_square):
        """
        Get the expect value and deviation from summation of observed value and the summation of its square, without
        reweight.

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
        expect_of_square = total_square / self._count
        expect = total / self._count
        square_of_expect = expect * expect
        variance = expect_of_square - square_of_expect
        variance /= self._count
        if variance < 0.0:
            # When total summate several same values, numeric error will lead variance < 0
            variance = 0.0
        deviation = np.sqrt(variance)
        return expect, deviation

    def _expect_and_deviation(self, total, total_square):
        """
        Get the expect value and deviation from summation of observed value and the summation of its square, with
        reweight.

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
        expect_num, deviation_num = self._expect_and_deviation_before_reweight(total, total_square)
        expect_den, deviation_den = self._expect_and_deviation_before_reweight(self._total_weight,
                                                                               self._total_weight_square)
        expect = expect_num / expect_den
        deviation = abs(expect) * np.sqrt((deviation_num / expect_num)**2 + (deviation_den / expect_den)**2)
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
    def _total_energy(self):
        """
        Get the observed energy.

        Returns
        -------
        tuple[float, float]
            The energy per site.
        """
        name = "energy"
        result = [
            self._expect_and_deviation(self._result[name][positions], self._result_square[name][positions])
            for positions, _ in self._observer[name].items()
        ]
        expect = sum(e for e, d in result)
        deviation = np.sqrt(sum(d * d for e, d in result))
        return expect, deviation

    @property
    def energy(self):
        """
        Get the observed energy per site.

        Returns
        -------
        tuple[float, float]
            The energy per site.
        """
        expect, deviation = self._total_energy
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
        energy, _ = self._total_energy
        return self._lattice_map(lambda x1, x2: 2 * (x1 / self._total_weight) - 2 * energy * (x2 / self._total_weight),
                                 self._EDelta, self._Delta)

    def _lattice_metric_mv(self, gradient, epsilon):
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
        result_1 = [
            [self._Delta[l1][l2].same_shape().zero() for l2 in range(self._owner.L2)] for l1 in range(self._owner.L1)
        ]
        for reweight, deltas in self._Deltas:
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
        r = self._lattice_map(lambda x1, x2: x1 - x2, b, self._lattice_metric_mv(x, epsilon))
        # p = r
        p = r
        for t in range(step):
            show(f"conjugate gradient step={t}")
            # alpha = (r @ r) / (p @ A @ p)
            alpha = self._lattice_dot(r, r) / self._lattice_dot(p, self._lattice_metric_mv(p, epsilon))
            # x = x + alpha * p
            x = self._lattice_map(lambda x1, x2: x1 + alpha * x2, x, p)
            # new_r = r - alpha * A @ p
            new_r = self._lattice_map(lambda x1, x2: x1 - alpha * x2, r, self._lattice_metric_mv(p, epsilon))
            # beta = (new_r @ new_r) / (r @ r)
            beta = self._lattice_dot(new_r, new_r) / self._lattice_dot(r, r)
            # r = new_r
            r = new_r
            # p = r + beta * p
            p = self._lattice_map(lambda x1, x2: x1 + beta * x2, r, p)
        return x


class Sampling:
    """
    Helper type for run sampling for sampling lattice.
    """

    __slots__ = ["_owner", "configuration"]

    def __init__(self, owner):
        """
        Create sampling object for the given sampling lattice.

        Parameters
        ----------
        owner : SamplingLattice
            The owner of this sampling object.
        """
        self._owner = owner
        self.configuration = Configuration(self._owner)

    def refresh_all(self):
        """
        Refresh the sampling system, need to be called after lattice tensor changed.
        """
        self.configuration.refresh_all()

    def __call__(self):
        """
        Get the next sampling configuration

        Returns
        -------
        tuple[float, Configuration]
            The sampled weight in importance sampling, and the result configuration system.
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
        current_configuration = tuple(self.configuration[l1l2o] for l1l2o in positions)  # tuple[EdgePoint, ...]
        element_pool = tensor_element(hamiltonian)
        if current_configuration not in element_pool:
            return ws
        possible_hopping = element_pool[current_configuration]
        if possible_hopping:
            hopping_number = len(possible_hopping)
            configuration_new, element = list(possible_hopping.items())[TAT.random.uniform_int(0, hopping_number - 1)()]
            hopping_number_s = len(element_pool[configuration_new])
            replacement = {positions[i]: configuration_new[i] for i in range(body)}
            wss = float(self.configuration.replace(replacement))  # which return a tensor, we only need its norm
            p = (wss**2) / (ws**2) * hopping_number / hopping_number_s
            if TAT.random.uniform_real(0, 1)() < p:
                ws = wss
                for i in range(body):
                    self.configuration[positions[i]] = configuration_new[i]
        return ws

    def __call__(self):
        owner = self._owner
        if not self.configuration.valid():
            raise RuntimeError("Configuration not initialized")
        ws = float(self.configuration.hole(()))
        if self._sweep_order is None:
            self._sweep_order = self._get_proper_position_order()
        for positions in self._sweep_order:
            hamiltonian = owner._hamiltonians[positions]
            ws = self._single_term(positions, hamiltonian, ws)
        self._sweep_order.reverse()
        for positions in self._sweep_order:
            hamiltonian = owner._hamiltonians[positions]
            ws = self._single_term(positions, hamiltonian, ws)
        return ws**2, self.configuration

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

        self._zero_configuration()
        for t in range(mpi_rank):
            self._next_configuration()

    def _zero_configuration(self):
        owner = self._owner
        for l1 in range(owner.L1):
            for l2 in range(owner.L2):
                for orbit, edge in self._edges[l1][l2].items():
                    self.configuration[l1, l2, orbit] = edge.get_point_from_index(0)

    def _next_configuration(self):
        owner = self._owner
        for l1 in range(owner.L1):
            for l2 in range(owner.L2):
                for orbit, edge in self._edges[l1][l2].items():
                    index = edge.get_index_from_point(self.configuration[l1, l2, orbit])
                    index += 1
                    if index == edge.dimension:
                        self.configuration[l1, l2, orbit] = edge.get_point_from_index(0)
                    else:
                        self.configuration[l1, l2, orbit] = edge.get_point_from_index(index)
                        return

    def __call__(self):
        for t in range(mpi_size):
            self._next_configuration()
        return 1., self.configuration


class DirectSampling(Sampling):
    """
    Direct sampling.
    """

    __slots__ = ["_cut_dimension", "_double_layer_auxiliaries"]

    def __init__(self, owner, cut_dimension):
        super().__init__(owner)
        self._cut_dimension = cut_dimension
        self.refresh_all()

    def refresh_all(self):
        super().refresh_all()
        owner = self._owner
        self._double_layer_auxiliaries = DoubleLayerAuxiliaries(owner.L1, owner.L2, self._cut_dimension, False,
                                                                owner.Tensor)
        for l1 in range(owner.L1):
            for l2 in range(owner.L2):
                this = owner[l1, l2].copy()
                self._double_layer_auxiliaries[l1, l2, "n"] = this
                self._double_layer_auxiliaries[l1, l2, "c"] = this.conjugate()

    def __call__(self):
        owner = self._owner
        random = TAT.random.uniform_real(0, 1)
        for l1 in range(owner.L1):
            for l2 in range(owner.L2):
                for orbit, edge in owner.physics_edges[l1, l2].items():
                    self.configuration[l1, l2, orbit] = None
        possibility = 1.
        for l1 in range(owner.L1):

            three_line_auxiliaries = DoubleLayerAuxiliaries(3, owner.L2, -1, False, owner.Tensor)
            line_3 = []
            for l2 in range(owner.L2):
                tensor_1 = self.configuration._up_to_down_site[l1 - 1, l2]()
                three_line_auxiliaries[0, l2, "n"] = tensor_1
                three_line_auxiliaries[0, l2, "c"] = tensor_1.conjugate()
                tensor_2 = owner[l1, l2]
                three_line_auxiliaries[1, l2, "n"] = tensor_2
                three_line_auxiliaries[1, l2, "c"] = tensor_2.conjugate()
                line_3.append(self._double_layer_auxiliaries._down_to_up_site[l1 + 1, l2]())
            three_line_auxiliaries._down_to_up[2].reset(line_3)

            for l2 in range(owner.L2):
                shrinked_site_tensor = owner[l1, l2]
                configuration = {}
                shrinkers = self.configuration._get_shrinker((l1, l2), configuration)
                for orbit, edge in owner.physics_edges[l1, l2].items():
                    hole = three_line_auxiliaries.hole([(1, l2, orbit)]).transpose(["I0", "O0"])
                    hole_edge = hole.edges("O0")
                    rho = np.array([])
                    for seg in hole_edge.segment:
                        symmetry, _ = seg
                        block_rho = hole.blocks[[("I0", -symmetry), ("O0", symmetry)]]
                        diag_rho = np.diagonal(block_rho)
                        rho = np.array([*rho, *diag_rho])
                    rho = rho.real
                    if len(rho) == 0:
                        return self()
                    rho = rho / np.sum(rho)
                    choice = self._choice(random(), rho)
                    possibility *= rho[choice]
                    configuration[orbit] = self.configuration[l1, l2, orbit] = hole_edge.get_point_from_index(choice)
                    _, shrinker = next(shrinkers)
                    shrinked_site_tensor = shrinked_site_tensor.contract(shrinker.edge_rename({"P": f"P{orbit}"}),
                                                                         {(f"P{orbit}", "Q")})
                    three_line_auxiliaries[1, l2, "n"] = shrinked_site_tensor
                    three_line_auxiliaries[1, l2, "c"] = shrinked_site_tensor.conjugate()
        return possibility, self.configuration

    def _choice(self, p, rho):
        for i, r in enumerate(rho):
            p -= r
            if p < 0:
                return i
        return i
