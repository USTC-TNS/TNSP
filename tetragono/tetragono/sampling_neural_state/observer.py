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

import numpy as np
import torch
from ..utility import allreduce_buffer, allreduce_number, show, showln
from .state import Configuration, index_tensor_element, torch_grad


class Observer():
    """
    Helper type for Observing the sampling neural state.
    """

    __slots__ = [
        "owner", "_observer", "_enable_gradient", "_enable_natural", "_start", "_result_reweight",
        "_result_reweight_square", "_result_square_reweight_square", "_count", "_total_weight", "_total_weight_square",
        "_total_log_ws", "_whole_result_reweight", "_whole_result_reweight_square",
        "_whole_result_square_reweight_square", "_total_imaginary_energy_reweight", "_Delta", "_EDelta", "_Deltas"
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
            self._Delta = None
            self._EDelta = None
            if self._enable_natural:
                self._Deltas = []

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
            allreduce_buffer(self._Delta)
            allreduce_buffer(self._EDelta)

    def __init__(
        self,
        owner,
        *,
        observer_set=None,
        enable_energy=False,
        enable_gradient=False,
        enable_natural_gradient=False,
    ):
        """
        Create observer object for the given sampling neural state.

        Parameters
        ----------
        owner : SamplingNeuralState
            The owner of this obsever object.
        observer_set : dict[str, dict[tuple[tuple[int, int, int], ...], Tensor]], optional
            The given observers to observe.
        enable_energy : bool, optional
            Enable observing the energy.
        enable_gradient : bool, optional
            Enable calculating the gradient.
        enable_natural_gradient : bool, optional
            Enable calculating the natural gradient.
        """
        self.owner = owner
        # The observables need to measure.
        # dict[str, dict[tuple[tuple[int, int, int], ...], Tensor]]
        self._observer = {}
        self._enable_gradient = False
        self._enable_natural = False

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
        self._Delta = None
        self._EDelta = None
        self._Deltas = None

        if observer_set is not None:
            for key, value in observer_set.items():
                self.add_observer(key, value)
            self._observer = observer_set
        if enable_energy:
            self.add_energy()
        if enable_gradient:
            self.enable_gradient()
        if enable_natural_gradient:
            self.enable_natural_gradient()

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

        L1 = self.owner.L1
        L2 = self.owner.L2
        orbit = max(orbit for [l1, l2, orbit], edge in self.owner.physics_edges) + 1

        result = {}
        for positions, observer in observers.items():
            tensor_positions = torch.tensor(positions, device=torch.device("cpu"))
            indices = (tensor_positions[:, 0] * L2 + tensor_positions[:, 1]) * orbit + tensor_positions[:, 2]
            element_pool = index_tensor_element(observer)
            result[positions] = (
                observer,
                indices,
                {
                    x: {
                        y: (
                            item,
                            tensor_y,
                            self._fermi_sign(tensor_x, tensor_y, indices),
                        ) for y, [item, tensor_x, tensor_y] in items.items()
                    } for x, items in element_pool.items()
                },
            )

        self._observer[name] = result

    def _fermi_sign(self, x, y, indices):
        if self.owner.op_pool is None:
            return (False, torch.empty([0, 3], device=torch.device("cpu")))

        L1, L2, orbit, dim = self.owner.op_pool.shape
        op = self.owner.op_pool.view([-1, dim])
        ops_x = op[indices, x]
        ops_y = op[indices, y]
        fix_update = (indices.unsqueeze(0) > indices.unsqueeze(1)).triu()

        mask_x = torch.zeros([L1 * L2 * orbit], device=torch.device("cpu"), dtype=torch.bool)
        mask_y = torch.zeros([L1 * L2 * orbit], device=torch.device("cpu"), dtype=torch.bool)
        for index, op_x, op_y in zip(indices, ops_x, ops_y):
            mask_x[:index] ^= op_x
            mask_y[:index] ^= op_y

        count = 0
        count += torch.sum(ops_x.unsqueeze(0) * ops_x.unsqueeze(1) * fix_update)
        count += torch.sum(ops_y.unsqueeze(0) * ops_y.unsqueeze(1) * fix_update)
        for index, op_x, op_y in zip(indices, ops_x, ops_y):
            count += (op_x ^ op_y) and mask_y[index]

        mask = mask_x ^ mask_y
        # The base sign and the indices where the parity should be checked.
        return (count % 2 != 0, mask.to(dtype=torch.int64).nonzero())

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

    def _fermi_sign_old(self, config, config_s, positions):
        if self.owner.op_pool is None:
            return +1

        L1, L2, orbit = config.shape
        length = len(config)

        op = torch.gather(self.owner.op_pool, 3, config.unsqueeze(-1)).view([-1])
        op_s = torch.gather(self.owner.op_pool, 3, config_s.unsqueeze(-1)).view([-1])
        cumsum = torch.cumsum(op, dim=0)
        cumsum[-1] = 0
        cumsum_s = torch.cumsum(op_s, dim=0)
        cumsum_s[-1] = 0

        positions = torch.tensor(positions, device=torch.device("cpu"))
        indices = (positions[:, 0] * L2 + positions[:, 1]) * orbit + positions[:, 2]
        op_indices = op[indices]
        op_indices_s = op_s[indices]
        fix_update = (indices.unsqueeze(0) > indices.unsqueeze(1)).triu()
        count = 0
        count += torch.sum(cumsum[indices - 1] * op_indices)
        count += torch.sum(cumsum_s[indices - 1] * op_indices_s)
        count += torch.sum(op_indices.unsqueeze(0) * op_indices.unsqueeze(1) * fix_update)
        count += torch.sum(op_indices_s.unsqueeze(0) * op_indices_s.unsqueeze(1) * fix_update)
        if count % 2 == 0:
            return +1
        else:
            return -1

    def __call__(self, configurations, amplitudes, weights, multiplicities):
        """
        Collect observer value from given configurations, the sampling should have distribution based on weights

        Parameters
        ----------
        configurations : list[Configuration]
            The configuration of the model.
        amplitudes : list[complex]
            The amplitudes of the configurations.
        weights : list[float]
            the sampled weight used in importance sampling.
        multiplicities : list[int]
            the sampled counts of the unique sampling.
        """
        batch_size = len(weights)

        configurations_cpu = configurations.cpu()
        amplitudes_cpu = amplitudes.cpu()
        weights_cpu = weights.cpu()
        multiplicities_cpu = multiplicities.to(dtype=torch.float64).cpu()
        # If we do not convert it to float64, later, torch int multiplied with python native float give torch float32

        self._count += multiplicities_cpu.sum().item()

        params_and_targets = []
        configurations_cpu_s = []

        result = [{
            name: {
                positions: 0.0 for positions, observer in observers.items()
            } for name, observers in self._observer.items()
        } for _ in range(batch_size)]
        whole_result = [{name: 0.0 for name in self._observer} for _ in range(batch_size)]

        quantum_chemistry_term = [0 for _ in range(batch_size)]
        if "quantum_chemistry_term" in self.owner.attribute:
            with torch_grad(False):
                # Precalculate hopping
                # batch_size * site * site
                # we only calculate site a and site b exchange here
                # since for quantum chemistry model hopping is just exchange if one has electron and the other is empty
                # configurations_cpu : batch * L1 * L2 * 1
                # configurations_precalc : (site * site-1 / 2) * batch * L1 * L2 * 1
                configurations_precalc = configurations.unsqueeze(0).repeat(
                    [self.owner.site_number * (self.owner.site_number - 1) // 2 + 1, 1, 1, 1, 1])
                for al1, al2 in self.owner.sites():
                    for bl1, bl2 in self.owner.sites():
                        ai = al1 * self.owner.L2 + al2
                        bi = bl1 * self.owner.L2 + bl2
                        if ai < bi:
                            i = (2 * self.owner.site_number - ai - 1) * ai // 2 + (bi - ai - 1) + 1
                            a = configurations_precalc[i, :, al1, al2, :].clone()
                            b = configurations_precalc[i, :, bl1, bl2, :]
                            configurations_precalc[i, :, al1, al2, :] = b
                            configurations_precalc[i, :, bl1, bl2, :] = a
                amplitudes_precalc = self.owner(configurations_precalc.reshape([-1, self.owner.L1, self.owner.L2, 1]),
                                                enable_grad=False).reshape([
                                                    self.owner.site_number * (self.owner.site_number - 1) // 2 + 1,
                                                    batch_size
                                                ])
                amplitudes_precalc_cpu = amplitudes_precalc.cpu()
                gradients_precalc_norm = torch.zeros_like(amplitudes_precalc_cpu)
                gradients_precalc_conj = torch.zeros_like(amplitudes_precalc_cpu)

                for batch_index in range(batch_size):
                    configuration_cpu = configurations_cpu[batch_index]
                    amplitude = amplitudes_cpu[batch_index]
                    weight = weights_cpu[batch_index]
                    multiplicity = multiplicities_cpu[batch_index]
                    reweight = (amplitude.abs()**2 / weight).item()  # <psi|s|psi> / p(s)
                    energy = 0
                    for positions, observer, exists in self.owner.attribute["quantum_chemistry_term"][0]:
                        if not all(configuration_cpu[exist] == 1 for exist in exists):
                            continue
                        body = 2
                        element_pool = index_tensor_element(observer)
                        positions_configuration = tuple(configuration_cpu[l1l2o].item() for l1l2o in positions)
                        if positions_configuration not in element_pool:
                            continue
                        sub_energy = 0
                        for positions_configuration_s, item in element_pool[positions_configuration].items():
                            configuration_cpu_s = configuration_cpu.clone()
                            for l1l2o, value in zip(positions, positions_configuration_s):
                                configuration_cpu_s[l1l2o] = value
                            # we only have hopping item, since position configuration found in element pool, it must be a swap item.
                            ((al1, al2, _), (bl1, bl2, _)) = positions
                            ai = al1 * self.owner.L2 + al2
                            bi = bl1 * self.owner.L2 + bl2
                            ai, bi = sorted([ai, bi])
                            i = (2 * self.owner.site_number - ai - 1) * ai // 2 + (bi - ai - 1) + 1
                            value = item * self._fermi_sign(
                                configuration_cpu, configuration_cpu_s,
                                positions) * amplitudes_precalc_cpu[i, batch_index].conj() / amplitude.conj()
                            sub_energy = sub_energy + value
                            if self._enable_gradient:
                                gradients_precalc_conj[i, batch_index] += multiplicity * reweight * value / 2
                        energy = energy + sub_energy
                        if self._enable_gradient:
                            gradients_precalc_norm[0, batch_index] += multiplicity * reweight * sub_energy / 2
                    for positions_1, observer_1, positions_2, observer_2, exists in self.owner.attribute[
                            "quantum_chemistry_term"][1]:
                        if not all(configuration_cpu[exist] == 1 for exist in exists):
                            continue
                        body_1 = body_2 = 2
                        element_pool_1 = index_tensor_element(observer_1)
                        element_pool_2 = index_tensor_element(observer_2)
                        positions_configuration_1 = tuple(configuration_cpu[l1l2o].item() for l1l2o in positions_1)
                        positions_configuration_2 = tuple(configuration_cpu[l1l2o].item() for l1l2o in positions_2)
                        if positions_configuration_1 not in element_pool_1:
                            continue
                        if positions_configuration_2 not in element_pool_2:
                            continue
                        sub_energy_1 = 0
                        for positions_configuration_s_1, item_1 in element_pool_1[positions_configuration_1].items():
                            configuration_cpu_s_1 = configuration_cpu.clone()
                            for l1l2o, value in zip(positions_1, positions_configuration_s_1):
                                configuration_cpu_s_1[l1l2o] = value
                            # we only have hopping item, since position configuration found in element pool, it must be a swap item.
                            ((al1, al2, _), (bl1, bl2, _)) = positions_1
                            ai = al1 * self.owner.L2 + al2
                            bi = bl1 * self.owner.L2 + bl2
                            ai, bi = sorted([ai, bi])
                            i = (2 * self.owner.site_number - ai - 1) * ai // 2 + (bi - ai - 1) + 1
                            value = item_1 * self._fermi_sign(
                                configuration_cpu, configuration_cpu_s_1,
                                positions_1) * amplitudes_precalc_cpu[i, batch_index].conj() / amplitude.conj()
                            sub_energy_1 = sub_energy_1 + value
                        sub_energy_2 = 0
                        for positions_configuration_s_2, item_2 in element_pool_2[positions_configuration_2].items():
                            configuration_cpu_s_2 = configuration_cpu.clone()
                            for l1l2o, value in zip(positions_2, positions_configuration_s_2):
                                configuration_cpu_s_2[l1l2o] = value
                            # we only have hopping item, since position configuration found in element pool, it must be a swap item.
                            ((al1, al2, _), (bl1, bl2, _)) = positions_2
                            ai = al1 * self.owner.L2 + al2
                            bi = bl1 * self.owner.L2 + bl2
                            ai, bi = sorted([ai, bi])
                            i = (2 * self.owner.site_number - ai - 1) * ai // 2 + (bi - ai - 1) + 1
                            value = item_2 * self._fermi_sign(
                                configuration_cpu, configuration_cpu_s_2,
                                positions_2) * amplitudes_precalc_cpu[i, batch_index].conj() / amplitude.conj()
                            sub_energy_2 = sub_energy_2 + value
                        energy = energy + sub_energy_1 * sub_energy_2.conj()
                        if self._enable_gradient:
                            for positions_configuration_s_1, item_1 in element_pool_1[positions_configuration_1].items(
                            ):
                                configuration_cpu_s_1 = configuration_cpu.clone()
                                for l1l2o, value in zip(positions_1, positions_configuration_s_1):
                                    configuration_cpu_s_1[l1l2o] = value
                                # we only have hopping item, since position configuration found in element pool, it must be a swap item.
                                ((al1, al2, _), (bl1, bl2, _)) = positions_1
                                ai = al1 * self.owner.L2 + al2
                                bi = bl1 * self.owner.L2 + bl2
                                ai, bi = sorted([ai, bi])
                                i = (2 * self.owner.site_number - ai - 1) * ai // 2 + (bi - ai - 1) + 1
                                value = item_1 * self._fermi_sign(
                                    configuration_cpu, configuration_cpu_s_1,
                                    positions_1) * amplitudes_precalc_cpu[i, batch_index].conj() / amplitude.conj()
                                gradients_precalc_conj[
                                    i, batch_index] += multiplicity * reweight * value * sub_energy_2.conj() / 2

                            for positions_configuration_s_2, item_2 in element_pool_2[positions_configuration_2].items(
                            ):
                                configuration_cpu_s_2 = configuration_cpu.clone()
                                for l1l2o, value in zip(positions_2, positions_configuration_s_2):
                                    configuration_cpu_s_2[l1l2o] = value
                                # we only have hopping item, since position configuration found in element pool, it must be a swap item.
                                ((al1, al2, _), (bl1, bl2, _)) = positions_2
                                ai = al1 * self.owner.L2 + al2
                                bi = bl1 * self.owner.L2 + bl2
                                ai, bi = sorted([ai, bi])
                                i = (2 * self.owner.site_number - ai - 1) * ai // 2 + (bi - ai - 1) + 1
                                value = item_2 * self._fermi_sign(
                                    configuration_cpu, configuration_cpu_s_2,
                                    positions_2) * amplitudes_precalc_cpu[i, batch_index].conj() / amplitude.conj()
                                gradients_precalc_norm[
                                    i, batch_index] += multiplicity * reweight * sub_energy_1 * value.conj() / 2

                    whole_result[batch_index]["energy"] += complex(energy)
                    quantum_chemistry_term[batch_index] = complex(energy)
                if self._enable_gradient:
                    edelta = 0
                    for i in range(self.owner.site_number * (self.owner.site_number - 1) // 2 + 1):
                        for j in range(batch_size):
                            if gradients_precalc_norm[i, j] != 0 or gradients_precalc_conj[i, j] != 0:
                                configuration = configurations_precalc[i, j]
                                amplitude = self.owner(configuration.unsqueeze(0), enable_grad=True)
                                grad = self.owner.holes(amplitude)
                                edelta = edelta + gradients_precalc_norm[i, j] * grad
                                edelta = edelta + gradients_precalc_conj[i, j] * grad.conj()
                    if self._EDelta is None:
                        self._EDelta = edelta
                    else:
                        self._EDelta += edelta

        for batch_index in range(batch_size):
            configuration_cpu = configurations_cpu[batch_index]
            amplitude = amplitudes_cpu[batch_index]
            inv_amplitude_conj = 1 / amplitude.conj()
            parity = torch.gather(self.owner.op_pool, 3, configuration_cpu.unsqueeze(-1)).reshape([-1])
            for name, observers in self._observer.items():
                for positions, [observer, tensor_indices, element_pool] in observers.items():
                    body = len(positions)
                    positions_configuration = tuple(configuration_cpu.view([-1])[tensor_indices].tolist())
                    if positions_configuration not in element_pool:
                        continue
                    for positions_configuration_s, [item, tensor_positions_configuration_s,
                                                    [base_sign,
                                                     fermi_sign]] in element_pool[positions_configuration].items():
                        configuration_cpu_s = configuration_cpu.clone()
                        configuration_cpu_s.view([-1])[tensor_indices] = tensor_positions_configuration_s
                        # self.owner(configuration_s) to be multiplied
                        total_parity = ((parity[fermi_sign].sum() % 2 != 0) ^ base_sign)
                        value = (-1 if total_parity else +1) * (item * inv_amplitude_conj)
                        if torch.equal(configuration_cpu_s, configuration_cpu):
                            result[batch_index][name][positions] += amplitude.conj().item() * complex(value)
                            whole_result[batch_index][name] += amplitude.conj().item() * complex(value)
                        else:
                            params_and_targets.append((complex(value), (batch_index, name, positions)))
                            configurations_cpu_s.append(configuration_cpu_s)

        if len(configurations_cpu_s) != 0:
            configurations_s = torch.stack(configurations_cpu_s, dim=0).to(device=configurations.device)
            amplitudes_s = self.owner(configurations_s, enable_grad=False)
            amplitudes_cpu_s = amplitudes_s.cpu()
            for amplitude_s, [param, [batch_index, name, positions]] in zip(amplitudes_cpu_s, params_and_targets):
                result[batch_index][name][positions] += amplitude_s.conj().item() * param
                whole_result[batch_index][name] += amplitude_s.conj().item() * param

        for batch_index in range(batch_size):
            amplitude = amplitudes_cpu[batch_index]
            weight = weights_cpu[batch_index]
            multiplicity = multiplicities_cpu[batch_index]

            reweight = (amplitude.abs()**2 / weight).item()  # <psi|s|psi> / p(s)
            self._total_weight += multiplicity * reweight
            self._total_weight_square += multiplicity * reweight * reweight
            self._total_log_ws += multiplicity * amplitude.abs().log().item()

            for name, observers in self._observer.items():
                for positions in observers:
                    to_save = result[batch_index][name][positions].real
                    self._result_reweight[name][positions] += multiplicity * to_save * reweight
                    self._result_reweight_square[name][positions] += multiplicity * to_save * reweight**2
                    self._result_square_reweight_square[name][positions] += multiplicity * to_save**2 * reweight**2
                to_save = whole_result[batch_index][name].real
                self._whole_result_reweight[name] += multiplicity * to_save * reweight
                self._whole_result_reweight_square[name] += multiplicity * to_save * reweight**2
                self._whole_result_square_reweight_square[name] += multiplicity * to_save**2 * reweight**2
                if name == "energy":
                    self._total_imaginary_energy_reweight += multiplicity * whole_result[batch_index][
                        name].imag * reweight
                if name == "energy" and self._enable_gradient:
                    Es = whole_result[batch_index][name] - quantum_chemistry_term[batch_index]
                    # The EDelta contributed by quantum chemistry term has been collected before.
                    if self.owner.Tensor.is_real:
                        Es = Es.real

                    config = configurations[batch_index].unsqueeze(0)
                    value = self.owner(config, enable_grad=True)
                    holes = self.owner.holes(value)
                    # <psi|s|partial_x psi> / <psi|s|psi>

                    if self._Delta is None:
                        self._Delta = holes * reweight * multiplicity
                    else:
                        self._Delta += holes * reweight * multiplicity
                    if self._EDelta is None:
                        self._EDelta = holes * reweight * multiplicity * Es
                    else:
                        self._EDelta += holes * reweight * multiplicity * Es
                    if self._enable_natural:
                        self._Deltas.append((reweight * multiplicity, Es, holes))

    @property
    def count(self):
        """
        Count the observation times.
        """
        return int(self._count)

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
        Tensor
            The gradient for every tensor.
        """
        energy = self._total_energy_with_imaginary_part()
        b = ((self._EDelta / self._total_weight) - energy * (self._Delta / self._total_weight))
        b *= 2
        return b.real

    def natural_gradient(self, step=None, error=None, epsilon=None):
        """
        Get the energy natural gradient for every tensor.

        Parameters
        ----------
        step : int, optional
            conjugate gradient method step count.
        error : float, optional
            conjugate gradient method expected error.
        epsilon : float, optional
            epsilon to avoid singularity of metric.
        """
        if epsilon is None:
            epsilon = 1e-2
        if step is None and error is None:
            step = 20
            error = 0.0
        elif step is None:
            step = -1
        elif error is None:
            error = 0.0

        show("calculating natural gradient")
        energy = self._total_energy_with_imaginary_part()
        delta = self._Delta / self._total_weight

        Delta = []
        for reweight_s, energy_s, delta_s in self._Deltas:
            param = (reweight_s / self._total_weight)**(1 / 2)
            Delta.append((delta_s - delta) * param)
        self._Deltas = None
        Delta = torch.stack(Delta, dim=0)
        Delta = torch.view_as_real(Delta).permute(0, 2, 1).reshape([-1, Delta.shape[1]])

        metric_trace = torch.sum(Delta.conj() * Delta)
        allreduce_buffer(metric_trace)
        absolute_epsilon = epsilon * metric_trace / len(delta)

        # A x = b
        # DT r D x = DT r E

        # Delta, NsNp
        def D(v):
            # Ns Np * Np => Ns
            return Delta @ v

        def DT(v):
            # Np Ns * Ns => Np
            result = torch.conj(Delta.T) @ v
            allreduce_buffer(result)
            return result

        def A(v):
            return DT(D(v))

        # b = DT(Energy)
        b = ((self._EDelta / self._total_weight) - energy * (self._Delta / self._total_weight))
        b = 2 * b.real
        b_square = torch.dot(torch.conj(b), b).real

        x = torch.zeros_like(b)
        r = b - A(x)
        p = r
        r_square = torch.dot(torch.conj(r), r).real
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
            pAp = (torch.conj(Dp) @ Dp + absolute_epsilon * p @ p).real
            allreduce_buffer(pAp)
            alpha = r_square / pAp
            x = x + alpha * p
            r = r - alpha * (DT(Dp) + absolute_epsilon * p)
            new_r_square = torch.dot(torch.conj(r), r).real
            beta = new_r_square / r_square
            r_square = new_r_square
            p = r + beta * p
            t += 1

        showln(f"natural gradient calculated step={t} r^2/b^2={r_square/b_square} {reason}")
        return x.real

    def normalization_parameter(self):
        """
        Get the normalization parameter.

        Returns
        -------
        float
            The normalization parameter.
        """
        return self._total_log_ws / self._count
