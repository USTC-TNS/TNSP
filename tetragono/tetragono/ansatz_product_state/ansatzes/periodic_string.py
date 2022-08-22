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
from ..state import AnsatzProductState
from .abstract_ansatz import AbstractAnsatz


class PeriodicString(AbstractAnsatz):

    __slots__ = ["owner", "tensor", "is_horizontal", "auxiliaries", "tensor_shrink"]

    def __init__(self, owner, direction, cut_dimension):
        """
        Create an perodic string ansatz.

        The state should have only single orbit for every site.

        Parameters
        ----------
        owner : AnsatzProductState
            The ansatz product state used to create open string.
        direction : "V" | "v" | "H" | "h"
            The direction of the periodic string.
        cut_dimension : int
            The dimension cut of the string.
        """
        self.owner: AbstractProductState = owner
        self.tensor = self.owner.Tensor(["P", "L", "R"], [2, cut_dimension, cut_dimension]).randn()
        self.is_horizontal = direction in ["H", "h"]

        self.refresh_auxiliaries()

    def get_index_configuration(self, configuration):
        if self.is_horizontal:
            return [[configuration[l1, l2, 0][1] for l2 in range(self.owner.L2)] for l1 in range(self.owner.L1)]
        else:
            return [[configuration[l1, l2, 0][1] for l1 in range(self.owner.L1)] for l2 in range(self.owner.L2)]

    def weight_and_delta(self, configs, calculate_delta):
        index_configurations = [self.get_index_configuration(site_configuration) for site_configuration in configs]

        weights = []
        if calculate_delta:
            deltas = []
        else:
            deltas = None

        for index_configuration in index_configurations:
            line_weights = [self._closed_product(line_config) for line_config in index_configuration]
            weight = np.prod(line_weights)
            weights.append(weight)

            if calculate_delta:
                delta = self.tensor.same_shape().zero()
                for line_config, line_weight in zip(index_configuration, line_weights):
                    line_length = len(line_config)
                    for inline_index in range(line_length):
                        hole = self._tensor_product(
                            self._open_product(line_config[inline_index + 1:], left_to_right=False),
                            self._open_product(line_config[:inline_index], left_to_right=True)) * (weight / line_weight)
                        hole = hole.edge_rename({"L": "R", "R": "L"})
                        hole = hole.expand({"P": (line_config[inline_index], 2)})
                        delta += hole
                deltas.append(np.array([delta], dtype=object))

        return weights, deltas

    def _closed_product(self, config):
        return self._open_product(config).trace({("L", "R")})[{}]

    def _open_product(self, config, left_to_right=True):
        if len(config) == 0:
            return None
        config = tuple(config)
        if config not in self.auxiliaries:
            if left_to_right:
                self.auxiliaries[config] = self._tensor_product(self._open_product(config[:-1], True),
                                                                self.tensor_shrink[config[-1]])
            else:
                self.auxiliaries[config] = self._tensor_product(self.tensor_shrink[config[0]],
                                                                self._open_product(config[1:], False))
        return self.auxiliaries[config]

    def _tensor_product(self, a, b):
        if a is None:
            return b
        elif b is None:
            return a
        else:
            return a.contract(b, {("R", "L")})

    def refresh_auxiliaries(self):
        self.auxiliaries = {}
        self.tensor_shrink = [self.tensor.shrink({"P": i}) for i in range(2)]

    def ansatz_prod_sum(self, a, b):
        result = 0.0
        for ai, bi in zip(self.buffers(a), self.buffers(b)):
            dot = ai.contract(bi, {(name, name) for name in ai.names})[{}]
            result += dot
        return result

    def ansatz_conjugate(self, a):
        return np.array([i.conjugate(default_is_physics_edge=True) for i in self.buffers(a)])

    def buffers(self, delta):
        if delta is None:
            recv = yield self.tensor
            if recv is not None:
                self.tensor = recv
        else:
            recv = yield next(iter(delta))
            if recv is not None:
                delta[0] = recv

    def elements(self, delta):
        for tensor in self.buffers(delta):
            storage = tensor.transpose(self.tensor.names).storage
            length = len(storage)
            for i in range(length):
                recv = yield storage[i]
                if recv is not None:
                    if tensor.names != self.tensor.names:
                        raise RuntimeError("Trying to set tensor element which mismatches the edge names.")
                    storage[i] = recv

    def buffer_count(self, delta):
        return 1

    def element_count(self, delta):
        return sum(tensor.norm_num() for tensor in self.buffers(delta))

    def buffers_for_mpi(self, delta):
        for tensor in self.buffers(delta):
            yield tensor.storage

    def normalize_ansatz(self):
        self.tensor /= self.tensor.norm_max()
