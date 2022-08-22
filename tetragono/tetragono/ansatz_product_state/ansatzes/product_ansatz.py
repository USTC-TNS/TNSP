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


class ProductAnsatz(AbstractAnsatz):

    __slots__ = ["owner", "ansatzes", "delta_part"]

    def __init__(self, owner, ansatzes):
        """
        An auxiliaries ansatz for the product of several other ansatzes.

        Parameters
        ----------
        owner : AnsatzProductState
            The ansatz product state used to create open string.
        ansatzes : list[AbstractAnsatz]
            The list of other ansatzes
        """
        self.owner: AnsatzProductState = owner
        self.ansatzes = ansatzes

        self.delta_part = []
        index = 0
        for ansatz in self.ansatzes:
            count = ansatz.buffer_count(None)
            self.delta_part.append(slice(index, index + count))
            index += count

    def weight_and_delta(self, configurations, calculate_delta):
        number = len(configurations)
        weights = [1.0 for _ in range(number)]
        if calculate_delta:
            deltas = [[] for _ in range(number)]
        else:
            deltas = None
        for ansatz in self.ansatzes:
            sub_weights, sub_deltas = ansatz.weight_and_delta(configurations, calculate_delta)
            for i in range(number):
                weights[i] *= sub_weights[i]
            if calculate_delta:
                for i in range(number):
                    deltas[i].append(sub_deltas[i] / sub_weights[i])

        if calculate_delta:
            for i in range(number):
                deltas[i] = np.concatenate(deltas[i]) * weights[i]
        return weights, deltas

    def refresh_auxiliaries(self):
        for ansatz in self.ansatzes:
            ansatz.refresh_auxiliaries()

    def ansatz_prod_sum(self, a, b):
        result = 0.0
        for part, ansatz in zip(self.delta_part, self.ansatzes):
            if a is None:
                a_part = None
            else:
                a_part = a[part]
            if b is None:
                b_part = None
            else:
                b_part = b[part]
            result += ansatz.ansatz_prod_sum(a_part, b_part)
        return result

    def ansatz_conjugate(self, a):
        result = []
        for part, ansatz in zip(self.delta_part, self.ansatzes):
            if a is None:
                a_part = None
            else:
                a_part = a[part]
            result.append(ansatz.ansatz_conjugate(a_part))
        return np.concatenate(result)

    def buffers(self, delta):
        if delta is not None:
            delta = iter(delta)
        for ansatz in self.ansatzes:
            iterator = ansatz.buffers(delta)
            try:
                recv = None
                while True:
                    recv = yield iterator.send(recv)
            except StopIteration:
                pass

    def elements(self, delta):
        if delta is not None:
            delta = iter(delta)
        for ansatz in self.ansatzes:
            iterator = ansatz.elements(delta)
            try:
                recv = None
                while True:
                    recv = yield iterator.send(recv)
            except StopIteration:
                pass

    def buffer_count(self, delta):
        delta = iter(delta)
        return sum(ansatz.buffer_count(delta) for ansatz in self.ansatzes)

    def element_count(self, delta):
        delta = iter(delta)
        return sum(ansatz.element_count(delta) for ansatz in self.ansatzes)

    def buffers_for_mpi(self, delta):
        if delta is not None:
            delta = iter(delta)
        for ansatz in self.ansatzes:
            yield from ansatz.buffers_for_mpi(delta)

    def normalize_ansatz(self):
        for ansatz in self.ansatzes:
            ansatz.normalize_ansatz()
