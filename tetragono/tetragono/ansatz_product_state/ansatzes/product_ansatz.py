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
from .abstract_ansatz import AbstractAnsatz


class ProductAnsatz(AbstractAnsatz):

    __slots__ = ["ansatzes", "names", "delta_part"]

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
        super().__init__(owner)
        if isinstance(ansatzes, dict):
            self.ansatzes = []
            self.names = []
            for name, ansatz in ansatzes.items():
                self.names.append(name)
                self.ansatzes.append(ansatz)
        else:
            self.ansatzes = ansatzes
            self.names = [None for _ in self.ansatzes]

        self.delta_part = []
        index = 0
        for ansatz in self.ansatzes:
            count = ansatz.tensor_count(None)
            self.delta_part.append(slice(index, index + count))
            index += count

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.ansatzes[index]
        else:
            return self.ansatzes[self.names.index(index)]

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

    def tensors(self, delta):
        if delta is not None:
            delta = iter(delta)
        for ansatz in self.ansatzes:
            iterator = ansatz.tensors(delta)
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

    def tensor_count(self, delta):
        if delta is not None:
            delta = iter(delta)
        return sum(ansatz.tensor_count(delta) for ansatz in self.ansatzes)

    def element_count(self, delta):
        if delta is not None:
            delta = iter(delta)
        return sum(ansatz.element_count(delta) for ansatz in self.ansatzes)

    def buffers(self, delta):
        if delta is not None:
            delta = iter(delta)
        for ansatz in self.ansatzes:
            yield from ansatz.buffers(delta)

    def recovery_real(self, delta):
        result = []
        for part, ansatz in zip(self.delta_part, self.ansatzes):
            delta_part = delta[part]
            result.append(ansatz.recovery_real(delta_part))
        return np.concatenate(result)

    def normalize_ansatz(self, log_ws=None):
        weights = [ansatz.normalize_ansatz() for ansatz in self.ansatzes]
        weight_sum = sum(weights)
        if log_ws is None:
            return weight_sum
        if weight_sum == 0:
            return
        for weight, ansatz in zip(weights, self.ansatzes):
            ansatz.normalize_ansatz(log_ws * weight / weight_sum)

    def show(self):
        result = self.__class__.__name__
        for ansatz_name, ansatz in zip(self.names, self.ansatzes):
            if ansatz_name is None:
                name = "?"
            else:
                name = ansatz_name
            result += "\n\t" + name + " : " + ansatz.show().replace("\n", "\n\t")
        return result

    def lock(self, path=""):
        if path == "":
            for ansatz in self.ansatzes:
                ansatz.lock()
        else:
            path_split = path.split(".")
            head = path_split[0]
            tail = ".".join(path_split[1:])
            if head.isdigit():
                head = int(head)
            self[head].lock(tail)

    def unlock(self, path=""):
        if path == "":
            self.fixed = False
            for ansatz in self.ansatzes:
                ansatz.unlock()
        else:
            path_split = path.split(".")
            head = path_split[0]
            tail = ".".join(path_split[1:])
            if head.isdigit():
                head = int(head)
            self[head].unlock(tail)
