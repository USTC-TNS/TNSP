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


class Marshall(AbstractAnsatz):

    __slots__ = ["owner", "mask_A", "SA"]

    def __init__(self, owner):
        super().__init__(owner)

        self.mask_A = np.array(
            [[1 if (l1 + l2) % 2 == 0 else 0 for l2 in range(self.owner.L2)] for l1 in range(self.owner.L1)])
        self.SA = self.mask_A.sum() / 2

    def weight_and_delta(self, configurations, calculate_delta):
        configs = [config._configuration for config in configurations]
        orbit0 = configs[0].export_orbit0(configs) * 2 - 1  # C * L1 * L2
        weights = (-1)**(self.SA - (orbit0 * self.mask_A * 0.5).reshape([orbit0.shape[0], -1]).sum(axis=1))
        if calculate_delta:
            deltas = [np.empty(0, dtype=object) for _ in weights]
        else:
            deltas = None
        return weights, deltas

    def refresh_auxiliaries(self):
        pass

    def ansatz_prod_sum(self, a, b):
        return 0

    def ansatz_conjugate(self, a):
        return a

    def tensors(self, delta):
        return
        yield

    def elements(self, delta):
        return
        yield

    def tensor_count(self, delta):
        return 0

    def element_count(self, delta):
        return 0

    def buffers(self, delta):
        return
        yield

    def show(self):
        return self.__class__.__name__
