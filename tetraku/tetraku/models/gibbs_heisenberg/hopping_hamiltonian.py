#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2022-2023 Chao Wang<1023649157@qq.com>
# Copyright (C) 2023 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

import TAT
import tetragono as tet


def hopping_hamiltonians(state):
    pauli_x_pauli_x = tet.common_tensor.No.pauli_x_pauli_x.to(float)

    hamiltonian = {}

    for l1 in range(state.L1):
        for l2 in range(state.L2):
            hamiltonian[(l1, l2, 0),] = hamiltonian[(l1, l2, 1),] = pauli_x_pauli_x

    return hamiltonian
