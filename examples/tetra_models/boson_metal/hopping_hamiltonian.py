#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2022 Chao Wang<1023649157@qq.com>
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
    pauli_y_pauli_y = tet.common_tensor.No.pauli_y_pauli_y.to(float)
    pauli_z = tet.common_tensor.No.pauli_z.to(float)
    identity = tet.common_tensor.No.identity.to(float)

    hop_term = (pauli_x_pauli_x + pauli_y_pauli_y)
    n_term = (pauli_z + identity)

    hami = {}

    for l1 in range(state.L1):
        for l2 in range(state.L2):
            hami[(l1, l2, 0),] = n_term
            if l1 != 0:
                hami[(l1 - 1, l2, 0), (l1, l2, 0)] = hop_term
            if l2 != 0:
                hami[(l1, l2 - 1, 0), (l1, l2, 0)] = hop_term

    return hami
