#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2022-2023 Chao Wang<1023649157@qq.com>
# Copyright (C) 2023-2024 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
    # Two part, normal Hamiltonian and hopping between subspace
    SS = tet.common_tensor.No.SS.to(float)
    between_subspace = tet.common_tensor.No.SxSx.to(float) - tet.common_tensor.No.SySy.to(float)

    hamiltonian = {}

    for l1 in range(state.L1):
        for l2 in range(state.L2):
            hamiltonian[(l1, l2, 0), (l1, l2, 1)] = between_subspace
            if l1 != 0:
                hamiltonian[(l1 - 1, l2, 0), (l1, l2, 0)] = hamiltonian[(l1 - 1, l2, 1), (l1, l2, 1)] = SS
            if l2 != 0:
                hamiltonian[(l1, l2 - 1, 0), (l1, l2, 0)] = hamiltonian[(l1, l2 - 1, 1), (l1, l2, 1)] = SS

    return hamiltonian
