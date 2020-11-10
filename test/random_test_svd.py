#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
from TAT.Tensor import DNo as Tensor

max_random = 8

for _ in range(1000):
    rank_A = np.random.randint(2, max_random)
    rank_contract = np.random.randint(1, rank_A)
    U_leg = np.random.choice(range(rank_A), rank_contract, False)

    dim_A = np.random.randint(1, max_random, size=rank_A)

    A = Tensor([f"A.{i}" for i in range(rank_A)], dim_A.tolist()).randn()

    U, S, V = A.svd({f"A.{i}" for i in U_leg}, "SVD.U", "SVD.V")
    re_A = U.multiple(S, "SVD.U", "U").contract(V, {("SVD.U", "SVD.V")})
    diff = re_A - A

    UTU = U.contract_all_edge(U.edge_rename({"SVD.U": "new"})).block[{}]
    VTV = V.contract_all_edge(V.edge_rename({"SVD.V": "new"})).block[{}]

    diff_U = UTU - np.identity(len(UTU))
    diff_V = VTV - np.identity(len(VTV))

    print(np.max([diff.norm_max(), np.max(np.abs(diff_U)), np.max(np.abs(diff_V))]))
