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

for _ in range(100):
    rank_A = np.random.randint(2, max_random)
    rank_B = np.random.randint(2, max_random)
    rank_contract = np.random.randint(1, np.min([rank_A, rank_B]))
    # print(rank_A, rank_B, rank_contract)

    contract_name_A = np.random.choice(range(rank_A), rank_contract, False)
    contract_name_B = np.random.choice(range(rank_B), rank_contract, False)

    dim_A = np.random.randint(1, max_random, size=rank_A)
    dim_B = np.random.randint(1, max_random, size=rank_B)
    dim_contract = np.random.randint(1, max_random, size=rank_contract)

    dim_A = [j if i not in contract_name_A else dim_contract[contract_name_A.tolist().index(i)] for i, j in enumerate(dim_A)]
    dim_B = [j if i not in contract_name_B else dim_contract[contract_name_B.tolist().index(i)] for i, j in enumerate(dim_B)]

    A = Tensor([f"A.{i}" for i in range(rank_A)], dim_A).randn()
    B = Tensor([f"B.{i}" for i in range(rank_B)], dim_B).randn()
    v_t = A.contract(B, {(f"A.{i}", f"B.{j}") for i, j in zip(contract_name_A, contract_name_B)}).block[{}]
    v_n = np.tensordot(A.block[{}], B.block[{}], [contract_name_A, contract_name_B])
    v_d = v_t - v_n
    print(np.max(np.abs(v_d)))
