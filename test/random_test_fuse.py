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
import TAT

max_random = 8

for i in range(max_random):
    a = TAT._internal.Name(str(i))

for _ in range(1000):
    rank_A = np.random.randint(3, max_random)
    rank_B = np.random.randint(3, max_random)
    rank_total = np.random.randint(2, np.min([rank_A, rank_B]))
    rank_contract = np.random.randint(1, rank_total)
    rank_fuse = rank_total - rank_contract

    total_leg_A = np.random.choice(range(rank_A), rank_total, False)
    total_leg_B = np.random.choice(range(rank_B), rank_total, False)
    contract_leg_A = total_leg_A[:rank_contract]
    contract_leg_B = total_leg_B[:rank_contract]
    fuse_leg_A = total_leg_A[rank_contract:]
    fuse_leg_B = total_leg_B[rank_contract:]

    # print("\n\nSTART\n\n")
    # print(rank_A, rank_B)
    # print(contract_leg_A)
    # print(contract_leg_B)
    # print(fuse_leg_A)
    # print(fuse_leg_B)

    name_list_A = [f"A.{i}" for i in range(rank_A)]
    name_list_B = [f"B.{i}" for i in range(rank_B)]
    for i in range(rank_fuse):
        name_list_A[fuse_leg_A[i]] = f"{i}"
        name_list_B[fuse_leg_B[i]] = f"{i}"

    dim_A = np.random.randint(1, max_random, rank_A).tolist()
    dim_B = np.random.randint(1, max_random, rank_B).tolist()
    for i in range(rank_total):
        dim_A[total_leg_A[i]] = dim_B[total_leg_B[i]] = np.random.randint(2, max_random)

    A = Tensor(name_list_A, dim_A).test()
    B = Tensor(name_list_B, dim_B).test()
    C = A.contract(B, {(f"A.{contract_leg_A[i]}", f"B.{contract_leg_B[i]}") for i in range(rank_contract)})
    # print(repr(A))
    # print(repr(B))
    # print(repr(C))
    a = A.block[{}]
    b = B.block[{}]

    index_A = [chr(ord('a') + i) for i in range(rank_A)]
    index_B = [chr(ord('A') + i) for i in range(rank_B)]
    index_C = []
    for i in range(rank_total):
        index_A[total_leg_A[i]] = index_B[total_leg_B[i]]
    # fuse的顺序是按照name自身顺序来的
    for i in fuse_leg_A:
        index_C.append(index_A[i])
    # 然后A的没有contract的边
    for i in range(rank_A):
        if i not in total_leg_A:
            index_C.append(index_A[i])
    # 然后A的没有contract的边
    for i in range(rank_B):
        if i not in total_leg_B:
            index_C.append(index_B[i])
    ein_conf = "".join(index_A) + "," + "".join(index_B) + "->" + "".join(index_C)
    # print(ein_conf)
    c = np.einsum(ein_conf, a, b)
    # print(c.shape)
    print(np.max(np.abs(C.block[{}] - c)))
