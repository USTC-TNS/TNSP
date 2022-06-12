#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2020-2022 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
from TAT.No.D import Tensor

max_random = 8

for _ in range(100):
    rank_A = np.random.randint(2, max_random)
    rank_contract = np.random.randint(1, rank_A)
    U_leg = np.random.choice(range(rank_A), rank_contract, False)

    dim_A = np.random.randint(1, max_random, size=rank_A)

    A = Tensor([f"A.{i}" for i in range(rank_A)], dim_A.tolist()).randn()

    Q, R = A.qr("Q", {f"A.{i}" for i in U_leg}, "QR.Q", "QR.R")
    re_A = Q.contract(R, {("QR.Q", "QR.R")})
    diff = re_A - A

    QTQ = Q.contract(Q.edge_rename({"QR.Q": "new"}), {(name, name) for name in Q.names if name != "QR.Q"})
    QTQ = QTQ.blocks[QTQ.names]

    diff_Q = QTQ - np.identity(len(QTQ))

    print(np.max([diff.norm_max(), np.max(np.abs(diff_Q))]))
    R_block = R.blocks[R.names]
    # print(R_block.shape)
    # print(R_block.reshape([-1, R_block.shape[-1]]))
    # print(R_block.reshape([R_block.shape[0], -1]))
