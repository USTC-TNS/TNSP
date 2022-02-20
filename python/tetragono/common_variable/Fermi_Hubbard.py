#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2021-2022 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
from . import Fermi
from .Fermi import Tensor, rename_io, CC, I, N


def dot(res, *b):
    for i in b:
        res = res.contract(i, set())
    return res


# Merge two spin
# This is following sjdong's convension
put_sign_in_H = True

# site1 up: 0
# site2 up: 1
# site1 down: 2
# site2 down: 3
# CSCS = CC(0,1)I(2)I(3) + I(0)I(1)CC(2,3)
CSCS = dot(
    rename_io(CC, {
        0: 0,
        1: 1
    }),
    rename_io(I, {0: 2}),
    rename_io(I, {0: 3}),
) + dot(
    rename_io(CC, {
        0: 2,
        1: 3
    }),
    rename_io(I, {0: 0}),
    rename_io(I, {0: 1}),
)
CSCS = CSCS.merge_edge({
    "I0": ["I0", "I2"],
    "O0": ["O0", "O2"],
    "I1": ["I1", "I3"],
    "O1": ["O1", "O3"],
}, put_sign_in_H, {"O0", "O1"})

NN = dot(rename_io(N, {0: 0}), rename_io(N, {0: 1}))
NN = NN.merge_edge({
    "I0": ["I0", "I1"],
    "O0": ["O0", "O1"],
}, put_sign_in_H, {"O0"})
