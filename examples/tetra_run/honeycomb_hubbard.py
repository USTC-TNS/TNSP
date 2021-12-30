#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2020-2021 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

t = -1
U = 12


class FakeEdge:

    def __init__(self, direction):
        self.direction = direction

    def __getitem__(self, x):
        return (list(x), self.direction)


Fedge = FakeEdge(False)
Tedge = FakeEdge(True)


def rename_io(t, m):
    res = {}
    for i, j in m.items():
        res[f"I{i}"] = f"I{j}"
        res[f"O{i}"] = f"O{j}"
    return t.edge_rename(res)


def dot(res, *b):
    for i in b:
        res = res.contract(i, set())
    return res


# Ops Before Merge

# EPR pair: (F T)
CP = TAT.Fermi.D.Tensor(["O", "I", "T"], [Fedge[0, 1], Tedge[0, -1], Fedge[-1,]]).range(1)
CM = TAT.Fermi.D.Tensor(["O", "I", "T"], [Fedge[0, 1], Tedge[0, -1], Tedge[+1,]]).range(1)
C0C1 = rename_io(CP, {"": 0}).contract(rename_io(CM, {"": 1}), {("T", "T")})
C1C0 = rename_io(CP, {"": 1}).contract(rename_io(CM, {"": 0}), {("T", "T")})
CC = C0C1 + C1C0  # rank = 4

I = TAT.Fermi.D.Tensor(["O", "I"], [Fedge[0, 1], Tedge[0, -1]]).identity({("I", "O")})

N = TAT.Fermi.D.Tensor(["O", "I"], [Fedge[0, 1], Tedge[0, -1]]).zero()
N[{"I": 1, "O": 1}] = 1

# Ops After Merge
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
    rename_io(I, {"": 2}),
    rename_io(I, {"": 3}),
) + dot(
    rename_io(CC, {
        0: 2,
        1: 3
    }),
    rename_io(I, {"": 0}),
    rename_io(I, {"": 1}),
)
CSCS = CSCS.merge_edge({
    "I0": ["I0", "I2"],
    "O0": ["O0", "O2"],
    "I1": ["I1", "I3"],
    "O1": ["O1", "O3"],
}, put_sign_in_H, {"O0", "O1"})

NN = dot(rename_io(N, {"": 0}), rename_io(N, {"": 1}))
NN = NN.merge_edge({
    "I0": ["I0", "I1"],
    "O0": ["O0", "O1"],
}, put_sign_in_H, {"O0"})


def create(L1, L2, D, T):
    state = tet.AbstractState(TAT.Fermi.D.Tensor, L1, L2)
    state.total_symmetry = T
    for i in range(L1):
        for j in range(L2):
            if (i, j) != (0, 0):
                state.physics_edges[i, j, 0] = [(0, 1), (1, 2), (2, 1)]
            if (i, j) != (L1 - 1, L2 - 1):
                state.physics_edges[i, j, 1] = [(0, 1), (1, 2), (2, 1)]
    for i in range(L1):
        for j in range(L2):
            if (i, j) != (0, 0):
                state.hamiltonians[(i, j, 0),] = U * NN
            if (i, j) != (L1 - 1, L2 - 1):
                state.hamiltonians[(i, j, 1),] = U * NN
            if (i, j) != (0, 0) and (i, j) != (L1 - 1, L2 - 1):
                state.hamiltonians[(i, j, 0), (i, j, 1)] = t * CSCS
            if i != 0:
                state.hamiltonians[(i - 1, j, 1), (i, j, 0)] = t * CSCS
            if j != 0:
                state.hamiltonians[(i, j - 1, 1), (i, j, 0)] = t * CSCS

    state = tet.AbstractLattice(state)
    tt = T / state.L1
    for l1 in range(state.L1 - 1):
        Q = int(T * (state.L1 - l1 - 1) / state.L1)
        state.virtual_bond[l1, 0, "D"] = [(Q - 1, D), (Q, D), (Q + 1, D)]
    for l1 in range(state.L1 - 1):
        for l2 in range(1, state.L2):
            state.virtual_bond[l1, l2, "D"] = [(0, D)]
    for l1 in range(state.L1):
        for l2 in range(state.L2 - 1):
            Q = int(tt * (state.L2 - l2 - 1) / state.L2)
            state.virtual_bond[l1, l2, "R"] = [(Q - 1, D), (Q, D), (Q + 1, D)]

    state = tet.SimpleUpdateLattice(state)
    return state
