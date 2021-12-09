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


class FakeEdge:

    def __init__(self, direction):
        self.direction = direction

    def __getitem__(self, x):
        return (list(x), self.direction)


Fedge = FakeEdge(False)
Tedge = FakeEdge(True)

FE = ((0, 0), 0)
FD = ((+1, -1), 0)
FU = ((+1, +1), 0)
TE = ((0, 0), 0)
TD = ((-1, +1), 0)
TU = ((-1, -1), 0)

CC = TAT.FermiU1.D.Tensor(["O0", "O1", "I0", "I1"], [
    Fedge[(0, 0), (+1, -1), (+1, +1)], Fedge[(0, 0), (+1, -1), (+1, +1)], Tedge[(0, 0), (-1, +1),
                                                                                (-1, -1)], Tedge[(0, 0), (-1, +1),
                                                                                                 (-1, -1)]
]).zero()
CC[{"O0": FD, "O1": FE, "I0": TE, "I1": TD}] = 1
CC[{"O0": FU, "O1": FE, "I0": TE, "I1": TU}] = 1
CC[{"O0": FE, "O1": FD, "I0": TD, "I1": TE}] = 1
CC[{"O0": FE, "O1": FU, "I0": TU, "I1": TE}] = 1

SS = CC.same_shape().zero().transpose(["O0", "I0", "O1", "I1"])
# UUDD -1/2
# DDUU -1/2
# UDDU +1/2
# DUUD +1/2
SS[{"O0": FU, "I0": TU, "O1": FD, "I1": TD}] = -1 / 2
SS[{"O0": FD, "I0": TD, "O1": FU, "I1": TU}] = -1 / 2
SS[{"O0": FU, "I0": TD, "O1": FD, "I1": TU}] = +1 / 2
SS[{"O0": FD, "I0": TU, "O1": FU, "I1": TD}] = +1 / 2

t = 1
J = 0.4

H = (-t) * CC + J * SS
# print(H.transpose(["O0", "O1", "I1", "I0"]).clear_symmetry().blocks[["O0", "O1", "I0", "I1"]].reshape([9, 9]))


def create(L1, L2, D, T):
    state = tet.AbstractState(TAT.FermiU1.D.Tensor, L1, L2)
    state.physics_edges = [(0, 0), (+1, -1), (+1, +1)]  # empty, down, up
    state.hamiltonians.vertical_bond = H
    state.hamiltonians.horizontal_bond = H
    state.total_symmetry = (T * 2, 0)  # T up and T down
    print("total symmetry", state.total_symmetry)
    t = T / state.L1

    state = tet.AbstractLattice(state)
    for l1 in range(state.L1 - 1):
        Q = int(T * (state.L1 - l1 - 1) / state.L1)
        state.virtual_bond[(l1, 0), "D"] = [((2 * Q - 2, 0), D), ((2 * Q - 1, -1), D), ((2 * Q - 1, +1), D),
                                            ((2 * Q, -2), D), ((2 * Q, 0), D), ((2 * Q, +2), D), ((2 * Q + 1, -1), D),
                                            ((2 * Q + 1, +1), D), ((2 * Q + 2, 0), D)]
    for l1 in range(state.L1 - 1):
        for l2 in range(1, state.L2):
            state.virtual_bond[(l1, l2), "D"] = [((0, 0), D)]
    for l1 in range(state.L1):
        for l2 in range(state.L2 - 1):
            Q = int(t * (state.L2 - l2 - 1) / state.L2)
            state.virtual_bond[(l1, l2), "R"] = [((2 * Q - 2, 0), D), ((2 * Q - 1, -1), D), ((2 * Q - 1, +1), D),
                                                 ((2 * Q, -2), D), ((2 * Q, 0), D), ((2 * Q, +2), D),
                                                 ((2 * Q + 1, -1), D), ((2 * Q + 1, +1), D), ((2 * Q + 2, 0), D)]

    state = tet.SimpleUpdateLattice(state)
    return state


def configuration(state):
    L1 = state.L1
    L2 = state.L2
    UpD = state.total_symmetry.fermi
    UmD = state.total_symmetry.u1
    U = (UpD + UmD) // 2
    D = (UpD - UmD) // 2
    for l1 in range(L1):
        for l2 in range(L2):
            state.configuration[l1, l2] = ((0, 0), 0)
    randL1 = TAT.random.uniform_int(0, L1 - 1)
    randL2 = TAT.random.uniform_int(0, L2 - 1)
    u = 0
    while u < U:
        l1 = randL1()
        l2 = randL2()
        if state.configuration[l1, l2][0] == state.Symmetry(0, 0):
            state.configuration[l1, l2] = ((1, +1), 0)
            u += 1
    d = 0
    while d < D:
        l1 = randL1()
        l2 = randL2()
        if state.configuration[l1, l2][0] == state.Symmetry(0, 0):
            state.configuration[l1, l2] = ((1, -1), 0)
            d += 1
    if len(state.configuration.hole(()).edges("T").segment) == 0:
        return configuration(state)
