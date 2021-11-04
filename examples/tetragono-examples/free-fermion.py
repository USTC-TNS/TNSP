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

import fire
import pickle
import TAT
import tetragono as tet

state = None


def create(file_name, L1, L2, D, T):
    state = tet.AbstractState(TAT.Fermi.D.Tensor, L1, L2)
    state.physics_edges = [0, 1]
    state.hamiltonians.vertical_bond = tet.common_variable.Fermi.CC
    state.hamiltonians.horizontal_bond = tet.common_variable.Fermi.CC
    state.total_symmetry = T
    t = T / state.L1

    state = tet.AbstractLattice(state)
    for l1 in range(state.L1 - 1):
        Q = int(T * (state.L1 - l1 - 1) / state.L1)
        state.virtual_bond[(l1, 0), "D"] = [(Q - 1, D), (Q, D), (Q + 1, D)]
    for l1 in range(state.L1 - 1):
        for l2 in range(1, state.L2):
            state.virtual_bond[(l1, l2), "D"] = [(0, D)]
    for l1 in range(state.L1):
        for l2 in range(state.L2 - 1):
            Q = int(t * (state.L2 - l2 - 1) / state.L2)
            state.virtual_bond[(l1, l2), "R"] = [(Q - 1, D), (Q, D), (Q + 1, D)]

    state = tet.SimpleUpdateLattice(state)
    with open(file_name, "wb") as file:
        pickle.dump(state, file)


def update(file_name, T, S, D, Dc):
    with open(file_name, "rb") as file:
        state = pickle.load(file)
    state.update(T, S, D)
    with open(file_name, "wb") as file:
        pickle.dump(state, file)
    state.initialize_auxiliaries(Dc)
    print(state.observe_energy())


if __name__ == "__main__":
    fire.core.Display = lambda lines, out: out.write("\n".join(lines) + "\n")
    fire.Fire()
