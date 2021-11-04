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


def calc_exact(file_name):
    with open(file_name, "rb") as file:
        state = pickle.load(file)
    state = tet.ExactState(state)
    state.update(2000, 0.5)


def create(file_name, L1, L2, D):
    state = tet.AbstractState(TAT.No.D.Tensor, L1, L2)
    state.physics_edges = 2
    state.hamiltonians.vertical_bond = tet.common_variable.No.SS
    state.hamiltonians.horizontal_bond = tet.common_variable.No.SS

    state = tet.AbstractLattice(state)
    state.virtual_bond[..., "R"] = D
    state.virtual_bond[..., "D"] = D

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
    print(state.exact_state().observe_energy())


def sampling(file_name):
    with open(file_name, "rb") as file:
        state = pickle.load(file)
    TAT.random.seed(1)
    state = tet.SamplingLattice(state, -1)
    for i in range(state.L1):
        for j in range(state.L2):
            state.configuration[i, j] = (i + j + 1) % 2

    sampling = tet.SweepSampling(state)
    observer = tet.Observer(state)
    observer.enable_gradient()
    for _ in range(100):
        observer.flush()
        for _ in range(100):
            observer(sampling())
            print(tet.common_variable.clear_line, "sampling result energy", observer.energy / (state.L1 * state.L2), end="\r")
        print()
        grad = observer.gradient
        for i in range(state.L1):
            for j in range(state.L2):
                state[i, j] -= 0.02 * grad[i][j]
        state.configuration.refresh_all()


if __name__ == "__main__":
    fire.core.Display = lambda lines, out: out.write("\n".join(lines) + "\n")
    fire.Fire()
