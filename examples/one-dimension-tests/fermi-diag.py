#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2021 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
import sys
import fire
from math import log

Tensor = TAT.Fermi.D.Tensor
Edge = TAT.Fermi.Edge
Sym = TAT.Fermi.Symmetry

hamiltonian = Tensor(["O1", "O0", "I0", "I1"], [Edge([(0, 1), (1, 1)], False), Edge([(0, 1), (1, 1)], False), Edge([(0, 1), (-1, 1)], True), Edge([(0, 1), (-1, 1)], True)]).zero()
hamiltonian[{"O0": (Sym(1), 0), "O1": (Sym(0), 0), "I0": (Sym(0), 0), "I1": (Sym(-1), 0)}] = 1
hamiltonian[{"O0": (Sym(0), 0), "O1": (Sym(1), 0), "I0": (Sym(-1), 0), "I1": (Sym(0), 0)}] = 1
print(hamiltonian)


def get_energy(state, N):
    name_pair = {("L", "L")} | {(f"P{n}", f"P{n}") for n in range(N)}
    statestate = state.conjugate().contract(state, name_pair)
    Hstate = None
    for n in range(N - 1):
        # n and n+1
        this = state.contract(hamiltonian, {(f"P{n}", "I0"), (f"P{n+1}", "I1")}).edge_rename({"O0": f"P{n}", "O1": f"P{n+1}"})
        if Hstate is None:
            Hstate = this
        else:
            Hstate += this
    stateHstate = state.conjugate().contract(Hstate, name_pair)
    return float(stateHstate / statestate) / N


def main(N, T, S, P):
    state = Tensor(["L", *[f"P{n}" for n in range(N)]], [Edge([(-P, 1)], True), *[Edge([0, 1]) for n in range(N)]]).randn()
    #def g(v = [1, 0, 0]):
    #    return v.pop()
    #state.set(g)
    state /= state.norm_2()
    print("state", state)

    op = ((-S) * hamiltonian).exponential({("I0", "O0"), ("I1", "O1")})
    op = hamiltonian.same_shape().identity({("I0", "O0"), ("I1", "O1")}) - S * hamiltonian
    #op = S * hamiltonian
    #print("op", op)
    for t in range(T):
        new_state = state.copy()
        for i in range(N - 1):
            # i and i+1
            new_state = new_state.contract(op, {(f"P{i}", "I0"), (f"P{i+1}", "I1")}).edge_rename({"O0": f"P{i}", "O1": f"P{i+1}"})
            #print("new state", new_state.transpose(state.name))
            # new_state -= state.contract(op, {(f"P{i}", "I0"), (f"P{i+1}", "I1")}).edge_rename({"O0": f"P{i}", "O1": f"P{i+1}"})
        amp = new_state.norm_2()
        state = new_state / amp
        print(t, log(amp) / (-S) / N, get_energy(state, N))


if __name__ == "__main__":

    def Display(lines, out):
        text = "\n".join(lines) + "\n"
        out.write(text)

    fire.core.Display = Display
    fire.Fire(main)
