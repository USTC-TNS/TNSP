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

Tensor = TAT.U1.D.Tensor
Edge = TAT.U1.Edge
Sym = TAT.U1.Symmetry


def main(N, T, S):
    hamiltonian = Tensor(["O0", "O1", "I0", "I1"], [Edge([0, 1]), Edge([0, 1]), Edge([0, -1]), Edge([0, -1])]).zero()
    hamiltonian[{"O0": (Sym(0), 0), "O1": (Sym(0), 0), "I0": (Sym(0), 0), "I1": (Sym(0), 0)}] = 1 / 4.
    hamiltonian[{"O0": (Sym(0), 0), "O1": (Sym(1), 0), "I0": (Sym(0), 0), "I1": (Sym(-1), 0)}] = -1 / 4.
    hamiltonian[{"O0": (Sym(1), 0), "O1": (Sym(0), 0), "I0": (Sym(-1), 0), "I1": (Sym(0), 0)}] = -1 / 4.
    hamiltonian[{"O0": (Sym(1), 0), "O1": (Sym(1), 0), "I0": (Sym(-1), 0), "I1": (Sym(-1), 0)}] = 1 / 4.
    hamiltonian[{"O0": (Sym(1), 0), "O1": (Sym(0), 0), "I0": (Sym(0), 0), "I1": (Sym(-1), 0)}] = 2 / 4.
    hamiltonian[{"O0": (Sym(0), 0), "O1": (Sym(1), 0), "I0": (Sym(-1), 0), "I1": (Sym(0), 0)}] = 2 / 4.
    print(hamiltonian)

    state = Tensor(["T", *[f"P{n}" for n in range(N)]], [Edge([-(N // 2)]), *[Edge([0, 1]) for n in range(N)]]).randn()
    state /= state.norm_2()
    print(state)

    for t in range(T):
        # state <- (1 - t H) state
        new_state = state.copy()
        tH = S * hamiltonian
        for i in range(N - 1):
            # i and i+1
            new_state -= state.contract(tH, {(f"P{i}", "I0"), (f"P{i+1}", "I1")}).edge_rename({"O0": f"P{i}", "O1": f"P{i+1}"})
        amp = new_state.norm_2()
        state = new_state / amp
        print(t, (1 - amp) / S / N)


if __name__ == "__main__":

    def Display(lines, out):
        text = "\n".join(lines) + "\n"
        out.write(text)

    fire.core.Display = Display
    fire.Fire(main)
