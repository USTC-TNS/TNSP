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


def get_state(chain):
    res = None
    for t in chain:
        if res is None:
            res = t
        else:
            res = res.contract(t, {("R", "L")})
    return res


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


def main(N, T, S, P, D):
    chain = []
    for n in range(N):
        names = []
        edges = []
        if n == 0:
            names.append("L")
            edges.append(Edge([(-P, 1)], True))
        else:
            Q = int(P * (N - n) / N)
            names.append("L")
            edges.append(Edge([(-Q + 1, D), (-Q, D), (-Q - 1, D)], True))
        names.append(f"P{n}")
        edges.append(Edge([0, 1]))
        if n != N - 1:
            Q = int(P * (N - 1 - n) / N)
            names.append("R")
            edges.append(Edge([(Q - 1, D), (Q, D), (Q + 1, D)]))
        site = Tensor(names, edges).randn()
        site /= site.norm_2()
        chain.append(site)

    [print(i, t) for i, t in enumerate(chain)]

    op = (hamiltonian * (-S)).exponential({("I0", "O0"), ("I1", "O1")})
    iden = hamiltonian.same_shape().identity({("I0", "O0"), ("I1", "O1")})
    print("id  ", iden)
    print("1-tH", iden - hamiltonian * S)
    print("op  ", op.transpose(hamiltonian.name))
    op = iden - hamiltonian * S # TODO exp bug

    for t in range(T):
        t_norm = 1
        for n in range(N - 1):
            # n and n+1
            BIG = chain[n].contract(chain[n + 1], {("R", "L")}).contract(op, {(f"P{n}", "I0"), (f"P{n+1}", "I1")}).edge_rename({"O0": f"P{n}", "O1": f"P{n+1}"})
            U, s, V = BIG.svd({"L", f"P{n}"}, "R", "L", D, "L", "R")
            chain[n] = U
            chain[n + 1] = V
            norm = s.norm_2()
            t_norm *= norm
            s /= norm
            chain[n + 1] = chain[n + 1].contract(s, {("L", "R")})
        for n in reversed(range(N - 1)):
            # n and n+1
            BIG = chain[n].contract(chain[n + 1], {("R", "L")}).contract(op, {(f"P{n}", "I0"), (f"P{n+1}", "I1")}).edge_rename({"O0": f"P{n}", "O1": f"P{n+1}"})
            U, s, V = BIG.svd({"L", f"P{n}"}, "R", "L", D, "L", "R")
            chain[n] = U
            chain[n + 1] = V
            norm = s.norm_2()
            t_norm *= norm
            s /= norm
            chain[n] = chain[n].contract(s, {("R", "L")})
        #print("chain")
        #[print(tensor) for tensor in chain]
        print(t, "\t", log(t_norm) / (-S) / N / 2, "\t", get_energy(get_state(chain), N))


if __name__ == "__main__":

    def Display(lines, out):
        text = "\n".join(lines) + "\n"
        out.write(text)

    fire.core.Display = Display
    fire.Fire(main)
