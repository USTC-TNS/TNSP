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

import TAT

import exact_diagnalization as ED

n1 = 4
n2 = 4

lattice = ED.SquareSpinLattice(n1, n2, 1)

for i in range(n1 - 1):
    for j in range(n2):
        lattice.set_bond([(i, j), (i + 1, j)], ED.SS)
for i in range(n1):
    for j in range(n2 - 1):
        lattice.set_bond([(i, j), (i, j + 1)], ED.SS)
for i in range(n1 - 1):
    for j in range(n2 - 1):
        lattice.set_bond([(i, j), (i + 1, j + 1)], ED.SS)

for _ in range(1000):
    lattice.update()
    print(lattice.energy / (n1 * n2))


def single_site_operator(n, op):
    return op.edge_rename({"I": f"{n}", "O": f"_{n}"})


def single_site_spin_vector(n):
    return [single_site_operator(n, ED.Sx), single_site_operator(n, ED.Sy), single_site_operator(n, ED.Sz)]


def kronecker(*args):
    l = list(args)
    res = l.pop()
    for i in l:
        same_name = set(i.name) & set(res.name)
        valid_name = (x.name for x in same_name if x.name[0] != '_')
        contract_set = set(((i, f"_{i}") for i in valid_name))
        res = res.contract(i, contract_set)
    return res


def vertex_operator(ns):
    ss = [single_site_spin_vector(n) for n in ns]  # op[3,3]

    def get(a, b, c):
        return kronecker(ss[0][a], ss[1][b], ss[2][c])

    return get(0, 1, 2) + get(1, 2, 0) + get(2, 0, 1) - get(2, 1, 0) - get(1, 0, 2) - get(0, 2, 1)


def vertex_operator_for_loop(i, j, up):
    if up:
        return vertex_operator([f"{i}.{j}", f"{i+1}.{j}", f"{i+1}.{j+1}"])
    else:
        return vertex_operator([f"{i}.{j}", f"{i}.{j+1}", f"{i+1}.{j+1}"])


print()
print("ORDER\n")
for i in range(n1 - 1):
    for j in range(n2 - 1):
        for k in [True, False]:
            for a in range(n1 - 1):
                for b in range(n2 - 1):
                    for c in [True, False]:
                        m = vertex_operator_for_loop(i, j, k)
                        n = vertex_operator_for_loop(a, b, c)
                        print(lattice.observe(kronecker(m, n)))
