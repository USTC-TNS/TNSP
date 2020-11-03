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
import fire
import exact_diagnalization as ED


def main(n1, n2, step, print_energy: bool = False):
    print("Construct Lattice...")
    lattice = ED.SquareSpinLattice(n1, n2, 1)
    print("Construct Lattice done")

    print("Set Bonds...")
    for i in range(n1 - 1):
        for j in range(n2):
            lattice.set_bond([(i, j), (i + 1, j)], ED.SS)
    for i in range(n1):
        for j in range(n2 - 1):
            lattice.set_bond([(i, j), (i, j + 1)], ED.SS)
    print("Set Bonds Done")
    for _ in range(step):
        lattice.update()
        if print_energy:
            print(lattice.energy / (n1 * n2))


if __name__ == "__main__":

    def Display(lines, out):
        text = "\n".join(lines) + "\n"
        out.write(text)

    fire.core.Display = Display
    fire.Fire(main)
