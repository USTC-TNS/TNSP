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

import pickle
import fire
import TAT
from square import *

if __name__ == "__main__":
    fire.core.Display = lambda lines, out: out.write("\n".join(lines) + "\n")

    def save(file_name: str, dimension: int = 2, seed: int = 0):
        TAT.set_random_seed(seed)
        lattice = SimpleUpdateLattice(4, 4, D=dimension)
        lattice.horizontal_bond_hamiltonian = SS
        lattice.vertical_bond_hamiltonian = SS
        with open(file_name, "wb") as file:
            pickle.dump(TAT.Name.dump(), file)
            pickle.dump(lattice, file)

    def update(file_name: str, step: int, delta_t: float, new_dimension: int = 0):
        lattice: SimpleUpdateLattice = None
        with open(file_name, "rb") as file:
            TAT.Name.load(pickle.load(file))
            lattice = pickle.load(file)
        lattice.simple_update(step, delta_t, new_dimension)
        with open(file_name, "wb") as file:
            pickle.dump(TAT.Name.dump(), file)
            pickle.dump(lattice, file)
        lattice_exact = ExactLattice(lattice)
        print("E1", lattice_exact.observe_energy())
        print("E2", lattice_exact.update())
        lattice_sampling = SamplingGradientLattice(lattice, 20)

    fire.Fire({"new": save, "update": update})
