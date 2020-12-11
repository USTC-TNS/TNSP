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

    class action:

        def __init__(self):
            TAT.random.seed(0)
            self.name = None
            self.lattice = None

        def show(self):
            print(self.lattice.__class__.__name__, "with M =", self.lattice.M, "and N =", self.lattice.N)
            return self

        def seed(self, seed: int):
            TAT.random.seed(seed)
            return self

        def file(self, name: str):
            self.name = name
            return self

        def open(self):
            with open(self.name, "rb") as file:
                TAT.Name.load(pickle.load(file))
                self.lattice = pickle.load(file)
            return self

        def new(self, M: int, N: int):
            self.lattice = SimpleUpdateLattice(M, N, D=2, d=2)
            self.lattice.horizontal_bond_hamiltonian = SS
            self.lattice.vertical_bond_hamiltonian = SS
            return self

        def update(self, step: int, delta_t: float, new_dimension: int):
            self.lattice = SimpleUpdateLattice(self.lattice)
            self.lattice.simple_update(step, delta_t, new_dimension)
            return self

        def equilibrate(self, step: int, Dc: int):
            if not isinstance(self.lattice, SamplingGradientLattice):
                self.lattice = SamplingGradientLattice(self.lattice, Dc=Dc)
                self.lattice.initialize_spin()
            self.lattice.equilibrate(step)
            return self

        def sampling(self, step: int, Dc: int):
            if not isinstance(self.lattice, SamplingGradientLattice):
                self.lattice = SamplingGradientLattice(self.lattice, Dc=Dc)
                self.lattice.initialize_spin()
            self.lattice.markov_chain(step, calculate_energy=True)
            return self

        def ergodic(self, Dc: int):
            if not isinstance(self.lattice, SamplingGradientLattice):
                self.lattice = SamplingGradientLattice(self.lattice, Dc=Dc)
                self.lattice.initialize_spin()
            self.lattice.ergodic(calculate_energy=True)
            return self

        def save(self):
            with open(self.name, "wb") as file:
                pickle.dump(TAT.Name.dump(), file)
                pickle.dump(self.lattice, file)
            return self

        def end(self):
            pass

    fire.Fire(action)
