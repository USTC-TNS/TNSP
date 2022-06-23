#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2021-2022 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
from boson_metal import abstract_lattice as create
from boson_metal.hopping_hamiltonian import hopping_hamiltonians
from boson_metal.restrict_Sz import restrict
from boson_metal.initial_state import initial_configuration

TAT.random.seed(2333)

# Create abstrace lattice first and cast it to su lattice
abstract_lattice = create(L1=10, L2=10, D=4, J=-1, K=3, mu=-1.6)
gm_lattice = tet.SamplingLattice(abstract_lattice)

# Create initial configuration
with tet.seed_differ:
    configuration = tet.Configuration(gm_lattice, -1)
    configuration = initial_configuration(configuration)
    config = configuration._configuration

# To run gradient, create observer first
observer = tet.Observer(
    gm_lattice,
    enable_energy=True,
    enable_gradient=True,
    enable_natural_gradient=True,
)
for grad_step in range(100):
    # Prepare sampling environment
    with tet.seed_differ, observer:
        # create sampling object and do sampling
        sampling = tet.SweepSampling(gm_lattice,
                                     cut_dimension=12,
                                     restrict_subspace=restrict,
                                     hopping_hamiltonians=hopping_hamiltonians(gm_lattice))
        sampling.configuration.load_configuration(config)
        for sampling_step in range(1000):
            if sampling_step % tet.mpi_size == tet.mpi_rank:
                possibility, configuration = sampling()
                observer(possibility, configuration)
                tet.show("sampling", sampling_step, "current energy is", *observer.energy)
        config = configuration._configuration
    tet.showln("grad", grad_step, *observer.energy)
    # Get gradient
    grad = observer.natural_gradient(step=20, epsilon=0.01)
    # Maybe you want to use momentum
    if grad_step == 0:
        total_grad = grad
    else:
        total_grad = total_grad * 0.9 + grad * 0.1
    # Apply gradient
    gm_lattice._lattice -= 0.01 * gm_lattice.fix_relative_to_lattice(total_grad)
    # Normalzie state
    observer.normalize_lattice()
