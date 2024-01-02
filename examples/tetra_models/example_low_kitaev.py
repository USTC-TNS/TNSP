#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2021-2024 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
import kitaev
import kitaev.Sz

TAT.random.seed(2333)

# Create abstrace lattice first and cast it to su lattice
abstract_lattice = kitaev.abstract_lattice(L1=2, L2=2, D=4, Jx=1, Jy=1, Jz=1)
su_lattice = tet.SimpleUpdateLattice(abstract_lattice)

# Use pythonic style, aka, piclle, to save data.
# But should check mpi rank when saving data.
tet.write_to_file(su_lattice, "/dev/null")

su_lattice.update(100, 0.01, 5)

# showln is a helper function, which will only call print in proc 0
ex_lattice = tet.conversion.simple_update_lattice_to_exact_state(su_lattice)
tet.showln("Exact energy is", ex_lattice.observe_energy())
ex_lattice.update(100, 4)
tet.showln("Exact energy is", ex_lattice.observe_energy())

gm_lattice = tet.conversion.simple_update_lattice_to_sampling_lattice(su_lattice)
# To run gradient, create observer first
observer1 = tet.Observer(gm_lattice,
                         enable_energy=True,
                         enable_gradient=True,
                         enable_natural_gradient=True,
                         observer_set={"Sz": kitaev.Sz.measurement(gm_lattice)})
# You can create another observer
observer2 = tet.Observer(gm_lattice, enable_energy=True, enable_gradient=True, enable_natural_gradient=True)
# Run gradient
for grad_step in range(10):
    # Choose observer
    if grad_step % 2 == 0:
        observer = observer1
    else:
        observer = observer2
    # Prepare sampling environment
    with tet.seed_differ, observer:
        # create sampling object and do sampling
        sampling = tet.DirectSampling(gm_lattice, cut_dimension=8, restrict_subspace=None, double_layer_cut_dimension=4)
        for sampling_step in range(1000):
            observer(*sampling())
    tet.showln("grad", grad_step, *observer.energy)
    # Get Sz measure result
    if observer == observer1:
        tet.showln("   Sz:", observer.result["Sz"])
    # Get gradient
    grad = observer.natural_gradient(step=20, epsilon=0.01)
    # Maybe you want to use momentum
    if grad_step == 0:
        total_grad = grad
    else:
        total_grad = total_grad * 0.9 + grad * 0.1
    # Randomize gradient
    this_grad = tet.lattice_randomize(total_grad)
    # Apply gradient
    gm_lattice.apply_gradient(gm_lattice.fix_relative_to_lattice(this_grad), 0.01)
    # Fix gauge
    gm_lattice.expand_dimension(1.0, 0)
    # Bcast buffer to avoid numeric error
    gm_lattice.bcast_lattice()
    # Maybe you want to save file
    tet.write_to_file(gm_lattice, "/dev/null")
