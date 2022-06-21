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
import signal
import tetragono as tet
import tetragono.multiple_product_state
import tetragono.multiple_product_ansatz.open_string


def create(L1, L2, D, J):
    state = tet.AbstractState(TAT.No.D.Tensor, L1, L2)
    state.physics_edges[...] = 2
    SS = tet.common_tensor.No.SS.to(float)
    state.hamiltonians["vertical_bond"] = -J * SS
    state.hamiltonians["horizontal_bond"] = -J * SS

    state = tet.multiple_product_state.MultipleProductState(state)

    index_to_site = []
    for l1 in range(L1):
        for l2 in range(L2) if l1 % 2 == 0 else reversed(range(L2)):
            index_to_site.append((l1, l2, 0))
    ansatz_1 = tet.multiple_product_ansatz.open_string.OpenString(state, index_to_site, D)

    index_to_site = []
    for l2 in range(L2):
        for l1 in range(L1) if l2 % 2 == 0 else reversed(range(L1)):
            index_to_site.append((l1, l2, 0))
    ansatz_2 = tet.multiple_product_ansatz.open_string.OpenString(state, index_to_site, D)

    state.add_ansatz(ansatz_1)
    state.add_ansatz(ansatz_2)

    return state


# Create state
TAT.random.seed(43)

L1 = 4
L2 = 4
state = create(L1, L2, D=4, J=-1)

# Initialize config
config = {}
for l1 in range(L1):
    for l2 in range(L2):
        config[l1, l2, 0] = (l1 + l2) % 2

# Create sampling and observer
sampling = tet.multiple_product_state.Sampling(state, hopping_hamiltonians=None)

observer = tet.multiple_product_state.Observer(state)
observer.add_energy()
observer.enable_gradient()

# Gradient descent
total_grad_step = 10000
total_sampling_step = 1000
grad_step_size = 0.01
use_relative = True
with tet.SignalHandler(signal.SIGINT) as sigint_handler:
    for grad_step in range(total_grad_step):
        with observer, tet.seed_differ:
            for sampling_step in range(total_sampling_step):
                if sampling_step % tet.mpi_size == tet.mpi_rank:
                    observer(config)
                    config = sampling.next(config)
                    tet.show(f"sampling {sampling_step}/{total_sampling_step}, energy={observer.energy}")
        tet.showln(f"gradient {grad_step}/{total_grad_step}, energy={observer.energy}")
        state.apply_gradient(observer.gradient, grad_step_size, relative=use_relative)
        if sigint_handler():
            break
