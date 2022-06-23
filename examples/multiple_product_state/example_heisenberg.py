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

# Create state
TAT.random.seed(43)

import heisenberg

L1 = 4
L2 = 4
state = heisenberg.abstract_state(L1, L2, J=-1)

state = tet.multiple_product_state.MultipleProductState(state)
import lx_cnn
import snake_string

state.add_ansatz(lx_cnn.ansatz(state, 64))
state.add_ansatz(snake_string.ansatz(state, "H", 2))
state.add_ansatz(snake_string.ansatz(state, "V", 2))

# Initialize config
config = {}
for l1 in range(L1):
    for l2 in range(L2):
        config[l1, l2, 0] = (l1 + l2) % 2

# Create sampling and observer

observer = tet.multiple_product_state.Observer(state)
observer.add_energy()
observer.enable_gradient()

# Gradient descent
total_grad_step = 1000000
total_sampling_step = 1000
grad_step_size = 0.01
use_relative = True
with tet.SignalHandler(signal.SIGINT) as sigint_handler:
    for grad_step in range(total_grad_step):
        with observer, tet.seed_differ:
            sampling = tet.multiple_product_state.Sampling(state, config, hopping_hamiltonians=None)
            for sampling_step in range(total_sampling_step):
                if sampling_step % tet.mpi_size == tet.mpi_rank:
                    observer(sampling.configuration)
                    for _ in range(state.site_number):
                        sampling()
                    tet.show(f"sampling {sampling_step}/{total_sampling_step}, energy={observer.energy}")
            config = sampling.configuration
        tet.showln(f"gradient {grad_step}/{total_grad_step}, energy={observer.energy}")
        state.apply_gradient(observer.gradient, grad_step_size, relative=use_relative)
        if sigint_handler():
            break
