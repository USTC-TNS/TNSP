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

import torch
import TAT
import signal
import tetragono as tet
import tetragono.multiple_product_state


def create(L1, L2, J):
    state = tet.AbstractState(TAT.No.D.Tensor, L1, L2)
    state.physics_edges[...] = 2
    SS = tet.common_tensor.No.SS.to(float)
    state.hamiltonians["vertical_bond"] = -J * SS
    state.hamiltonians["horizontal_bond"] = -J * SS

    state = tet.multiple_product_state.MultipleProductState(state)

    max_int = 2**31
    random_int = TAT.random.uniform_int(0, max_int - 1)
    torch.manual_seed(random_int())
    network = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.MaxPool2d(kernel_size=(2, 2)),
        torch.nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)),
    ).double()
    ansatz = tet.multiple_product_state.ConvolutionalNeural(state, network)

    state.add_ansatz(ansatz)

    return state


# Create state
TAT.random.seed(43)

L1 = 4
L2 = 4
state = create(L1, L2, J=-1)

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
total_grad_step = 1000000
total_sampling_step = 1000
grad_step_size = 0.01
use_relative = True
with tet.SignalHandler(signal.SIGINT) as sigint_handler:
    for grad_step in range(total_grad_step):
        with observer, tet.seed_differ:
            for sampling_step in range(total_sampling_step):
                if sampling_step % tet.mpi_size == tet.mpi_rank:
                    observer(config)
                    for _ in range(state.site_number):
                        config = sampling.next(config)
                    tet.show(f"sampling {sampling_step}/{total_sampling_step}, energy={observer.energy}")
        tet.showln(f"gradient {grad_step}/{total_grad_step}, energy={observer.energy}")
        state.apply_gradient(observer.gradient, grad_step_size, relative=use_relative)
        if sigint_handler():
            break
