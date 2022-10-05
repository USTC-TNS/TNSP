#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2022 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
import tetragono as tet


class Reshape(torch.nn.Module):

    def __init__(self, L1, L2):
        super(Reshape, self).__init__()
        self.shape = [L1, L2]

    def forward(self, x):
        return x.view(list(x.shape)[:-1] + self.shape)


def single_layer(k, m0, m1, m2, L1, L2):
    padding = (k - 1) // 2
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=m0,
                        out_channels=m1,
                        kernel_size=(k, k),
                        stride=(1, 1),
                        padding=(padding, padding),
                        padding_mode="circular"),
        torch.nn.Flatten(-2, -1),
        torch.nn.MaxPool1d(kernel_size=2),
        torch.nn.ConvTranspose1d(in_channels=m1, out_channels=m2, kernel_size=2, stride=2, padding=0),
        Reshape(L1, L2),
    ).double()


def ansatz(state):
    """
    Create pbc version lx style deep cnn ansatz.

    The code here was designed by Xiao Liang.
    See https://link.aps.org/doi/10.1103/PhysRevB.98.104426 for more information.
    """
    torch.manual_seed(tet.seed_differ.random_int())
    L1 = state.L1
    L2 = state.L2
    network = torch.nn.Sequential(
        single_layer(5, 1, 64, 64, L1, L2),
        single_layer(5, 64, 32, 32, L1, L2),
        single_layer(3, 32, 32, 32, L1, L2),
        single_layer(3, 32, 32, 32, L1, L2),
        single_layer(3, 32, 32, 32, L1, L2),
        single_layer(3, 32, 32, 1, L1, L2),
    ).double()
    return tet.ansatz_product_state.ansatzes.ConvolutionalNeural(state, network)
