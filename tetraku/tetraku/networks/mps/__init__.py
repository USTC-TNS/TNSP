#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2024 Hao Zhang<zh970205@mail.ustc.edu.cn>
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


class WaveFunction(torch.nn.Module):

    def __init__(
        self,
        *,
        L1,
        L2,
        orbit_num,
        physical_dim,
        hidden_dim,
        is_complex,
    ):
        super().__init__()
        self.L1 = L1
        self.L2 = L2
        self.orbit_num = orbit_num
        self.sites = L1 * L2 * orbit_num
        self.physical_dim = physical_dim
        self.hidden_dim = hidden_dim
        self.is_complex = is_complex

        self.tensor = torch.nn.ParameterList(
            torch.randn([physical_dim, hidden_dim, hidden_dim]) for _ in range(self.sites))

    def forward(self, x):
        dtype = next(self.parameters()).dtype
        x = x.reshape([x.size(0), -1])
        vector = torch.ones([x.size(0), 1, self.hidden_dim], dtype=dtype, device=x.device)
        for xi, tensori in zip(x.permute(1, 0), self.tensor):
            # config * hidden * hidden
            tensori = tensori.unsqueeze(0).expand([x.size(0), -1, -1, -1])
            # batch, config * hidden * hidden
            # batch
            xi = xi.reshape([-1, 1, 1, 1]).expand([-1, -1, self.hidden_dim, self.hidden_dim])
            # batch, config=1, hidden=1, hidden=1
            matrix = torch.gather(tensori, 1, xi).squeeze(1)
            # batch, hidden, hidden
            vector = torch.matmul(vector, matrix)
        vector = torch.matmul(vector, torch.ones([x.size(0), self.hidden_dim, 1], dtype=dtype, device=x.device))
        vector = vector.squeeze(-1).squeeze(-1)
        return vector


def network(
    state,
    hidden_dim,
):
    max_orbit_index = max(orbit for [l1, l2, orbit], edge in state.physics_edges)
    max_physical_dim = max(edge.dimension for [l1, l2, orbit], edge in state.physics_edges)
    network = WaveFunction(
        L1=state.L1,
        L2=state.L2,
        orbit_num=max_orbit_index + 1,
        physical_dim=max_physical_dim,
        hidden_dim=hidden_dim,
        is_complex=state.Tensor.is_complex,
    ).double()
    return network
