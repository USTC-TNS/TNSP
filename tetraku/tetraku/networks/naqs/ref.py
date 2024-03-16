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
from src.naqs.network.nade import *
# https://github.com/tomdbar/naqs-for-quantum-chemistry
# This python module is not well packaged.
# User need to install it by themselves before using this network.


class WaveFunction(torch.nn.Module):

    def __init__(
        self,
        *,
        L1,
        L2,
        orbit_num,
        physical_dim,
        is_complex,
        **kwargs,
    ):
        super().__init__()
        self.L1 = L1
        self.L2 = L2
        self.orbit_num = orbit_num
        self.sites = L1 * L2 * orbit_num
        assert physical_dim == 2
        assert is_complex == True

        assert "n_alpha_electrons" in kwargs
        assert "n_beta_electrons" in kwargs
        self.model = ComplexAutoregressiveMachine1D_OrbitalNade(
            self.sites,
            **kwargs,
        )

    def generate(self, batch_size, alpha=1):
        device = next(self.parameters()).device
        assert alpha == 1

        self.model.sample()
        config, count, probability, psi = self.model(batch_size)
        config = config.to(device=device)
        count = count.to(device=device)
        probability = probability.to(device=device)
        psi = psi.to(device=device)
        batch_size = count.size(0)
        # config : unique_batch * sites
        # count : unique_batch
        # probability : unique_batch
        # psi : unique_batch * 2 (log amp and theta)
        config = (config + 1) / 2
        config = config.to(torch.int64)
        config = config.reshape([batch_size, self.L1, self.L2, self.orbit_num])
        psi = torch.view_as_complex(psi).exp()
        probability = probability
        return config, psi.to(dtype=torch.complex128), probability.to(dtype=torch.float64), count.to(dtype=torch.int64)

    def forward(self, x):
        self.model.predict()
        x = x.reshape([x.size(0), -1])  # batch * site
        result = self.model(x * 2 - 1)  # batch * site/2 * 4 * 2
        # 4 order: 00, 10, 01, 11
        x_split = x.reshape([x.size(0), -1, 2])  # batch * site/2 * 2
        x_index = x_split[:, :, 0] + 2 * x_split[:, :, 1]  # batch * site/2
        log_psi = torch.gather(result, -2, x_index.unsqueeze(-1).unsqueeze(-1).expand([-1, -1, -1, 2]))
        # batch * site/2 * 1 * 2
        log_psi = torch.view_as_complex(log_psi)  # batch * site/2 * 1
        log_psi = log_psi.sum(dim=[1, 2])  # batch
        psi = log_psi.exp()
        return psi.to(dtype=torch.complex128)


def network(state, **kwargs):
    max_orbit_index = max(orbit for [l1, l2, orbit], edge in state.physics_edges)
    max_physical_dim = max(edge.dimension for [l1, l2, orbit], edge in state.physics_edges)
    network = WaveFunction(
        L1=state.L1,
        L2=state.L2,
        orbit_num=max_orbit_index + 1,
        physical_dim=max_physical_dim,
        is_complex=state.Tensor.is_complex,
        **kwargs,
    )
    return network
