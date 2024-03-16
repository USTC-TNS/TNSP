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


class Mapping(torch.nn.Module):

    def __init__(self, M, N):
        super().__init__()
        self.M = M
        self.N = N
        self.register_buffer('comb', self.compute_combinations(M, N))
        self.space = self.comb[M, N].item()

    def forward(self, x):
        # x : batch * M
        index = torch.zeros([x.shape[0]], dtype=torch.int64, device=x.device)
        left = torch.ones([x.shape[0]], dtype=torch.int64, device=x.device) * self.N
        for i in range(self.M):
            xi = x[:, i]
            index = index + self.comb[self.M - 1 - i, left] * xi
            left = left - xi
        return index

    def compute_combinations(self, M, N):
        c = torch.zeros((M + 1, N + 1), dtype=torch.int64)
        for m in range(0, M + 1):
            c[m, 0] = 1
        for m in range(1, M + 1):
            for n in range(1, min(m, N) + 1):
                c[m, n] = c[m - 1, n - 1] + c[m - 1, n]
        return c


class RealWaveFunction(torch.nn.Module):

    def __init__(self, L1, L2, orbit, dimension, spin_up, spin_down):
        super().__init__()
        assert dimension == 2
        self.half_site = L1 * L2 * orbit // 2
        self.spin_up = spin_up
        self.spin_down = spin_down
        self.model_up = Mapping(self.half_site, self.spin_up)
        self.model_down = Mapping(self.half_site, self.spin_down)
        self.wave_function = torch.nn.Parameter(
            torch.randn(
                [self.model_up.space, self.model_down.space],
                dtype=torch.float64,
            ))

    def forward(self, x):
        # x : batch * L1 * L2 * orbit
        x = x.flatten(start_dim=1)
        # x : batch * site
        x_up = x[:, 0::2]
        x_down = x[:, 1::2]
        # x_up/down : batch * site/2
        i_up = self.model_up(x_up)
        i_down = self.model_down(x_down)
        # i_up/down : batch
        valid = torch.logical_and(x_up.sum(dim=-1) == self.spin_up, x_down.sum(dim=-1) == self.spin_down)
        return torch.where(valid, self.wave_function[i_up * valid, i_down * valid], 0)


class ComplexWaveFunction(torch.nn.Module):

    def __init__(self, L1, L2, orbit, dimension, spin_up, spin_down):
        super().__init__()
        self.real = RealWaveFunction(L1, L2, orbit, dimension, spin_up, spin_down)
        self.imag = RealWaveFunction(L1, L2, orbit, dimension, spin_up, spin_down)

    def forward(self, x):
        return self.real(x) + 1j * self.imag(x)


class WaveFunction(torch.nn.Module):

    def __init__(self, L1, L2, orbit, dimension, is_complex, spin_up, spin_down):
        super().__init__()
        if is_complex:
            self.model = ComplexWaveFunction(L1, L2, orbit, dimension, spin_up, spin_down)
        else:
            self.model = RealWaveFunction(L1, L2, orbit, dimension, spin_up, spin_down)

    def forward(self, x):
        return self.model(x)


def network(state, spin_up, spin_down):
    max_orbit_index = max(orbit for [l1, l2, orbit], edge in state.physics_edges)
    max_physical_dim = max(edge.dimension for [l1, l2, orbit], edge in state.physics_edges)
    network = WaveFunction(
        L1=state.L1,
        L2=state.L2,
        orbit=max_orbit_index + 1,
        dimension=max_physical_dim,
        is_complex=state.Tensor.is_complex,
        spin_up=spin_up,
        spin_down=spin_down,
    )
    return network
