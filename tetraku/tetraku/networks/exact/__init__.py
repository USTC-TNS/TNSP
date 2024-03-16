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


class RealWaveFunction(torch.nn.Module):

    def __init__(self, L1, L2, orbit, dimension):
        super().__init__()
        self.wave_function = torch.nn.Parameter(
            torch.randn(
                [dimension for _ in range(L1 * L2 * orbit)],
                dtype=torch.float64,
            ))

    def forward(self, x):
        # x : batch * L1 * L2 * orbit
        x = x.flatten(start_dim=-3)
        # x : batch * site
        stride = torch.tensor(self.wave_function.stride(), device=x.device)
        # stride : site
        index = (x * stride).sum(dim=-1)
        # index : batch
        wave = self.wave_function.view([-1])
        # wave : hilbert
        result = torch.gather(wave, 0, index)
        # result : batch
        # result[b] = wave[index[b]]
        return result


class ComplexWaveFunction(torch.nn.Module):

    def __init__(self, L1, L2, orbit, dimension):
        super().__init__()
        self.real = RealWaveFunction(L1, L2, orbit, dimension)
        self.imag = RealWaveFunction(L1, L2, orbit, dimension)

    def forward(self, x):
        return self.real(x) + 1j * self.imag(x)


class WaveFunction(torch.nn.Module):

    def __init__(self, L1, L2, orbit, dimension, is_complex):
        super().__init__()
        if is_complex:
            self.model = ComplexWaveFunction(L1, L2, orbit, dimension)
        else:
            self.model = RealWaveFunction(L1, L2, orbit, dimension)

    def forward(self, x):
        return self.model(x)


def network(state):
    max_orbit_index = max(orbit for [l1, l2, orbit], edge in state.physics_edges)
    max_physical_dim = max(edge.dimension for [l1, l2, orbit], edge in state.physics_edges)
    network = WaveFunction(
        L1=state.L1,
        L2=state.L2,
        orbit=max_orbit_index + 1,
        dimension=max_physical_dim,
        is_complex=state.Tensor.is_complex,
    )
    return network
