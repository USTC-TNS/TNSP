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


class FakeLinear(torch.nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros([dim_out]))

    def forward(self, x):
        shape = x.shape[:-1]
        prod = torch.tensor(shape).prod()
        return self.bias.view([1, -1]).expand([prod, -1]).view([*shape, -1])


def Linear(dim_in, dim_out):
    if dim_in == 0:
        return FakeLinear(dim_in, dim_out)
    else:
        return torch.nn.Linear(dim_in, dim_out)


class MLP(torch.nn.Module):

    def __init__(self, dim_input, dim_output, hidden_size):
        super().__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.hidden_size = hidden_size
        self.depth = len(hidden_size)

        self.model = torch.nn.Sequential(*(Linear(
            dim_input if i == 0 else hidden_size[i - 1],
            dim_output if i == self.depth else hidden_size[i],
        ) if j == 0 else torch.nn.SiLU() for i in range(self.depth + 1) for j in range(2) if i != self.depth or j != 1))

    def forward(self, x):
        return self.model(x)


class WaveFunction(torch.nn.Module):

    def __init__(
        self,
        *,
        L1,
        L2,
        orbit_num,
        physical_dim,
        is_complex,
        spin_up,
        spin_down,
        hidden_size,
        ordering,
    ):
        super().__init__()
        self.L1 = L1
        self.L2 = L2
        self.orbit_num = orbit_num
        self.sites = L1 * L2 * orbit_num // 2
        assert physical_dim == 2
        assert is_complex == True
        self.spin_up = spin_up
        self.spin_down = spin_down
        self.hidden_size = tuple(hidden_size)

        self.amplitude = torch.nn.ModuleList([MLP(i * 2, 4, self.hidden_size) for i in range(self.sites)])
        self.phase = torch.nn.ModuleList([MLP(i * 2, 4, self.hidden_size) for i in range(self.sites)])

        if isinstance(ordering, int) and ordering == +1:
            ordering = list(range(self.sites))
        if isinstance(ordering, int) and ordering == -1:
            ordering = list(reversed(range(self.sites)))
        self.register_buffer('ordering', torch.tensor(ordering, dtype=torch.int64), persistent=True)
        ordering_bak = torch.zeros(self.sites, dtype=torch.int64)
        ordering_bak.scatter_(0, self.ordering, torch.arange(self.sites))
        self.register_buffer('ordering_bak', ordering_bak, persistent=True)

    def mask(self, x):
        # x : batch * i * 2
        i = x.size(1)
        # number : batch * 2
        number = x.sum(dim=1)

        up_electron = number[:, 0]
        down_electron = number[:, 1]
        up_hole = i - up_electron
        down_hole = i - down_electron

        add_up_electron = up_electron < self.spin_up
        add_down_electron = down_electron < self.spin_down
        add_up_hole = up_hole < self.sites - self.spin_up
        add_down_hole = down_hole < self.sites - self.spin_down

        add_up = torch.stack([add_up_hole, add_up_electron], dim=-1).unsqueeze(-1)
        add_down = torch.stack([add_down_hole, add_down_electron], dim=-1).unsqueeze(-2)
        add = torch.logical_and(add_up, add_down)
        return add

    def normalize_amplitude(self, x):
        param = -(2 * x).exp().sum(dim=[1, 2]).log() / 2
        x = x + param.unsqueeze(-1).unsqueeze(-1)
        return x

    def forward(self, x):
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        batch_size = x.size(0)
        x = x.reshape([batch_size, self.sites, 2])
        x = torch.index_select(x, 1, self.ordering_bak)

        xf = x.to(dtype=dtype)
        arange = torch.arange(batch_size, device=device)
        total_amplitude = 0
        total_phase = 0
        for i in range(self.sites):
            amplitude = self.amplitude[i](xf[:, :i].reshape([batch_size, 2 * i])).reshape([batch_size, 2, 2])
            phase = self.phase[i](xf[:, :i].reshape([batch_size, 2 * i])).reshape([batch_size, 2, 2])
            amplitude = amplitude + torch.where(self.mask(x[:, :i]), 0, -torch.inf)
            amplitude = self.normalize_amplitude(amplitude)
            amplitude = amplitude[arange, x[:, i, 0], x[:, i, 1]]
            phase = phase[arange, x[:, i, 0], x[:, i, 1]]
            total_amplitude = total_amplitude + amplitude
            total_phase = total_phase + phase
        return (total_amplitude + 1j * total_phase).exp()

    def binomial(self, count, possibility):
        possibility = torch.clamp(possibility, min=0, max=1)
        possibility = torch.where(count == 0, 0, possibility)
        dist = torch.distributions.binomial.Binomial(count, possibility)
        result = dist.sample()
        result = result.to(dtype=torch.int64)
        # Numerical error since result was cast to float.
        return torch.clamp(result, min=torch.zeros_like(count), max=count)

    def generate(self, batch_size, alpha=1):
        # https://arxiv.org/pdf/2109.12606
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        assert alpha == 1

        x = torch.empty([1, 0, 2], device=device, dtype=torch.int64)
        multiplicity = torch.tensor([batch_size], dtype=torch.int64, device=device)
        amplitude_phase = torch.tensor([0], dtype=dtype.to_complex(), device=device)
        for i in range(self.sites):
            local_batch_size = x.size(0)

            xf = x.to(dtype=dtype)
            amplitude = self.amplitude[i](xf.reshape([local_batch_size, 2 * i])).reshape([local_batch_size, 2, 2])
            phase = self.phase[i](xf.reshape([local_batch_size, 2 * i])).reshape([local_batch_size, 2, 2])
            amplitude = amplitude + torch.where(self.mask(x), 0, -torch.inf)
            amplitude = self.normalize_amplitude(amplitude)
            delta_amplitude_phase = (amplitude + 1j * phase).reshape([local_batch_size, 4])
            probability = (2 * amplitude).exp().reshape([local_batch_size, 4])
            probability = probability / probability.sum(dim=-1).unsqueeze(-1)

            sample0123 = multiplicity
            prob23 = probability[:, 2] + probability[:, 3]
            prob01 = probability[:, 0] + probability[:, 1]
            sample23 = self.binomial(sample0123, prob23)
            sample3 = self.binomial(sample23, probability[:, 3] / prob23)
            sample2 = sample23 - sample3
            sample01 = sample0123 - sample23
            sample1 = self.binomial(sample01, probability[:, 1] / prob01)
            sample0 = sample01 - sample1

            x0 = torch.cat([x, torch.tensor([[0, 0]], device=device).expand(local_batch_size, -1, -1)], dim=1)
            x1 = torch.cat([x, torch.tensor([[0, 1]], device=device).expand(local_batch_size, -1, -1)], dim=1)
            x2 = torch.cat([x, torch.tensor([[1, 0]], device=device).expand(local_batch_size, -1, -1)], dim=1)
            x3 = torch.cat([x, torch.tensor([[1, 1]], device=device).expand(local_batch_size, -1, -1)], dim=1)

            new_x = torch.cat([x0, x1, x2, x3])
            new_multiplicity = torch.cat([sample0, sample1, sample2, sample3])
            new_amplitude_phase = (amplitude_phase.unsqueeze(0) + delta_amplitude_phase.permute(1, 0)).reshape([-1])

            selected = new_multiplicity != 0
            x = new_x[selected]
            multiplicity = new_multiplicity[selected]
            amplitude_phase = new_amplitude_phase[selected]

        real_amplitude = amplitude_phase.exp()
        real_probability = (real_amplitude.conj() * real_amplitude).real
        x = torch.index_select(x, 1, self.ordering)
        return x.reshape([x.size(0), self.L1, self.L2, self.orbit_num]), real_amplitude, real_probability, multiplicity


def network(state, spin_up, spin_down, hidden_size, ordering=+1):
    max_orbit_index = max(orbit for [l1, l2, orbit], edge in state.physics_edges)
    max_physical_dim = max(edge.dimension for [l1, l2, orbit], edge in state.physics_edges)
    network = WaveFunction(
        L1=state.L1,
        L2=state.L2,
        orbit_num=max_orbit_index + 1,
        physical_dim=max_physical_dim,
        is_complex=state.Tensor.is_complex,
        spin_up=spin_up,
        spin_down=spin_down,
        hidden_size=hidden_size,
        ordering=ordering,
    ).double()
    return network
