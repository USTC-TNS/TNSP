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
from ._sinusoidal_3d import PositionalEncoding3D


class FeedForward(torch.nn.Module):

    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.LayerNorm(embedding_dim),
            torch.nn.Linear(embedding_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x):
        # x: batch, site, embedding
        x = self.model(x)
        # x: batch, site, embedding
        return x


class SelfAttention(torch.nn.Module):

    def __init__(self, embedding_dim, heads_num):
        super().__init__()
        self.norm = torch.nn.LayerNorm(embedding_dim)
        self.attention = torch.nn.MultiheadAttention(embedding_dim, heads_num, batch_first=True)

    def forward(self, x):
        # x: batch, site, embedding
        x = self.norm(x)
        # x: batch, site, embedding
        x, _ = self.attention(x, x, x, need_weights=False)
        # x: batch, site, embedding
        return x


class EncoderUnit(torch.nn.Module):

    def __init__(self, embedding_dim, heads_num, feed_forward_dim):
        super().__init__()
        self.attention = SelfAttention(embedding_dim, heads_num)
        self.feed_forward = FeedForward(embedding_dim, feed_forward_dim)

    def forward(self, x):
        # x: batch, site, embedding
        x = x + self.attention(x)
        # x: batch, site, embedding
        x = x + self.feed_forward(x)
        # x: batch, site, embedding
        return x


class Transformers(torch.nn.Module):

    def __init__(self, embedding_dim, heads_num, feed_forward_dim, depth):
        super().__init__()
        self.layers = torch.nn.ModuleList(EncoderUnit(embedding_dim, heads_num, feed_forward_dim) for _ in range(depth))

    def forward(self, x):
        # x: batch, site, embedding
        for layer in self.layers:
            x = layer(x)
        # x: batch, site, embedding
        return x


class Embedding(torch.nn.Module):

    def __init__(self, L1, L2, orbit_num, physical_dim, embedding_dim):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.randn([physical_dim, embedding_dim]))
        self.position = PositionalEncoding3D(embedding_dim)

    def forward(self, x):
        batch_size, L1, L2, orbit = x.shape

        result = self.parameter[x.flatten(start_dim=-3)]
        # param: config, embedding
        # x: batch, L1, L2, orbit
        # result: batch, site, embedding

        result = result.view([batch_size, L1, L2, orbit, -1])
        # result: batch, L1, L2, orbit, embedding

        result = result + self.position(result)

        return result.flatten(start_dim=-4, end_dim=-2)


class WaveTail(torch.nn.Module):

    def __init__(self, sites, embedding_dim, hidden_dim):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.LayerNorm(embedding_dim),
            torch.nn.Linear(embedding_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Flatten(start_dim=-2),
            torch.nn.AvgPool1d(kernel_size=sites),
            torch.nn.Flatten(start_dim=-2),
        )

    def forward(self, x):
        # x: batch, site, embedding
        x = self.model(x)
        # x: batch
        return x


class RealTail(torch.nn.Module):

    def __init__(self, sites, embedding_dim, hidden_dim):
        super().__init__()
        self.sign = WaveTail(sites, embedding_dim, hidden_dim)
        self.absolute = WaveTail(sites, embedding_dim, hidden_dim)

    def forward(self, x):
        sign = torch.cos(self.sign(x))
        absolute = self.absolute(x).exp()
        return sign * absolute


class ComplexTail(torch.nn.Module):

    def __init__(self, sites, embedding_dim, hidden_dim):
        super().__init__()
        self.sign = WaveTail(sites, embedding_dim, hidden_dim)
        self.absolute = WaveTail(sites, embedding_dim, hidden_dim)

    def forward(self, x):
        sign = torch.polar(torch.ones([], device=x.device), self.sign(x))
        absolute = self.absolute(x).exp()
        return sign * absolute


class WaveFunction(torch.nn.Module):

    def __init__(
        self,
        *,
        L1,
        L2,
        orbit_num,
        physical_dim,
        embedding_dim,
        heads_num,
        feed_forward_dim,
        tail_dim,
        depth,
        is_complex,
    ):
        super().__init__()
        self.embedding = Embedding(L1, L2, orbit_num, physical_dim, embedding_dim)
        self.transformers = Transformers(embedding_dim, heads_num, feed_forward_dim, depth)
        if is_complex:
            self.tail = ComplexTail(L1 * L2 * orbit_num, embedding_dim, tail_dim)
        else:
            self.tail = RealTail(L1 * L2 * orbit_num, embedding_dim, tail_dim)

    def forward(self, x):
        # x: batch, L1, L2, orbit
        x = self.embedding(x)
        # x: batch, site, embedding
        x = self.transformers(x)
        # x: batch, site, embedding
        return self.tail(x)

    def normalize(self, param):
        self.tail.absolute.model[3].bias.data -= param


def network(
    state,
    embedding_dim=512,
    heads_num=8,
    feed_forward_dim=2048,
    tail_dim=2048,
    depth=1,
):
    max_orbit_index = max(orbit for [l1, l2, orbit], edge in state.physics_edges)
    max_physical_dim = max(edge.dimension for [l1, l2, orbit], edge in state.physics_edges)
    network = WaveFunction(
        L1=state.L1,
        L2=state.L2,
        orbit_num=max_orbit_index + 1,
        physical_dim=max_physical_dim,
        embedding_dim=embedding_dim,
        heads_num=heads_num,
        feed_forward_dim=feed_forward_dim,
        tail_dim=tail_dim,
        depth=depth,
        is_complex=state.Tensor.is_complex,
    ).double()
    return network
