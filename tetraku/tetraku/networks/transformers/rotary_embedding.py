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
from ._common import FeedForward, RealTail, ComplexTail


def apply_rotary_emb(xq, xk, freqs_cis):
    # Ref: https://github.com/meta-llama/llama/blob/main/llama/model.py
    batch_size, sites, heads_num, heads_dim = xq.shape
    # xq, xk: batch, sites, heads_num, heads_dim
    xq = torch.view_as_complex(xq.view([batch_size, sites, heads_num, -1, 2]))
    xk = torch.view_as_complex(xk.view([batch_size, sites, heads_num, -1, 2]))
    # xq, xk: batch, sites, heads_num, heads_dim // 2
    # freqs_cis: sites, heads_dim // 2
    freqs_cis = freqs_cis.view([1, sites, 1, -1])
    # freqs_cis: 1, sites, 1, heads_dim // 2
    xq = torch.view_as_real(xq * freqs_cis).flatten(start_dim=-2)
    xk = torch.view_as_real(xk * freqs_cis).flatten(start_dim=-2)
    # xq, xk: batch, sites, heads_num, heads_dim
    return xq, xk


class SelfAttention(torch.nn.Module):

    def __init__(self, embedding_dim, heads_num):
        super().__init__()

        self.heads_num = heads_num
        self.heads_dim = embedding_dim // heads_num
        assert self.heads_num * self.heads_dim == embedding_dim
        assert self.heads_dim % 2 == 0

        self.norm = torch.nn.LayerNorm(embedding_dim)

        self.qkv = torch.nn.Linear(embedding_dim, 3 * embedding_dim)
        self.out = torch.nn.Linear(embedding_dim, embedding_dim)
        self.freqs = torch.nn.Parameter(torch.randn([self.heads_dim // 2, 3]) / 10000)

    def forward(self, x, position):
        # position: site, 3
        # x: batch, site, embedding
        x = self.norm(x)
        # x: batch, site, embedding
        batch_size, sites, embedding_dim = x.shape
        q, k, v = self.qkv(x).split(embedding_dim, dim=2)
        q = q.view([batch_size, sites, self.heads_num, self.heads_dim])
        k = k.view([batch_size, sites, self.heads_num, self.heads_dim])
        v = v.view([batch_size, sites, self.heads_num, self.heads_dim])
        # q, k, v: batch, site, heads_num, heads_dim
        freqs = self.freqs @ position.transpose(0, 1)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs).transpose(0, 1)
        # freqs_cis: site, heads_dim // 2
        q, k = apply_rotary_emb(q, k, freqs_cis)
        # Apply RoPE
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # q, k, v: batch, heads_num, site, heads_dim
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        # attn: batch, heads_num, site, heads_dim
        out = attn.transpose(1, 2).contiguous().view([batch_size, sites, self.heads_num * self.heads_dim])
        # out: batch, site, embedding_dim
        return self.out(out)


class EncoderUnit(torch.nn.Module):

    def __init__(self, embedding_dim, heads_num, feed_forward_dim):
        super().__init__()
        self.attention = SelfAttention(embedding_dim, heads_num)
        self.feed_forward = FeedForward(embedding_dim, feed_forward_dim)

    def forward(self, x, position):
        # x: batch, site, embedding
        x = x + self.attention(x, position)
        # x: batch, site, embedding
        x = x + self.feed_forward(x)
        # x: batch, site, embedding
        return x


class Transformers(torch.nn.Module):

    def __init__(self, embedding_dim, heads_num, feed_forward_dim, depth):
        super().__init__()
        self.layers = torch.nn.ModuleList(EncoderUnit(embedding_dim, heads_num, feed_forward_dim) for _ in range(depth))

    def forward(self, x, position):
        # x: batch, site, embedding
        for layer in self.layers:
            x = layer(x, position)
        # x: batch, site, embedding
        return x


class Embedding(torch.nn.Module):

    def __init__(self, physical_dim, embedding_dim):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.randn([physical_dim, embedding_dim]))

    def forward(self, x):
        result = self.parameter[x.flatten(start_dim=-3)]
        # param: config, embedding
        # x: batch, L1, L2, orbit
        # result: batch, site, embedding

        _, L1, L2, orbit_num = x.shape
        position = torch.stack([
            torch.arange(L1, device=x.device).view([L1, 1, 1]).expand([L1, L2, orbit_num]),
            torch.arange(L2, device=x.device).view([1, L2, 1]).expand([L1, L2, orbit_num]),
            torch.arange(orbit_num, device=x.device).view([1, 1, orbit_num]).expand([L1, L2, orbit_num]),
        ]).flatten(start_dim=-3).transpose(0, 1).to(dtype=result.dtype)

        return result, position


class WaveFunction(torch.nn.Module):

    def __init__(
        self,
        *,
        physical_dim,
        embedding_dim,
        heads_num,
        feed_forward_dim,
        tail_dim,
        depth,
        is_complex,
    ):
        super().__init__()
        self.embedding = Embedding(physical_dim, embedding_dim)
        self.transformers = Transformers(embedding_dim, heads_num, feed_forward_dim, depth)
        if is_complex:
            self.tail = ComplexTail(embedding_dim, tail_dim)
        else:
            self.tail = RealTail(embedding_dim, tail_dim)

    def forward(self, x):
        # x: batch, L1, L2, orbit
        x, position = self.embedding(x)
        # x: batch, site, embedding
        x = self.transformers(x, position)
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
    depth=6,
):
    max_orbit_index = max(orbit for [l1, l2, orbit], edge in state.physics_edges)
    max_physical_dim = max(edge.dimension for [l1, l2, orbit], edge in state.physics_edges)
    network = WaveFunction(
        physical_dim=max_physical_dim,
        embedding_dim=embedding_dim,
        heads_num=heads_num,
        feed_forward_dim=feed_forward_dim,
        tail_dim=tail_dim,
        depth=depth,
        is_complex=state.Tensor.is_complex,
    ).double()
    return network
