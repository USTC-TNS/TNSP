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


class FeedForward(torch.nn.Module):

    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.LayerNorm(embedding_dim),
            torch.nn.Linear(embedding_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, embedding_dim),
        )

    def generator(self, max_sites):
        result = None
        while True:
            xi = yield result
            result = self(xi)

    def forward(self, x):
        # x: batch, site, embedding
        x = self.model(x)
        # x: batch, site, embedding
        return x


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

    def generator(self, max_sites):
        scale_factor = self.heads_dim**(-1 / 2)
        result = None
        sites = 0
        while True:
            xi = yield result
            # xi: batch, embedding
            xi = self.norm(xi)
            # xi: batch, embedding
            batch_size, embedding_dim = xi.shape
            qi, ki, vi = self.qkv(xi).split(embedding_dim, dim=1)
            qi = qi.view([batch_size, self.heads_num, self.heads_dim])
            ki = ki.view([batch_size, self.heads_num, self.heads_dim])
            vi = vi.view([batch_size, self.heads_num, self.heads_dim])
            # qi, ki, vi: batch, heads_num, heads_dim
            if sites == 0:
                # kv cache
                ks = torch.zeros(
                    [max_sites, batch_size, self.heads_num, self.heads_dim],
                    dtype=xi.dtype,
                    device=xi.device,
                )
                vs = torch.zeros(
                    [max_sites, batch_size, self.heads_num, self.heads_dim],
                    dtype=xi.dtype,
                    device=xi.device,
                )
            ks[sites] = ki
            vs[sites] = vi
            sites += 1
            attn_weight = torch.einsum("bhe,sbhe->sbh", qi * scale_factor, ks[:sites])
            attn_weight = attn_weight.softmax(dim=0)
            attn = torch.einsum("sbh,sbhe->bhe", attn_weight, vs[:sites])
            # attn: batch, heads_num, site, heads_dim
            out = attn.contiguous().view([batch_size, -1])
            # out: batch, site, embedding_dim
            result = self.out(out)

    def forward(self, x):
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
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # q, k, v: batch, heads_num, site, heads_dim
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        # attn: batch, heads_num, site, heads_dim
        out = attn.transpose(1, 2).contiguous().view([batch_size, sites, self.heads_num * self.heads_dim])
        # out: batch, site, embedding_dim
        return self.out(out)


class DecoderUnit(torch.nn.Module):

    def __init__(self, embedding_dim, heads_num, feed_forward_dim):
        super().__init__()
        self.attention = SelfAttention(embedding_dim, heads_num)
        self.feed_forward = FeedForward(embedding_dim, feed_forward_dim)

    def generator(self, max_sites):
        result = None
        attention = self.attention.generator(max_sites)
        attention.send(None)
        feed_forward = self.feed_forward.generator(max_sites)
        feed_forward.send(None)
        while True:
            xi = yield result
            xi = xi + attention.send(xi)
            xi = xi + feed_forward.send(xi)
            result = xi

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
        self.layers = torch.nn.ModuleList(DecoderUnit(embedding_dim, heads_num, feed_forward_dim) for _ in range(depth))

    def generator(self, max_sites):
        result = None
        layers = [layer.generator(max_sites) for layer in self.layers]
        [layer.send(None) for layer in layers]
        while True:
            xi = yield result
            for layer in layers:
                xi = layer.send(xi)
            result = xi

    def forward(self, x):
        # x: batch, site, embedding
        for layer in self.layers:
            x = layer(x)
        # x: batch, site, embedding
        return x


class WaveTail(torch.nn.Module):

    def __init__(self, embedding_dim, hidden_dim, physical_dim):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.LayerNorm(embedding_dim),
            torch.nn.Linear(embedding_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, physical_dim),
        )

    def forward(self, x):
        # x: batch, site, config
        x = self.model(x)
        # x: batch, site, config
        return x


class RealTail(torch.nn.Module):

    def __init__(self, embedding_dim, hidden_dim, physical_dim):
        super().__init__()
        self.sign = WaveTail(embedding_dim, hidden_dim, physical_dim)
        self.absolute = WaveTail(embedding_dim, hidden_dim, physical_dim)

    def generator(self, max_sites):
        result = None
        while True:
            xi = yield result
            result = self(xi)

    def forward(self, x):
        sign = self.sign(x).cos()
        absolute = self.absolute(x).exp()
        return sign * absolute


class ComplexTail(torch.nn.Module):

    def __init__(self, embedding_dim, hidden_dim, physical_dim):
        super().__init__()
        self.sign = WaveTail(embedding_dim, hidden_dim, physical_dim)
        self.absolute = WaveTail(embedding_dim, hidden_dim, physical_dim)

    def forward(self, x):
        sign = torch.polar(torch.ones([], device=x.device, dtype=x.dtype), self.sign(x))
        absolute = self.absolute(x).exp()
        return sign * absolute

    def generator(self, max_sites):
        result = None
        while True:
            xi = yield result
            result = self(xi)
