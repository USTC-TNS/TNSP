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


class WaveTail(torch.nn.Module):

    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.LayerNorm(embedding_dim),
            torch.nn.Linear(embedding_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Flatten(start_dim=-2),
        )

    def forward(self, x):
        # x: batch, site
        x = self.model(x)
        # x: batch
        x = x.mean(dim=-1)
        return x


class RealTail(torch.nn.Module):

    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.sign = WaveTail(embedding_dim, hidden_dim)
        self.absolute = WaveTail(embedding_dim, hidden_dim)

    def forward(self, x):
        sign = self.sign(x).cos()
        absolute = self.absolute(x).exp()
        return sign * absolute


class ComplexTail(torch.nn.Module):

    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.sign = WaveTail(embedding_dim, hidden_dim)
        self.absolute = WaveTail(embedding_dim, hidden_dim)

    def forward(self, x):
        sign = torch.polar(torch.ones([], device=x.device, dtype=x.dtype), self.sign(x))
        absolute = self.absolute(x).exp()
        return sign * absolute
