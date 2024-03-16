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
from ._common import Transformers, RealTail, ComplexTail


class Embedding(torch.nn.Module):

    def __init__(self, sites, physical_dim, embedding_dim):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.randn([sites, physical_dim, embedding_dim]))

    def generator(self, max_sites):
        result = None
        sites = 0
        while True:
            xi = yield result
            # xi: batch
            parameter = self.parameter[sites]
            # param: config, embedding
            xi = xi.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, parameter.size(-1))
            # xi: batch, config=1, embedding
            parameter = parameter.unsqueeze(0).expand(xi.size(0), -1, -1)
            # param: batch, config, embedding
            result = torch.gather(parameter, -2, xi)
            # result: batch, 1, embedding
            result = result.squeeze(-2)
            sites += 1

    def forward(self, x):
        # x: batch, sites
        x = x.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.parameter.size(-1))
        # x: batch, sites, config=1, embedding

        # param: sites, config, embedding
        parameter = self.parameter.unsqueeze(0).expand(x.size(0), -1, -1, -1)
        # param: batch, sites, config, embedding

        result = torch.gather(parameter, -2, x)
        # result: batch, site, 1, embedding

        return result.squeeze(-2)


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
        self.L1 = L1
        self.L2 = L2
        self.orbit_num = orbit_num
        self.sites = L1 * L2 * orbit_num

        self.embedding = Embedding(self.sites + 1, physical_dim, embedding_dim)
        self.transformers = Transformers(embedding_dim, heads_num, feed_forward_dim, depth)
        if is_complex:
            self.tail = ComplexTail(embedding_dim, tail_dim, physical_dim)
        else:
            self.tail = RealTail(embedding_dim, tail_dim, physical_dim)

    def generate(self, batch_size, alpha=1):
        device = next(self.parameters()).device

        embedding = self.embedding.generator(self.sites + 1)
        embedding.send(None)
        transformers = self.transformers.generator(self.sites + 1)
        transformers.send(None)
        tail = self.tail.generator(self.sites + 1)
        tail.send(None)

        xi = torch.zeros([batch_size], device=device, dtype=torch.int64)
        x = []
        psi = []
        prob = []
        for sites in range(self.sites):
            psii = tail.send(transformers.send(embedding.send(xi)))
            psii = psii / (psii.conj() * psii).sum(dim=-1).unsqueeze(-1).sqrt()
            probi = (psii.conj() * psii).real
            if alpha != 1:
                probi = probi**alpha
                probi = probi / probi.sum(dim=-1).unsqueeze(-1)
            xi = torch.multinomial(probi, 1).squeeze(-1)
            psii = torch.gather(psii, -1, xi.unsqueeze(-1)).squeeze(-1)
            probi = torch.gather(probi, -1, xi.unsqueeze(-1)).squeeze(-1)
            x.append(xi)
            psi.append(psii)
            prob.append(probi)
        x = torch.stack(x, dim=-1).reshape([-1, self.L1, self.L2, self.orbit_num])
        psi = torch.stack(psi, dim=-1).prod(dim=-1)
        prob = torch.stack(prob, dim=-1).prod(dim=-1)
        return x, psi, prob, torch.ones_like(prob, dtype=torch.int64)

    def forward(self, x):
        # x: batch, L1, L2, orbit

        x = x.reshape([x.size(0), -1])
        # x: batch, site

        xp = torch.cat((torch.zeros([x.size(0), 1], dtype=x.dtype, device=x.device), x[:, :-1]), dim=1)
        # xp: batch, site

        psi = self.tail(self.transformers(self.embedding(xp)))
        psi = psi / (psi.conj() * psi).sum(dim=-1).unsqueeze(-1).sqrt()
        # psi: batch, site, config

        psi = torch.gather(psi, -1, x.unsqueeze(-1)).squeeze(-1).prod(dim=-1)
        # psi: batch
        return psi

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
