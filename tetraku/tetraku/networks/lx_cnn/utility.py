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


class LxMapping(torch.nn.Module):

    def forward(self, x):
        # input: batch, L1, L2, orbit
        # output: batch, channel, L1, L2
        return (x * 2 - 1).double().squeeze(-1).unsqueeze(-3)


class LastProd(torch.nn.Module):

    def forward(self, x):
        return x.prod(dim=-1)
