#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2022 Hao Zhang<zh970205@mail.ustc.edu.cn>
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


class FakeEdge:
    __slots__ = ["direction"]

    def __init__(self, direction):
        self.direction = direction

    def __getitem__(self, x):
        return (list(x), self.direction)


Fedge = FakeEdge(False)
Tedge = FakeEdge(True)


def rename_io(t, m):
    if not isinstance(m, dict):
        m = {i: j for i, j in enumerate(m)}
    res = {}
    for i, j in m.items():
        res[f"I{i}"] = f"I{j}"
        res[f"O{i}"] = f"O{j}"
    return t.edge_rename(res)


def kronecker_product(res, *b):
    for i in b:
        res = res.contract(i, set())
    return res
