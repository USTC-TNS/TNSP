#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2021-2022 Hao Zhang<zh970205@mail.ustc.edu.cn>
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


def safe_contract(tensor_1, tensor_2, pair, *, contract_all_physics_edges=False):
    new_pair = set()
    if contract_all_physics_edges:
        for name in tensor_1.names:
            if str(name)[0] == "P" and name in tensor_2.names:
                new_pair.add((name, name))
    for name_1, name_2 in pair:
        if name_1 in tensor_1.names and name_2 in tensor_2.names:
            new_pair.add((name_1, name_2))

    return tensor_1.contract(tensor_2, new_pair)


def safe_rename(tensor, name_map):
    new_name_map = {}
    for key, value in name_map.items():
        if key in tensor.names:
            new_name_map[key] = value
    return tensor.edge_rename(new_name_map)
