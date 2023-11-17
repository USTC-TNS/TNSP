#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2020-2023 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

# this will never be pickled
element_pool = {}


def tensor_element(tensor):
    tensor_id = id(tensor)
    if tensor_id not in element_pool:
        element_pool[tensor_id] = calculate_element(tensor)
    return element_pool[tensor_id]


def loop_nonzero_tensor(tensor, names, rank):
    if tensor.data.nelement() == 0:
        return
    if rank == 0:
        # rank == 0
        yield [], tensor
        return
    indices = [0 for _ in range(rank)]
    while True:
        if tensor.data[tuple(indices)] != 0:
            element = tensor.__class__(
                names=tensor.names,
                edges=tuple(tensor.edges[i].__class__(
                    fermion=tensor.fermion,
                    dtypes=tensor.dtypes,
                    symmetry=tuple(symmetryp[indices[i]].reshape([1]) for symmetry in tensor.edges[i].symmetry),
                    dimension=1,
                    arrow=tensor.edges[i].arrow,
                    parity=tensor.edges[i].parity[indices[i]].reshape([1]),
                ) for i in range(rank)),
                fermion=tensor.fermion,
                dtypes=tensor.dtypes,
                data=tensor.data[tuple(indices)].reshape([1 for _ in range(rank)]),
                mask=tensor.mask[tuple(indices)].reshape([1 for _ in range(rank)]),
            )
            yield [tensor.edges[i].point_by_index(indices[i]) for i in range(rank)], element

        edge_position = rank - 1
        indices[edge_position] += 1
        while indices[edge_position] == tensor.edges[edge_position].dimension:
            if edge_position == 0:
                return
            indices[edge_position] = 0
            edge_position -= 1
            indices[edge_position] += 1


def conjugate_symmetry(edge_point):
    return (-edge_point[0], edge_point[1])


def calculate_element(tensor):
    # tensor: Tensor
    # result: dict[[EdgePoint], dict[[EdgePoint], Tensor]]
    # where first [EdgePoint] is edge f"I{i}" and the second is f"O{i}"
    # f"O{i}" is original and f"I{i}" is conjugated
    # so all edge have arrow=False
    result = {}
    names = tensor.names
    names_to_index = {str(n): i for i, n in enumerate(names)}
    rank = tensor.rank
    body = rank // 2
    for edge_point, value in loop_nonzero_tensor(tensor, names, rank):
        # index: [EdgePoint]
        # value: Tensor
        edge_in = tuple(conjugate_symmetry(edge_point[names_to_index[f"I{i}"]]) for i in range(body))
        edge_out = tuple(edge_point[names_to_index[f"O{i}"]] for i in range(body))
        if edge_in not in result:
            result[edge_in] = {}
        result[edge_in][edge_out] = value
    return result
