#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2020-2022 Hao Zhang<zh970205@mail.ustc.edu.cn>
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


def loop_nonzero_block(block, symmetries, rank, names, template):
    indices = [0 for _ in range(rank)]
    for i in range(rank):
        if block.shape[i] == 0:
            # dims == 0
            return
    while True:
        value = block[tuple(indices)]
        if value != 0:
            # yield
            template[{names[i]: 0 for i in range(rank)}] = value
            yield [(symmetries[i], indices[i]) for i in range(rank)], template.copy()

        edge_position = rank - 1

        indices[edge_position] += 1
        while indices[edge_position] == block.shape[edge_position]:
            if edge_position == 0:
                return
            indices[edge_position] = 0
            edge_position -= 1
            indices[edge_position] += 1


def loop_nonzero_tensor(tensor, names, rank):
    # see include/TAT/structure/edge.hpp
    edges = [tensor.edges(i).segment for i in range(rank)]
    arrow = [tensor.edges(i).arrow for i in range(rank)]
    Edge = tensor.model.Edge
    if rank == 0:
        # rank == 0
        yield [], tensor
        return
    symmetry_indices = [0 for _ in range(rank)]
    for i in range(rank):
        if len(edges[i]) == 0:
            # dims == 0
            return
    zero_symmetry = tensor.model.Symmetry()
    while True:
        symmetries = [edges[i][symmetry_indices[i]][0] for i in range(rank)]
        if sum(symmetries, start=zero_symmetry) == zero_symmetry:
            block = tensor.blocks[[(names[i], symmetries[i]) for i in range(rank)]]
            template_edges = [Edge([symmetries[i]], arrow[i]) for i in range(rank)]
            template = type(tensor)(names, template_edges)
            yield from loop_nonzero_block(block, symmetries, rank, names, template)

        edge_position = rank - 1

        symmetry_indices[edge_position] += 1
        while symmetry_indices[edge_position] == len(edges[edge_position]):
            if edge_position == 0:
                return
            symmetry_indices[edge_position] = 0
            edge_position -= 1
            symmetry_indices[edge_position] += 1


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
