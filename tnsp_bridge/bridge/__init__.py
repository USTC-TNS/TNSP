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

from __future__ import annotations
import numpy as np
import TAT


def read_block(getline):
    """
    Read single block.
    """
    head = getline()
    if head == " readable_data T":
        return read_data(getline)
    elif head == " readable_data F":
        return read_empty(getline)
    else:
        raise RuntimeError("format error")


def read_data(getline):
    """
    Read single unempty block, used in many style, like tensor shape or tensor content.
    """
    value_type, total_number, block_number, _ = (int(i) for i in getline().split())
    block_begin = [int(i) for i in getline().split()]
    block_end = [int(i) for i in getline().split()]
    if value_type == 1:
        # int
        return_type = int
        content = [int(i) for i in getline().split()]
    elif value_type == 2 or value_type == 3:
        # float
        return_type = float
        content = [float(i) for i in getline().split()]
    elif value_type == 4 or value_type == 5:
        # complex
        return_type = complex
        real = [float(i) for i in getline().split()]
        imag = [float(i) for i in getline().split()]
        content = [complex(i, j) for i, j in zip(real, imag)]
    if getline() != " End_data":
        raise RuntimeError("format error")
    if len(block_begin) != block_number or len(block_end) != block_number:
        raise RuntimeError("format error")
    if len(content) != total_number:
        raise RuntimeError("format error")
    return return_type, [content[block_begin[i] - 1:block_end[i]] for i in range(block_number)]


def read_empty(getline):
    """
    Read single empty block.
    """
    if getline() != " Empty DataArray":
        raise RuntimeError("format error")
    return None, None


def bridge(getline):
    """
    Convert tnsp tensor to TAT tensor.

    Parameters
    ----------
    getline
        A function return next line read from tnsp output each time called.

    Returns
    -------
    Tensor
        The result tensor
    """
    flag = getline()
    if flag == " T T T T":
        return bridge_fermi(getline)
    elif flag == " T T T F":
        return bridge_fermi(getline, named=False)
    elif flag == " T F F T":
        return bridge_no(getline)
    elif flag == " T F F F":
        return bridge_no(getline, named=False)
    raise RuntimeError("bridge error")


def bridge_no(getline, named=True):
    read_block(getline)
    read_block(getline)
    _, [dimension] = read_block(getline)
    if named:
        names = getline().split()
    else:
        names = [f"UnnamedEdge{i}" for i in range(len(dimension))]

    content_type, content = read_block(getline)
    if content_type == None:
        return
    tensor_type = TAT(content_type)
    tensor = tensor_type(names, dimension).zero()
    names.reverse()
    dimension.reverse()
    tensor = tensor.transpose(names)
    tensor.blocks[names] = np.array(content).reshape(dimension)
    return tensor


def bridge_fermi(getline, named=True):
    _, symmetry = read_block(getline)
    _, dimension = read_block(getline)
    _, [block_number, arrow] = read_block(getline)
    if not len(symmetry) == len(dimension) == len(block_number) == len(arrow):
        raise RuntimeError("bridge error")
    rank = len(symmetry)
    for i in range(rank):
        if not len(symmetry[i]) == len(dimension[i]) == block_number[i]:
            raise RuntimeError("bridge error")
    arrow = [False if i == 1 else True for i in arrow]
    edges = [([(int(s), d) for s, d in zip(symmetry[i], dimension[i])], arrow[i]) for i in range(rank)]
    if named:
        names = getline().split()
    else:
        names = [f"UnnamedEdge{i}" for i in range(len(dimension))]

    content_type, content = read_block(getline)
    if content_type == None:
        return
    tensor_type = TAT(content_type, "Fermi")
    tensor = tensor_type(names, edges).zero()
    names.reverse()
    edges.reverse()
    block_number.reverse()
    arrow.reverse()
    symmetry.reverse()
    dimension.reverse()
    tensor = tensor.transpose(names)

    index = [0 for i in range(rank)]
    current = 0
    while True:
        block_src = content[current]
        if len(block_src) != 0:
            block_position = [(names[i], int(symmetry[i][index[i]])) for i in range(rank)]
            block_dimension = tuple(dimension[i][index[i]] for i in range(rank))
            if tensor.blocks[block_position].shape != block_dimension:
                raise RuntimeError("bridge error")
            tensor.blocks[block_position] = np.array(block_src).reshape(block_dimension)

        current += 1
        active = rank - 1
        index[active] += 1
        while index[active] == block_number[active]:
            if active == 0:
                if current != len(content):
                    raise RuntimeError("bridge error")
                return tensor
            index[active] = 0
            active -= 1
            index[active] += 1
