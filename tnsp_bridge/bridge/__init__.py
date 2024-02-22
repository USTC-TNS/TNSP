#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2021-2024 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

import numpy as np
import TAT


def _read_block(getline):
    """
    Read single block.
    """
    head = getline()
    if head == " readable_data T":
        return _read_data(getline)
    elif head == " readable_data F":
        return _read_empty(getline)
    else:
        raise RuntimeError(f"data header should be either `readable_data F' or `readable_data T', but got: `{head}'")


def _read_data(getline):
    """
    Read single unempty block, used in many style, like tensor shape or tensor content.
    """
    value_type, total_number, block_number, _ = (int(i) for i in getline().split())
    # The fourth argument is the max length for string, this is not needed in python.
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
    elif value_type == 6:
        # bool
        return_type = bool
        content = [bool(i) for i in getline().split()]
    elif value_type == 7 or value_type == 8:
        # str
        return_type = str
        content = [str(i) for i in getline().split()]
    else:
        raise RuntimeError(f"Unrecognized type number: `{value_type}'")
    end_data = getline()
    if end_data != " End_data":
        raise RuntimeError(f"The data block should end with ` End_data', but got: `{end_data}'")
    if len(block_begin) != block_number or len(block_end) != block_number:
        raise RuntimeError("The number of block begin, end mismatch the block number")
    if len(content) != total_number:
        raise RuntimeError("The number of content mismatches the total number")
    return return_type, [content[block_begin[i] - 1:block_end[i]] for i in range(block_number)]


def _read_empty(getline):
    """
    Read single empty block.
    """
    empty_data_array = getline()
    if empty_data_array != " Empty DataArray":
        raise RuntimeError(f"empty dataarray is expected but get: `{empty_data_array}'")
    return None


def bridge(getline, *, parity=False, compat=False):
    """
    Convert tnsp tensor to TAT tensor.

    Parameters
    ----------
    getline : Callable[[], str]
        A function return next line read from tnsp output each time called.
    parity : bool, default=False
        Whether to use Z2 instead of U1 in symmetry tensor.
    compat : bool, default=False
        Whether to read data in the old data.

    Returns
    -------
    Tensor
        The result tensor
    """
    flag = getline().split()
    # first line contains: dimension, symmetry, fermi, named, rank
    # rank is duplicated here so do not use it
    dimension, symmetry, fermi, named = [i == "T" for i in flag[:4]]
    if dimension is False:
        raise RuntimeError("dimension flag is False")
    if symmetry:
        return _bridge_symmetry(getline, named=named, fermi=fermi, parity=parity, compat=compat)
    else:
        if fermi is True:
            raise RuntimeError("fermi flag is True but symmetry flag i False")
        return _bridge_non_symmetry(getline, named=named, compat=compat)


def _bridge_non_symmetry(getline, *, named, compat):
    if _read_block(getline) is not None:
        raise RuntimeError("The first block of data for no symmetry should be empty")
    if _read_block(getline) is not None:
        raise RuntimeError("The second block of data for no symmetry should be empty")
    if compat:
        _, [dimension] = _read_block(getline)
    else:
        dimension = [int(i) for i in getline().split()]
    if named:
        names = getline().split()
    else:
        names = [f"UnnamedEdge{i}" for i in range(len(dimension))]
    names.reverse()
    dimension.reverse()

    content_type, content = _read_block(getline)

    if content_type is bool and named is False:
        return np.array(content).reshape(dimension).tolist()
    if content_type is str and named is False:
        return np.array(content).reshape(dimension).tolist()
    if content_type is int and named is False:
        return np.array(content).reshape(dimension).tolist()

    if content_type is float:
        tensor_type = TAT.No.float.Tensor
    elif content_type is complex:
        tensor_type = TAT.No.complex.Tensor
    else:
        raise RuntimeError(f"Unrecognized content type: `{content_type}'")
    tensor = tensor_type(names, dimension)
    tensor.blocks[names] = np.array(content).reshape(dimension)

    return tensor


def _bridge_symmetry(getline, *, named, fermi, parity, compat):
    _, symmetry = _read_block(getline)
    if parity:
        symmetry = [[False if int(s) == +1 else True for s in sym] for sym in symmetry]
    else:
        symmetry = [[int(s) for s in sym] for sym in symmetry]
    _, dimension = _read_block(getline)
    if compat:
        if fermi:
            _, [block_number, arrow] = _read_block(getline)
            arrow = [False if int(i) == 1 else True for i in arrow]
        else:
            _, [block_number] = _read_block(getline)
            arrow = [False for _ in symmetry]
    else:
        block_number = [int(i) for i in getline().split()]
        if fermi:
            arrow = [False if int(i) == 1 else True for i in getline().split()]
        else:
            arrow = [False for _ in symmetry]
    if not len(symmetry) == len(dimension) == len(block_number) == len(arrow):
        raise RuntimeError("The number of symmetry, dimension, block number or arrow mismatch")
    rank = len(symmetry)
    for i in range(rank):
        if not len(symmetry[i]) == len(dimension[i]) == block_number[i]:
            raise RuntimeError("The segments for symmetry, dimension mismatch the block number")
    edges = [([(s, d) for s, d in zip(symmetry[i], dimension[i])], arrow[i]) for i in range(rank)]
    if named:
        names = getline().split()
    else:
        names = [f"UnnamedEdge{i}" for i in range(len(dimension))]
    symmetry.reverse()
    dimension.reverse()
    block_number.reverse()
    arrow.reverse()
    edges.reverse()
    names.reverse()

    content_type, content = _read_block(getline)
    if fermi:
        if parity:
            model = TAT.FermiZ2
        else:
            model = TAT.FermiU1
    else:
        if parity:
            model = TAT.BoseZ2
        else:
            model = TAT.BoseU1
    if content_type is float:
        tensor_type = model.float.Tensor
    elif content_type is complex:
        tensor_type = model.complex.Tensor
    else:
        raise RuntimeError(f"Unrecognized content type: `{content_type}'")

    tensor = tensor_type(names, edges)

    index = [0 for i in range(rank)]
    current = 0
    while True:
        block_src = content[current]
        if len(block_src) != 0:
            block_position = [(names[i], symmetry[i][index[i]]) for i in range(rank)]
            block_dimension = tuple(dimension[i][index[i]] for i in range(rank))
            if tensor.blocks[block_position].shape != block_dimension:
                raise RuntimeError("Block dimension mismatch")
            tensor.blocks[block_position] = np.array(block_src).reshape(block_dimension)

        current += 1
        active = rank - 1
        index[active] += 1
        while index[active] == block_number[active]:
            if active == 0:
                if current != len(content):
                    raise RuntimeError("Total number of content mismatch")
                return tensor
            index[active] = 0
            active -= 1
            index[active] += 1
