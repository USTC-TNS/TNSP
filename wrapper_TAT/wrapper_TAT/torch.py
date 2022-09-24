#!/usr/bin/env python
# -*- coding: utf-8 -*-
# copyright (c) 2022 hao zhang<zh970205@mail.ustc.edu.cn>
#
# this program is free software: you can redistribute it and/or modify
# it under the terms of the gnu general public license as published by
# the free software foundation, either version 3 of the license, or
# any later version.
#
# this program is distributed in the hope that it will be useful,
# but without any warranty; without even the implied warranty of
# merchantability or fitness for a particular purpose.  see the
# gnu general public license for more details.
#
# you should have received a copy of the gnu general public license
# along with this program.  if not, see <https://www.gnu.org/licenses/>.
#

from __future__ import annotations
import operator
from multimethod import multimethod
import torch
import TAT

torch.set_num_threads(1)


class IndexMap:
    __slots__ = ["index", "map"]

    def __init__(self, index=ord('a')):
        self.index = index
        self.map = {}

    def __getitem__(self, key):
        if key not in self.map:
            self.map[key] = self.index
            self.index += 1
        return self.map[key]


class TensorMeta(type):

    @staticmethod
    def _op(op):

        def op_func(self, value):
            if isinstance(value, self.__class__):
                data = op(self.data, value.transpose(self.names).data)
            else:
                data = op(self.data, value)
            return self.__class__(self.names.copy(), data)

        return op_func

    @staticmethod
    def _rop(op):

        def rop_func(self, value):
            data = op(value, self.data)
            return self.__class__(self.names.copy(), data)

        return rop_func

    @staticmethod
    def _iop(op):

        def iop_func(self, value):
            if isinstance(value, self.__class__):
                self.data = op(self.data, value.transpose(self.names).data)
            else:
                self.data = op(self.data, value)
            return self

        return iop_func

    def __new__(cls, name, bases, attrs):
        for op_name in ["add", "sub", "mul", "truediv"]:
            attrs[f"__{op_name}__"] = cls._op(getattr(operator, op_name))
            attrs[f"__r{op_name}__"] = cls._rop(getattr(operator, op_name))
            attrs[f"__i{op_name}__"] = cls._iop(getattr(operator, f"i{op_name}"))
        return super().__new__(cls, name, bases, attrs)


class TensorBlock:
    __slots__ = ["owner"]

    def __init__(self, owner):
        self.owner = owner

    def __getitem__(self, location):
        einsum_map = IndexMap()
        eq = []
        for name in self.owner.names:
            eq.append(chr(einsum_map[name]))
        eq.append("->")
        for single_edge_location in location:
            if isinstance(single_edge_location, tuple):
                name, _ = single_edge_location
            else:
                name = single_edge_location
            eq.append(chr(einsum_map[name]))
        # Return torch tensor directly.
        array = self.owner.data
        return torch.einsum("".join(eq), array)

    def __setitem__(self, key, value):
        raise NotImplementedError()


class Tensor(metaclass=TensorMeta):

    __slots__ = ["names", "data"]

    model = TAT.No

    @property
    def rank(self):
        return len(self.names)

    @multimethod
    def edges(self, index: int):
        return TAT.No.Edge(self.data.shape[index])

    @multimethod
    def edges(self, name: str):
        return self.edges(self.names.index(name))

    @property
    def blocks(self):
        return TensorBlock(self)

    @property
    def storage(self):
        self.data = self.data.contiguous()
        return self.data.view([-1])

    def __float__(self):
        return float(self.data.item())

    def __complex__(self):
        return complex(self.data.item())

    def __repr__(self):
        return ("{names:[" + ",".join(self.names) + "],edges:[" + ",".join(str(i) for i in self.data.size()) + "]}")

    def __str__(self):
        return ("{names:[" + ",".join(self.names) + "],edges:[" + ",".join(str(i) for i in self.data.size()) +
                "],blocks:[" + ",".join(str(i.item()) for i in self.data.view([-1])) + "]}")

    @multimethod
    def __init__(self):
        self.names = []
        self.data = None

    @multimethod
    def __init__(self, names: list[str], edges: list):
        self.names = names.copy()
        self.data = torch.zeros([edge.dimension if isinstance(edge, TAT.No.Edge) else edge for edge in edges],
                                dtype=self.dtype)

    @multimethod
    def __init__(self, names: list[str], data: torch.Tensor):
        self.names = names
        self.data = data

    @multimethod
    def __init__(self, tensor):
        self.names = tensor.names
        self.data = torch.tensor(tensor.blocks[self.names], dtype=self.dtype)

    @multimethod
    def __init__(self, string: str):
        tensor = self.TAT_Tensor(string)
        self.__init__(tensor)

    def copy(self):
        return self.__class__(self.names.copy(), self.data.clone())

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self):
        return self.copy()

    def same_shape(self):
        return self.__class__(self.names.copy(), torch.zeros_like(self.data))

    def map(self, *args, **kwargs):
        raise NotImplementedError()

    def transform(self, *args, **kwargs):
        raise NotImplementedError()

    def sqrt(self, *args, **kwargs):
        raise NotImplementedError()

    def set(self, *args, **kwargs):
        raise NotImplementedError()

    def zero(self):
        torch.zeros(*self.data.shape, out=self.data, dtype=self.dtype)
        return self

    def range(self, first=0, step=1):
        flatten = self.data.view([-1])
        torch.arange(first, first + step * len(flatten), step, out=flatten, dtype=self.dtype)

    def __getitem__(self, location):
        indices = tuple(location[name] for name in self.names)
        return self.data[indices].item()

    def __setitem__(self, location, value):
        indices = tuple(location[name] for name in self.names)
        self.data[indices] = value

    def norm_max(self):
        return torch.norm(self.data.view([-1]), p=torch.inf).cpu()

    def norm_num(self):
        return len(self.data.view([-1]))

    def norm_sum(self):
        return sum(abs(self.data.view([-1])))

    def norm_2(self):
        return torch.norm(self.data.view([-1]), p=2).cpu()

    def edge_rename(self, name_dictionary: dict[str, str]):
        names = [name_dictionary[name] if name in name_dictionary else name for name in self.names]
        return self.__class__(names, self.data)

    def transpose(self, new_names):
        if new_names == self.names:
            return self.__class__(self.names.copy(), self.data)
        einsum_map = IndexMap()
        eq = []
        for name in self.names:
            eq.append(chr(einsum_map[name]))
        eq.append("->")
        for name in new_names:
            eq.append(chr(einsum_map[name]))
        data = torch.einsum("".join(eq), self.data)
        return self.__class__(new_names.copy(), data)

    def contract(self, another_tensor, contract_names: set[tuple[str, str]], fuse_names: set[str] = None):
        if fuse_names is None:
            fuse_names = set()
        einsum_map = IndexMap()
        # (0, ?) -> self names
        # (1, ?) -> another names
        # (2, (?, ?)) -> contract names
        # (3, ?) -> fuse names
        result_names = []
        eq = []
        for name in self.names:
            for contract_name in contract_names:
                if contract_name[0] == name:
                    # contract
                    eq.append(chr(einsum_map[2, contract_name]))
                    break
            else:
                if name in fuse_names:
                    eq.append(chr(einsum_map[3, name]))
                    result_names.append((3, name))
                else:
                    eq.append(chr(einsum_map[0, name]))
                    result_names.append((0, name))
        eq.append(",")
        for name in another_tensor.names:
            for contract_name in contract_names:
                if contract_name[1] == name:
                    # contract
                    eq.append(chr(einsum_map[2, contract_name]))
                    break
            else:
                if name in fuse_names:
                    eq.append(chr(einsum_map[3, name]))
                else:
                    eq.append(chr(einsum_map[1, name]))
                    result_names.append((1, name))
        eq.append("->")
        for name in result_names:
            eq.append(chr(einsum_map[name]))
        data = torch.einsum("".join(eq), self.data, another_tensor.data)
        return self.__class__([name for _, name in result_names], data)

    def identity(self, *args, **kwargs):
        raise NotImplementedError()

    def exponential(self, *args, **kwargs):
        raise NotImplementedError()

    def conjugate(self, default_is_physics_edge=False, exclude_names_set=None):
        return self.__class__(self.names.copy(), self.data.conj())

    def trace(self, trace_names: set[tuple[str, str]]):
        einsum_map = IndexMap()
        # (0, ?) -> self names
        # (1, (?, ?)) -> trace names
        result_names = []
        eq = []
        for name in self.names:
            for trace_name in trace_names:
                if name in trace_name:
                    eq.append(chr(einsum_map[1, trace_name]))
                    break
            else:
                eq.append(chr(einsum_map[0, name]))
                result_names.append((0, name))
        eq.append("->")
        for name in result_names:
            eq.append(chr(einsum_map[name]))
        data = torch.einsum("".join(eq), self.data)
        return self.__class__([name for _, name in result_names], data)

    def svd(self, *args, **kwargs):
        raise NotImplementedError()

    def qr(self, *args, **kwargs):
        raise NotImplementedError()

    def shrink(self, shrink_configuration):
        result_names = []
        slices = []
        for name in self.names:
            if name in shrink_configuration:
                slices.append(shrink_configuration[name])
            else:
                slices.append(slice(None))
                result_names.append(name)
        data = self.data[slices].clone()
        return self.__class__(result_names, data)

    def expand(self, *args, **kwargs):
        raise NotImplementedError()

    def rand(self, min, max):
        tensor = self.TAT_Tensor(self.names, self.data.shape).rand(min, max)
        self.data = torch.tensor(tensor.blocks[self.names], dtype=self.dtype)
        return self

    def randn(self, mean=0, stddev=1):
        tensor = self.TAT_Tensor(self.names, self.data.shape).randn(mean, stddev)
        self.data = torch.tensor(tensor.blocks[self.names], dtype=self.dtype)
        return self


class S_Tensor(Tensor):
    __slots__ = []

    dtype = torch.float32
    TAT_Tensor = TAT.No.S.Tensor

    is_real = True
    is_complex = False


class D_Tensor(Tensor):
    __slots__ = []

    dtype = torch.float64
    TAT_Tensor = TAT.No.D.Tensor

    is_real = True
    is_complex = False


class C_Tensor(Tensor):
    __slots__ = []

    dtype = torch.complex64
    TAT_Tensor = TAT.No.C.Tensor

    is_real = False
    is_complex = True


class Z_Tensor(Tensor):
    __slots__ = []

    dtype = torch.complex128
    TAT_Tensor = TAT.No.Z.Tensor

    is_real = False
    is_complex = True
