#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2023 Hao Zhang<zh970205@mail.ustc.edu.cn>
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

import ctypes
import numpy as np


class Context():

    def __init__(self, scalapack, layout, nprow, npcol):
        self.scalapack = scalapack
        self.layout = ctypes.c_char(layout)
        self.nprow = ctypes.c_int(nprow)
        self.npcol = ctypes.c_int(npcol)

    def barrier(self, scope=b'A'):
        self.scalapack.blacs_barrier(self.ictxt, ctypes.c_char(scope))

    def _call_blacs_pinfo(self):
        self.rank = ctypes.c_int()
        self.size = ctypes.c_int()
        self.scalapack.blacs_pinfo(self.rank, self.size)

    def _call_blacs_get(self):
        self.ictxt = ctypes.c_int()
        self.scalapack.blacs_get(Scalapack.neg_one, Scalapack.zero, self.ictxt)

    def _call_blacs_gridinit(self):
        if self.nprow.value == -1:
            self.nprow = ctypes.c_int(self.size.value // self.npcol.value)
        if self.npcol.value == -1:
            self.npcol = ctypes.c_int(self.size.value // self.nprow.value)
        self.scalapack.blacs_gridinit(self.ictxt, self.layout, self.nprow, self.npcol)

    def _call_blacs_gridinfo(self):
        self.myrow = ctypes.c_int()
        self.mycol = ctypes.c_int()
        self.scalapack.blacs_gridinfo(self.ictxt, self.nprow, self.npcol, self.myrow, self.mycol)

    def __enter__(self):
        self._call_blacs_pinfo()
        self._call_blacs_get()
        self._call_blacs_gridinit()
        self._call_blacs_gridinfo()
        self.valid = self.rank.value < self.nprow.value * self.npcol.value
        return self

    def _call_blacs_gridexit(self):
        self.scalapack.blacs_gridexit(self.ictxt)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.valid:
            self._call_blacs_gridexit()
        if exc_type is not None:
            return False

    def __bool__(self):
        return self.valid

    def array(self, m, n, mb, nb, *, data=None, dtype=None):
        return Array(self, m, n, mb, nb, data=data, dtype=dtype)


class ArrayDesc(ctypes.Structure):
    _fields_ = [
        ("ctypes", ctypes.c_int),
        ("ctxt", ctypes.c_int),
        ("m", ctypes.c_int),
        ("n", ctypes.c_int),
        ("mb", ctypes.c_int),
        ("nb", ctypes.c_int),
        ("rsrc", ctypes.c_int),
        ("csrc", ctypes.c_int),
        ("lld", ctypes.c_int),
    ]

    @property
    def c_ctypes(self):
        return ctypes.c_int.from_buffer(self, self.__class__.ctypes.offset)

    @property
    def c_ctxt(self):
        return ctypes.c_int.from_buffer(self, self.__class__.ctxt.offset)

    @property
    def c_m(self):
        return ctypes.c_int.from_buffer(self, self.__class__.m.offset)

    @property
    def c_n(self):
        return ctypes.c_int.from_buffer(self, self.__class__.n.offset)

    @property
    def c_mb(self):
        return ctypes.c_int.from_buffer(self, self.__class__.mb.offset)

    @property
    def c_nb(self):
        return ctypes.c_int.from_buffer(self, self.__class__.nb.offset)

    @property
    def c_rsrc(self):
        return ctypes.c_int.from_buffer(self, self.__class__.rsrc.offset)

    @property
    def c_csrc(self):
        return ctypes.c_int.from_buffer(self, self.__class__.csrc.offset)

    @property
    def c_lld(self):
        return ctypes.c_int.from_buffer(self, self.__class__.lld.offset)


class Array(ArrayDesc):

    def __init__(self, context, m, n, mb, nb, *, data=None, dtype=None):
        super().__init__(1, context.ictxt, m, n, mb, nb, 0, 0, 0)
        self.context = context
        self.local_m = self.context.scalapack.numroc(self.c_m, self.c_mb, context.myrow, Scalapack.zero, context.nprow)
        self.local_n = self.context.scalapack.numroc(self.c_n, self.c_nb, context.mycol, Scalapack.zero, context.npcol)
        self.lld = self.local_m  # fortran style

        if data is not None:
            self.data = data
            if not self.data.flags.f_contiguous:
                raise RuntimeError("Scalapack array must be Fortran contiguous")
        elif context:
            if dtype is None:
                raise RuntimeError("dtype need to be specified to create array without data")
            self.data = np.zeros([self.local_m, self.local_n], dtype=dtype, order="F")
        else:
            self.data = np.zeros([1, 1], dtype=dtype, order="F")
            # Just a placeholder

    def scalapack_params(self):
        return (
            Scalapack.numpy_ptr(self.data),
            Scalapack.one,
            Scalapack.one,
            self,
        )

    def lapack_params(self):
        return (
            Scalapack.numpy_ptr(self.data),
            self.c_lld,
        )


class Scalapack():

    def __init__(self, lib):
        self.lib = ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
        self.function_database = {}

        for name in ["p?gemm", "p?gemv", "?gemv", "p?gemr2d"]:
            setattr(self, name.replace("?", ""),
                    {btype: getattr(self, name.replace("?", btype.lower())) for btype in "SDCZ"})
        self.pheevd = {"S": self.pssyevd, "D": self.pdsyevd, "C": self.pcheevd, "Z": self.pzheevd}

    def __call__(self, layout, nprow, npcol):
        return Context(self, layout, nprow, npcol)

    def __getattr__(self, name):
        if name not in self.function_database:
            self.function_database[name] = self._fortran_function(getattr(self.lib, name + "_"))
        return self.function_database[name]

    class Val():

        def __init__(self, value):
            self.value = value

    @classmethod
    def _resolve_arg(cls, arg):
        if isinstance(arg, cls.Val):
            return arg.value
        elif isinstance(arg, int):
            arg = ctypes.c_int(arg)
            return ctypes.byref(arg)
        elif isinstance(arg, bytes):
            arg = ctypes.c_char_p(arg)
            return arg
        else:
            return ctypes.byref(arg)

    @classmethod
    def _fortran_function(cls, function):

        def result(*args):
            return function(*(cls._resolve_arg(arg) for arg in args))

        result.__doc__ = function.__doc__
        return result

    @classmethod
    def numpy_ptr(cls, array):
        return cls.Val(array.ctypes.data_as(ctypes.c_void_p))

    zero = ctypes.c_int(0)
    one = ctypes.c_int(1)
    neg_one = ctypes.c_int(-1)

    s_zero = ctypes.c_float(0)
    s_one = ctypes.c_float(1)
    d_zero = ctypes.c_double(0)
    d_one = ctypes.c_double(1)
    c_zero = (ctypes.c_float * 2)(0, 0)
    c_one = (ctypes.c_float * 2)(1, 0)
    z_zero = (ctypes.c_double * 2)(0, 0)
    z_one = (ctypes.c_double * 2)(1, 0)
    f_zero = {"I": zero, "S": s_zero, "D": d_zero, "C": c_zero, "Z": z_zero}
    f_one = {"I": one, "S": s_one, "D": d_one, "C": c_one, "Z": z_one}

    import ctypes
