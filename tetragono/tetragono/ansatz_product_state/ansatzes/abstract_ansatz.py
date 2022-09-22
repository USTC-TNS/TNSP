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

from copyreg import _slotnames
from ..state import AnsatzProductState


class AbstractAnsatzMeta(type):

    @staticmethod
    def getstate(self):
        state = {key: getattr(self, key) for key in _slotnames(self.__class__)}
        return state

    @staticmethod
    def setstate(self, state):
        type_name = self.__class__.__name__

        if isinstance(state, tuple):
            state = state[1]
        if "fixed" not in state:
            state["fixed"] = False
        if type_name in ["ProductAnsatz", "SumAnsatz"]:
            if "names" not in state:
                state["names"] = [None for _ in state["ansatzes"]]
        if type_name in ["OpenString", "ClosedString"]:
            if "_weight_pool" not in state:
                state["_weight_pool"] = {}
            if "_delta_pool" not in state:
                state["_delta_pool"] = {}
        for key, value in state.items():
            setattr(self, key, value)

    def __new__(cls, name, bases, attrs):
        attrs["__setstate__"] = cls.setstate
        attrs["__getstate__"] = cls.getstate
        return type.__new__(cls, name, bases, attrs)


class AbstractAnsatz(metaclass=AbstractAnsatzMeta):

    __slots__ = ["owner", "fixed"]

    def __init__(self, owner):
        self.owner: AnsatzProductState = owner
        self.fixed = False

    def weight_and_delta(self, configurations, calculate_delta):
        """
        Calculate the weight and delta of the given configurations.

        The delta share the same structure and shape with the data stored in ansatz, which is usually a list of tensor.

        Parameters
        ----------
        configurations : list[Configuration]
            The given configuration list to calculate weight and delta.
        calculate_delta : bool
            Whether to calculate delta.

        Returns
        -------
        tuple[list[float | complex], None | list[Tensors]]
        """
        raise NotImplementedError("weight and delta not implemented")

    def refresh_auxiliaries(self):
        """
        Refresh auxiliaries after updating state.
        """
        raise NotImplementedError("refresh auxiliaries not implemented")

    def ansatz_prod_sum(self, a, b):
        """
        Calculate the summary of product of two tensors data. If None is given, the data ansatz itself stored will
        be used.

        Parameters
        ----------
        a, b : Tensors
            The two tensors data.

        Returns
        -------
        float | complex
            The dot of a and b.
        """
        raise NotImplementedError("ansatz_prod_sum not implemented")

    def ansatz_conjugate(self, a):
        """
        Calculate the conjugate of a tensors data, If None is given, the data ansatz itself stored will be used.

        Parameters
        ----------
        a : Tensors
            a tensors data.

        Returns
        -------
        Tensors
            The conjugate of the input.
        """
        raise NotImplementedError("ansatz_conjugate not implemented")

    def ansatz_dot(self, a, b):
        """
        Calculate the dot of two tensors data. If None is given, the data ansatz itself stored will be used.

        Parameters
        ----------
        a, b : Tensors
            The two tensors data.

        Returns
        -------
        float | complex
            The dot of a and b.
        """
        return self.ansatz_prod_sum(self.ansatz_conjugate(a), b).real

    def tensors(self, delta):
        """
        Get tensors of this ansatz.

        Parameters
        ----------
        delta : Iterator | Tensors

        Yields
        ------
        Tensor
            The tensors of this ansatz.
        """
        raise NotImplementedError("tensors not implemented")

    def elements(self, delta):
        """
        Get elements of this ansatz.

        Parameters
        ----------
        delta : Iterator | Tensors

        Yields
        ------
        float | complex
            The elements of this ansatz.
        """
        raise NotImplementedError("elements not implemented")

    def tensor_count(self, delta):
        """
        Get tensor count of this ansatz.

        Parameters
        ----------
        delta : Iterator | Tensors

        Returns
        -------
        int
            The tensor count of this ansatz.
        """
        raise NotImplementedError("tensor count not implemented")

    def element_count(self, delta):
        """
        Get element count of this ansatz.

        Parameters
        ----------
        delta : Iterator | Tensors

        Returns
        -------
        int
            The element count of this ansatz.
        """
        raise NotImplementedError("element count not implemented")

    def buffers(self, delta):
        """
        Get buffers of this ansatz, which can be used in mpi.

        Parameters
        ----------
        delta : Iterator | Tensors

        Yields
        ------
        Buffer
            The buffers of this ansatz.
        """
        raise NotImplementedError("buffers not implemented")

    def recovery_real(self, delta):
        """
        Recovery the input to real. If None is given, return True if and only if it is needed to reocovery to real data.

        This is used for real ansatz and complex state, the gradient calculated will be complex but real number is
        needed to be applied to the ansatz.

        Parameters
        ----------
        delta : Tensors
        """
        return delta

    def normalize_ansatz(self, log_ws=None):
        """
        Normalize this ansatz. If None is given, return the normalizable weight.
        """
        if log_ws is None:
            return 0

    def show(self):
        raise NotImplementedError("show not implemented")

    def lock(self, path=""):
        self.fixed = True

    def unlock(self, path=""):
        self.fixed = False
