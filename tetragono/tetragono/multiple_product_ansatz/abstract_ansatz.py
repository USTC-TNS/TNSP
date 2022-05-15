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


class AbstractAnsatz:

    __slots__ = []

    def weight(self, configuration):
        """
        Calculate the weight of the given configuration.

        Parameters
        ----------
        configuration : dict[tuple[int, int, int], int]
            The given configuration to calculate weight.

        Returns
        -------
        complex | float
            The result weight.
        """
        raise NotImplementedError("weight not implemented")

    def delta(self, configuration):
        """
        Calculate the delta of the given configuration.

        Parameters
        ----------
        configuration : dict[tuple[int, int, int], int]
            The given configuration to calculate delta.

        Returns
        -------
        Delta
            The delta object, which will be called by allreduce_delta and apply_gradient.
        """
        raise NotImplementedError("delta not implemented")

    @staticmethod
    def allreduce_delta(delta):
        """
        Allreduce the delta calculated by processes inplacely.

        Parameters
        ----------
        delta : Delta
            The delta calculated by this process.
        """
        raise NotImplementedError("allreduce delta not implemented")

    def apply_gradient(gradient, step_size, relative):
        """
        Apply the gradient to this subansatz.

        Parameters
        ----------
        gradient : Delta
            The gradient.
        step_size : float
            The step size.
        relative : bool
            use relative step size or not.
        """
        raise NotImplementedError("apply gradient not implemented")
