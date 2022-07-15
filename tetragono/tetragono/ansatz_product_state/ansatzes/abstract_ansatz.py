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
        configuration : list[list[dict[int, EdgePoint]]]
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
        configuration : list[list[dict[int, EdgePoint]]]
            The given configuration to calculate delta.

        Returns
        -------
        Delta
            The delta object, which will be called by allreduce and apply_gradient.
        """
        raise NotImplementedError("delta not implemented")

    def weight_and_delta(self, configurations, calculate_delta):
        """
        Calculate the weight and delta of the given configurations.

        Parameters
        ----------
        configurations : list[list[list[dict[int, EdgePoint]]]]
            The given configuration list to calculate weight and delta.
        calculate_delta : bool
            Whether to calculate delta.

        Returns
        -------
        tuple[list[float | complex], None | list[Delta]]
        """
        weight = [self.weight(configuration) for configuration in configurations]
        if calculate_delta:
            delta = [self.delta(configuration) for configuration in configurations]
        else:
            delta = None
        return weight, delta

    def get_norm_max(self, delta):
        """
        Get the max norm of the delta or state.

        Parameters
        ----------
        delta : None | Delta
            The delta or state to calculate max norm.

        Returns
        -------
            The max norm
        """
        raise NotImplementedError("get norm max not implemented")

    def refresh_auxiliaries(self):
        """
        Refresh auxiliaries after updating state.
        """
        raise NotImplementedError("refresh auxiliaries not implemented")

    @staticmethod
    def delta_dot_sum(a, b):
        """
        Calculate the dot of two delta.

        Parameters
        ----------
        a, b : Delta
            The two delta.

        Returns
        -------
        float
            The dot of a and b.
        """
        raise NotImplementedError("delta_dot_sum not implemented")

    @staticmethod
    def delta_update(a, b):
        """
        Add delta b into delta a.

        Parameters
        ----------
        a, b : Delta
            The two delta.
        """
        raise NotImplementedError("delta_update not implemented")

    def buffers(self, delta):
        """
        Get buffers of this ansatz.

        Yields
        ------
        buffer
            The buffers of this ansatz.
        """
        raise NotImplementedError("buffers not implemented")

    def elements(self, delta):
        """
        Get elements of this ansatz.

        Yields
        ------
        float | complex
            The elements of this ansatz.
        """
        raise NotImplementedError("elements not implemented")

    def buffer_count(self, delta):
        """
        Get buffer count of this ansatz.

        Returns
        -------
        int
            The buffer count of this ansatz.
        """
        raise NotImplementedError("buffer count not implemented")

    def element_count(self, delta):
        """
        Get element count of this ansatz.

        Returns
        -------
        int
            The element count of this ansatz.
        """
        raise NotImplementedError("element count not implemented")

    def buffers_for_mpi(self, delta):
        """
        Get buffers of this ansatz, which can be used in mpi.

        Yields
        ------
        buffer
            The buffers of this ansatz.
        """
        raise NotImplementedError("buffers for mpi not implemented")
